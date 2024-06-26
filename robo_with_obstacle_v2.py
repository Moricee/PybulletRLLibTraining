import os
import argparse
import random

import numpy as np
import matplotlib.pyplot as plt

import pybullet as p
import pybullet_data
import pyb_utils as putils

import gymnasium as gym
from gymnasium.spaces import Discrete

import ray
from ray import air, tune
from ray.rllib.env.env_context import EnvContext
from ray.rllib.utils.framework import try_import_tf, try_import_torch
from ray.rllib.utils.test_utils import check_learning_achieved

tf1, tf, tfv = try_import_tf()
torch, nn = try_import_torch()


class Constants:
    MAX_TIMESTEPS = 150000
    MAX_TIMESTEPS_EPISODE = 500
    MAX_ITERS = 500
    MAX_DISTANCE = 8

    GOAL_POSITION = [0, 10, 0.2]

    ROOM_SIZE = {"X": 400,
                 "Y": 700}

    MAX_FORCE_WHEELS = 100

    IMAGE_WIDTH = 84
    IMAGE_HEIGHT = 84

    WALL_MASS = 1000

    MAX_NUM_COLLISIONS = 1


parser = argparse.ArgumentParser()
parser.add_argument(
    "--run", type=str, default="DQN", help="The RLlib-registered algorithm to use."
)
parser.add_argument(
    "--framework",
    choices=["tf", "tf2", "torch"],
    default="torch",
    help="The DL framework specifier.",
)
parser.add_argument(
    "--as-test",
    action="store_true",
    help="Whether this script should be run as a test: --stop-reward must "
         "be achieved within --stop-timesteps AND --stop-iters.",
)
parser.add_argument(
    "--stop-iters", type=int, default=Constants.MAX_ITERS, help="Number of iterations to train."
)
parser.add_argument(
    "--stop-timesteps", type=int, default=Constants.MAX_TIMESTEPS, help="Number of timesteps to train."
)
parser.add_argument(
    "--stop-reward", type=float, default=70, help="Reward at which we stop training."
)
parser.add_argument(
    "--with-tune",
    action="store_true",
    help="Run without Tune using a manual train loop instead. In this case,"
         "use PPO without grid search and no TensorBoard.",
)
parser.add_argument(
    "--local-mode",
    action="store_true",
    help="Init Ray in local mode for easier debugging.",
)
parser.add_argument(
    "--restore-from-checkpoint",
    type=str,
    default="",
    help="Path from which to restore the Checkpoint"
)


class CorridorWithTurn(gym.Env):

    def __init__(self, env_config: EnvContext):
        self.end_pos = env_config["corridor_length"]
        self.client_id = p.connect(p.GUI)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())  # optionally
        p.setGravity(0, 0, -10)

        plane_id = p.loadURDF("plane.urdf", globalScaling=2, basePosition=[2, 0, 0])
        # plane_texture_id = p.loadTexture('textures/IllustrationForCube.bmp')
        # p.changeVisualShape(plane_id, -1, textureUniqueId=plane_texture_id)
        start_pos = [0, -5.3, 0.3]
        # p.getQuaternionFromAxisAngle(angle=np.pi/2, axis=[0,0,1])
        start_orientation_walls_vert = [np.sin(np.pi / 2), 1, 0, 0]
        start_orientation_walls_hori = [0, 0, 0, 1]
        start_orientation_robo = [0, 0, 0, 1]
        self.robo_id = p.loadURDF("r2d2/r2d2_short.urdf", start_pos, start_orientation_robo)
        self.num_joints = p.getNumJoints(self.robo_id)
        self.robo_fell_over = 0
        self.prev_distance = 0
        self.distance_to_goal = 0
        self.positions = []
        self.episode_reward = 0
        self.data = []
        self.force_left_wheels = 0  # Force applied to left wheels
        self.force_right_wheels = 0  # Force applied to right wheels
        self.passed_time_steps = 0
        self.reset_count = 0
        self.collision_count = 0
        # Arrays that will contain the last three frames captured by the robot's camera
        self.rgba0 = np.zeros((Constants.IMAGE_HEIGHT, Constants.IMAGE_WIDTH, 4), np.int32)
        self.rgba1 = np.zeros((Constants.IMAGE_HEIGHT, Constants.IMAGE_WIDTH, 4), np.int32)
        self.rgba2 = np.zeros((Constants.IMAGE_HEIGHT, Constants.IMAGE_WIDTH, 4), np.int32)
        print(self.num_joints)
        for jointIndex in range(0, self.num_joints):
            print(p.getJointInfo(self.robo_id, jointIndex))
        robo_base_box_index = 14

        # Creating the outer and inner walls of the testing grounds

        inner_wall_id_coll = p.createCollisionShape(p.GEOM_BOX, halfExtents=[5.9, 0.2, 1])
        inner_wall_id = p.createVisualShape(p.GEOM_BOX, halfExtents=[5.9, 0.2, 1], rgbaColor=[0, 0.5, 0.5, 1])

        outer_wall_id_coll = p.createCollisionShape(p.GEOM_BOX, halfExtents=[5.9, 0.2, 1])
        outer_wall_id = p.createVisualShape(p.GEOM_BOX, halfExtents=[5.9, 0.2, 1], rgbaColor=[0, 0.5, 0.5, 1])

        wall_ids = []

        wall_ids.append(p.createMultiBody(baseMass=Constants.WALL_MASS,
                                          baseCollisionShapeIndex=inner_wall_id_coll,
                                          baseVisualShapeIndex=inner_wall_id,
                                          basePosition=[-2, 0, 1],
                                          baseOrientation=start_orientation_walls_vert,
                                          useMaximalCoordinates=True))

        wall_ids.append(p.createMultiBody(baseMass=Constants.WALL_MASS,
                                          baseCollisionShapeIndex=inner_wall_id_coll,
                                          baseVisualShapeIndex=inner_wall_id,
                                          basePosition=[2, 0, 1],
                                          baseOrientation=start_orientation_walls_vert,
                                          useMaximalCoordinates=True))

        wall_ids.append(p.createMultiBody(baseMass=Constants.WALL_MASS,
                                          baseCollisionShapeIndex=outer_wall_id_coll,
                                          baseVisualShapeIndex=outer_wall_id,
                                          basePosition=[-6, 0, 1],
                                          baseOrientation=start_orientation_walls_vert,
                                          useMaximalCoordinates=True))

        wall_ids.append(p.createMultiBody(baseMass=Constants.WALL_MASS,
                                          baseCollisionShapeIndex=outer_wall_id_coll,
                                          baseVisualShapeIndex=outer_wall_id,
                                          basePosition=[6, 0, 1],
                                          baseOrientation=start_orientation_walls_vert,
                                          useMaximalCoordinates=True))

        wall_ids.append(p.createMultiBody(baseMass=Constants.WALL_MASS,
                                          baseCollisionShapeIndex=outer_wall_id_coll,
                                          baseVisualShapeIndex=outer_wall_id,
                                          basePosition=[0, -6.1, 1],
                                          baseOrientation=start_orientation_walls_hori,
                                          useMaximalCoordinates=True))

        wall_ids.append(p.createMultiBody(baseMass=Constants.WALL_MASS,
                                          baseCollisionShapeIndex=outer_wall_id_coll,
                                          baseVisualShapeIndex=outer_wall_id,
                                          basePosition=[0, 6.1, 1],
                                          baseOrientation=start_orientation_walls_hori,
                                          useMaximalCoordinates=True))

        # Creating the obstacles consisting of three cubes and three cylinders
        obstacle_cube_id_coll = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.5, 0.5, 0.5])
        obstacle_cube_id = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.5, 0.5, 0.5], rgbaColor=[0.5, 0, 0.2, 1])

        obstacle_cyl_id_coll = p.createCollisionShape(p.GEOM_CYLINDER, radius=0.25)
        obstacle_cyl_id = p.createVisualShape(p.GEOM_CYLINDER, radius=0.25, rgbaColor=[0, 0.5, 0.5, 1])

        obstacle_ids = []
        '''
        obstacle_ids.append(p.createMultiBody(baseMass=WALL_MASS,
                                              baseCollisionShapeIndex=obstacle_cube_id_coll,
                                              baseVisualShapeIndex=obstacle_cube_id,
                                              basePosition=[0.5, 0.5, 0.6],
                                              baseOrientation=start_orientation_walls_hori,
                                              useMaximalCoordinates=True))

        obstacle_ids.append(p.createMultiBody(baseMass=WALL_MASS,
                                              baseCollisionShapeIndex=obstacle_cube_id_coll,
                                              baseVisualShapeIndex=obstacle_cube_id,
                                              basePosition=[1, -2, 0.6],
                                              baseOrientation=start_orientation_walls_hori,
                                              useMaximalCoordinates=True))

        obstacle_ids.append(p.createMultiBody(baseMass=WALL_MASS,
                                              baseCollisionShapeIndex=obstacle_cube_id_coll,
                                              baseVisualShapeIndex=obstacle_cube_id,
                                              basePosition=[-0.8, 3, 0.6],
                                              baseOrientation=start_orientation_walls_hori,
                                              useMaximalCoordinates=True))

        

        obstacle_ids.append(p.createMultiBody(baseMass=WALL_MASS,
                                              baseCollisionShapeIndex=obstacle_cyl_id_coll,
                                              baseVisualShapeIndex=obstacle_cyl_id,
                                              basePosition=[-1, -2, 0.6],
                                              baseOrientation=start_orientation_walls_hori,
                                              useMaximalCoordinates=True))

        obstacle_ids.append(p.createMultiBody(baseMass=WALL_MASS,
                                              baseCollisionShapeIndex=obstacle_cyl_id_coll,
                                              baseVisualShapeIndex=obstacle_cyl_id,
                                              basePosition=[0.2, 3, 0.6],
                                              baseOrientation=start_orientation_walls_hori,
                                              useMaximalCoordinates=True))
                                              
        '''
        obstacle_ids.append(p.createMultiBody(baseMass=Constants.WALL_MASS,
                                              baseCollisionShapeIndex=obstacle_cyl_id_coll,
                                              baseVisualShapeIndex=obstacle_cyl_id,
                                              basePosition=[0, -3, 0.5],
                                              baseOrientation=start_orientation_walls_hori,
                                              useMaximalCoordinates=True))

        # Creating a wall behind the ball so the robo can't drive too far behind it
        obstacle_wall_id_coll = p.createCollisionShape(p.GEOM_BOX, halfExtents=[1.8, 0.5, 1])
        obstacle_wall_id = p.createVisualShape(p.GEOM_BOX, halfExtents=[1.8, 0.5, 1], rgbaColor=[0.5, 0, 0.5, 1])

        obstacle_ids.append(p.createMultiBody(baseMass=Constants.WALL_MASS,
                                              baseCollisionShapeIndex=obstacle_wall_id_coll,
                                              baseVisualShapeIndex=obstacle_wall_id,
                                              basePosition=[0, 0.2, 1],
                                              baseOrientation=start_orientation_walls_hori,
                                              useMaximalCoordinates=True))

        collision_pairs = []
        for i in range(0, len(wall_ids)):
            collision_pairs.append((self.robo_id, wall_ids[i]))

        for i in range(0, len(obstacle_ids)):
            collision_pairs.append((self.robo_id, obstacle_ids[i]))

        self.collision_detector_avoid = putils.collision.CollisionDetector(self.client_id, collision_pairs)

        goal_box_id_coll = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.3, 0.3, 0.3])
        goal_box_id = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.3, 0.3, 0.3], rgbaColor=[0, 0, 0, 1])

        self.goal_ball_body_id = p.createMultiBody(baseMass=p.GEOM_BOX,
                                                   baseCollisionShapeIndex=goal_box_id_coll,
                                                   baseVisualShapeIndex=goal_box_id,
                                                   basePosition=[0, -1, 0.3],
                                                   baseOrientation=start_orientation_walls_hori,
                                                   useMaximalCoordinates=True)

        self.collision_detector_goal = putils.CollisionDetector(self.client_id,
                                                                [(self.robo_id, self.goal_ball_body_id)])

        self.goal_pos, goal_orientation = p.getBasePositionAndOrientation(self.goal_ball_body_id)
        self.goal_pos = np.array(self.goal_pos)

        robo_pos, robo_orn = p.getBasePositionAndOrientation(self.robo_id)
        robo_pos = np.array(robo_pos)
        self.initial_distance = np.linalg.norm(self.goal_pos - robo_pos)
        print("Initial distance to goal:", self.initial_distance)
        self.prev_distance = self.initial_distance

        self.above_camera = putils.Camera.from_camera_position(
            camera_position=(0, 0.1, 10),
            target_position=(0, 0, 0),
            near=0.1,
            far=20,
            width=100,
            height=100,
        )

        # Set the position of the above camera
        self.above_camera.set_camera_pose([0, 0.1, 10], [0, 0, 0])

        # Get the state of the robot's head link
        head_link_state = p.getLinkState(self.robo_id, 13, computeForwardKinematics=True)
        link_position = np.array(head_link_state[0])
        link_orientation = head_link_state[1]

        # Calculate the camera position and target position
        point_in_front_of_camera = putils.quaternion_rotate(link_orientation, np.array([0, 0.2, 0.4]))
        camera_position = link_position + point_in_front_of_camera
        target_position = camera_position + point_in_front_of_camera

        self.robo_camera = putils.Camera.from_distance_rpy(target_position=target_position, distance=0.2, near=0.001,
                                                           far=20, width=Constants.IMAGE_WIDTH,
                                                           height=Constants.IMAGE_HEIGHT, fov=90)
        self.robo_camera.set_camera_pose(camera_position, target_position)
        rgba, depth, seg = self.robo_camera.get_frame()
        self.rgba = np.array(rgba)
        self.depth = np.array(depth)
        print(self.depth.shape)
        self.robo_pos, self.robo_orn = p.getBasePositionAndOrientation(self.robo_id)
        # save the frame
        self.above_camera.save_frame("frame.png", rgba=rgba)

        self.action_space = Discrete(3)

        spaces = {}
        spaces["passed_time_steps"] = gym.spaces.Box(low=np.array(0),
                                                     high=np.array(Constants.MAX_TIMESTEPS_EPISODE + 1),
                                                     dtype=np.int32)
        # Define the observation spaces
        spaces["rgba0"] = gym.spaces.Box(low=np.full((Constants.IMAGE_HEIGHT, Constants.IMAGE_WIDTH, 4), 0),
                                         high=np.full((Constants.IMAGE_HEIGHT, Constants.IMAGE_WIDTH, 4), 255 + 1),
                                         dtype=np.int32)
        spaces["rgba1"] = gym.spaces.Box(low=np.full((Constants.IMAGE_HEIGHT, Constants.IMAGE_WIDTH, 4), 0),
                                         high=np.full((Constants.IMAGE_HEIGHT, Constants.IMAGE_WIDTH, 4), 255 + 1),
                                         dtype=np.int32)
        spaces["rgba2"] = gym.spaces.Box(low=np.full((Constants.IMAGE_HEIGHT, Constants.IMAGE_WIDTH, 4), 0),
                                         high=np.full((Constants.IMAGE_HEIGHT, Constants.IMAGE_WIDTH, 4), 255 + 1),
                                         dtype=np.int32)

        self.observation_space = gym.spaces.Dict(spaces)

        self.initial_state = p.saveState()
        # Set the seed. This is only used for the final (reach goal) reward.
        self.reset(seed=env_config.worker_index * env_config.num_workers)

    def get_observation_vector(self):
        obs = {}
        obs["passed_time_steps"] = np.array(self.passed_time_steps, dtype=np.int32)
        # Shift the old frames
        self.rgba0 = np.array(self.rgba1)
        self.rgba1 = np.array(self.rgba2)

        # Store the current frame
        self.rgba2 = np.array(self.rgba)

        # Add the frames to the observation vector
        obs["rgba0"] = self.rgba0
        obs["rgba1"] = self.rgba1
        obs["rgba2"] = self.rgba2

        # Return the observation vector
        return obs

    def reset(self, *, seed=None, options=None):
        random.seed(seed)
        p.restoreState(self.initial_state)
        lines = ['Readme', 'How to write text files in Python']
        with open('positions.txt', 'a') as f:
            f.write(str(self.positions))
            f.write('\n')

        with open('rewards.txt', 'a') as f:
            f.write(str(self.episode_reward))
            f.write('\n')

        self.positions = []
        self.force_left_wheels = 0
        self.force_right_wheels = 0
        self.collision_count = 0
        self.passed_time_steps = 0
        self.robo_fell_over = 0
        self.distance_to_goal = self.prev_distance = self.initial_distance
        self.robo_pos, self.robo_orn = p.getBasePositionAndOrientation(self.robo_id)
        self.rgba, self.depth, seg = self.robo_camera.get_frame()
        self.positions.append([self.robo_pos[0], self.robo_pos[1]])
        self.episode_reward = 0
        # p.resetBasePositionAndOrientation(self.goal_ball_body_id, [(random.random() - 0.5) * 3, -1, 0.6],[0, 0, 0, 1])
        obs = self.get_observation_vector()
        self.reset_count += 1
        return obs, {}

    def step(self, action):
        reward = 0
        done = truncated = False

        assert action in [0, 1, 2], action
        if action == 0:
            # forward with force 80
            self.force_right_wheels = -80
            self.force_left_wheels = -80
        elif action == 1:
            # turn right
            self.force_right_wheels = 0
            self.force_left_wheels = -50
        elif action == 2:
            # turn left
            self.force_right_wheels = -50
            self.force_left_wheels = 0

        p.setJointMotorControlArray(bodyIndex=self.robo_id,
                                    jointIndices=[2, 3, 6, 7],
                                    controlMode=p.VELOCITY_CONTROL,
                                    targetVelocities=[self.force_right_wheels, self.force_right_wheels,
                                                      self.force_left_wheels, self.force_left_wheels],
                                    forces=[Constants.MAX_FORCE_WHEELS, Constants.MAX_FORCE_WHEELS,
                                            Constants.MAX_FORCE_WHEELS,
                                            Constants.MAX_FORCE_WHEELS])

        self.prev_distance = np.array(self.distance_to_goal)
        for i in range(24):
            p.stepSimulation()

            # Check if the agent made the robot collide with a wall or obstacle and reset the simulation if that happened
            if self.collision_detector_avoid.in_collision():
                print("Collision detected!")
                self.collision_count += 1
                reward = -1
                truncated = True
                break

            # Check if the agent reached the goal
            if self.collision_detector_goal.in_collision():
                print("Goal Reached!")
                reward = 50
                done = True
                break

        self.robo_pos, self.robo_orn = p.getBasePositionAndOrientation(self.robo_id)
        self.positions.append([self.robo_pos[0], self.robo_pos[1]])

        head_link_state = p.getLinkState(self.robo_id, 13, computeForwardKinematics=True)
        head_link_position = np.array(head_link_state[0])
        head_link_orientation = head_link_state[1]

        camera_position, target_position = calculate_camera_position_and_orientation(head_link_orientation,
                                                                                     head_link_position)

        self.robo_camera.set_camera_pose(camera_position, target_position)

        self.rgba, self.depth, seg = self.robo_camera.get_frame()

        # Reward the agent for getting closer to the goal
        self.distance_to_goal = np.linalg.norm(self.goal_pos - self.robo_pos)
        if self.distance_to_goal < self.prev_distance:
            reward += 0.1

            for i in range(len(self.rgba)):
                for j in range(len(self.rgba[i])):
                    if self.rgba[i][j][2] <= 0 and self.rgba[i][j][0] <= 0 >= self.rgba[i][j][1]:
                        # print(self.rgba[i][j][2], self.rgba[i][j][0], self.rgba[i][j][1])
                        # print('black pixel spotted')
                        reward += 0.00005

        # Calculate a factor that can be used to calculate a reward depending on how close the agent is to the goal
        reward_factor = self.distance_to_goal / self.initial_distance

        # Reset the simulation when the agent hasn't reached the goal in a certain time frame
        if self.passed_time_steps >= Constants.MAX_TIMESTEPS_EPISODE:
            print("Didn't reach goal. Distance left:", self.distance_to_goal)
            reward = -1
            truncated = True

        # Reset the simulation when the agent makes the robot fall over
        if head_link_position[2] < 0.3:
            print("Robo fell over.")
            self.robo_fell_over = 1
            reward = -1
            truncated = True

        if self.passed_time_steps % 500 == 0:
            rgba, depth, seg = self.above_camera.get_frame()

            # save the frame
            self.above_camera.save_frame("frame.png", rgba=rgba)

        self.passed_time_steps += 1
        self.episode_reward += reward
        obs = self.get_observation_vector()
        # print(obs[0:15])

        return (
            obs,
            reward,
            done,
            truncated,
            {},
        )


def calculate_camera_position_and_orientation(head_link_orientation, head_link_position):
    # Calculates the position and orientation for the front camera of the robot
    # Calculate the offset for the camera and the target position
    point_in_front_of_camera = putils.quaternion_rotate(head_link_orientation, np.array([0, 0.2, 0]))

    # Calculate the camera position and target position
    camera_position = head_link_position + point_in_front_of_camera
    target_position = camera_position + point_in_front_of_camera

    return camera_position, target_position


def plt_image(img):
    plt.axis('off')
    plt.imshow(img)
    plt.show()


from ray.tune.logger import pretty_print
from ray.rllib.algorithms.dqn import DQNConfig

if __name__ == '__main__':

    args = parser.parse_args()
    print(f"Running with following CLI options: {args}")

    ray.init(local_mode=args.local_mode)

    # Can also register the env creator function explicitly with:
    # register_env("corridor", lambda config: SimpleCorridor(config))

    replay_config = {
        "type": "MultiAgentPrioritizedReplayBuffer",
        "storage_unit": "timesteps",
        "capacity": 30000,
        "prioritized_replay_alpha": 1,
        # Beta parameter for sampling from prioritized replay buffer.
        "prioritized_replay_beta": 0.4,
        # Epsilon to add to the TD errors when updating priorities.
        "prioritized_replay_eps": 1e-6,
        # The number of continuous environment steps to replay at once. This may
        # be set to greater than 1 to support recurrent models.
        "replay_sequence_length": 1,
    }

    exploration_config = {
        # Exploration subclass by name or full path to module+class
        # (e.g. “ray.rllib.utils.exploration.epsilon_greedy.EpsilonGreedy”)
        "type": "EpsilonGreedy",
        # Parameters for the Exploration class' constructor:
        "initial_epsilon": 1.0,
        "final_epsilon": 0.02,
        "epsilon_timesteps": 50000,  # Timesteps over which to anneal epsilon.
    }

    config = (
        DQNConfig(args.run)
        .environment(CorridorWithTurn, env_config={"corridor_length": 350})
        .framework(args.framework)
        .rollouts(num_rollout_workers=0, create_env_on_local_worker=True)
        # Use GPUs iff `RLLIB_NUM_GPUS` env var set to > 0.
        .resources(num_gpus=int(os.environ.get("RLLIB_NUM_GPUS", "0")))
        .training(double_q=False, lr_schedule=[[0, 1e-4], [50000, 1e-6]], v_min=-30, v_max=100, noisy=False,
                  replay_buffer_config=replay_config)
        .exploration(explore=True, exploration_config=exploration_config)
        # lr=7e-5,
        #
    )
    config.gamma = 0.999

    stop = {
        "training_iteration": args.stop_iters,
        "timesteps_total": args.stop_timesteps,
        "episode_reward_mean": args.stop_reward,
    }

    if args.with_tune:

        # automated run with Tune and grid search and TensorBoard
        print("Training automatically with Ray Tune")
        tuner = tune.Tuner(
            args.run,
            param_space=config.to_dict(),
            run_config=air.RunConfig(stop=stop),
        )
        results = tuner.fit()

        if args.as_test:
            print("Checking if learning goals were achieved")
            check_learning_achieved(results, args.stop_reward)

    else:
        print("Running manual train loop without Ray Tune.")
        # use fixed learning rate instead of grid search (needs tune)
        # config.lr = 1e-3
        algo = config.build()

        if args.restore_from_checkpoint != "":
            algo.restore(args.restore_from_checkpoint)
            result = algo.train()
            args.stop_iters += result["training_iteration"]
            args.stop_timesteps += result["timesteps_total"]

        # run manual training loop and print results after each iteration
        steps = 0
        for _ in range(args.stop_iters):
            result = algo.train()
            print("Current Exploration Epsilon:", algo.get_policy().get_exploration_state()['cur_epsilon'])
            print(pretty_print(result))
            # stop training of the target train steps or reward are reached
            if (
                    result["timesteps_total"] >= args.stop_timesteps
                    or result["episode_reward_mean"] >= args.stop_reward
            ):
                break

        save_result = algo.save()
        path_to_checkpoint = save_result.checkpoint.path
        print(
            "An Algorithm checkpoint has been created inside directory: "
            f"'{path_to_checkpoint}'."
        )
        algo.stop()

    ray.shutdown()

    # Moving the agent's head to the target position

    # Receiving and applying commands to the agent

    # Storing the position of the agent's head

    # Assigning a reward to the agent

    # Converting the positions list to a numpy array and printing it
