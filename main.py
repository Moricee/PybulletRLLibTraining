import os

import pybullet as p
import pyb_utils as putils
import time
import pybullet_data
import numpy as np
import gymnasium as gym
import argparse
import random
from gymnasium.spaces import Discrete, Box
import ray
from ray import air, tune
from ray.rllib.algorithms import Algorithm
from ray.rllib.env.env_context import EnvContext
from ray.rllib.utils.framework import try_import_tf, try_import_torch
from ray.rllib.utils.test_utils import check_learning_achieved
import matplotlib.pyplot as plt

MAX_TIMESTEPS = 500000
MAX_TIMESTEPS_EPISODE = 1000
MAX_ITERS = 500

GOAL_COORDINATES_X = np.array([160, 180])  # horizontal span of the goal area
GOAL_COORDINATES_Y = np.array([230, 250])  # vertical span of the goal area

ROOM_SIZE = {"X": 400,
             "Y": 700}

MAX_FORCE_WHEELS = 100

IMAGE_WIDTH = 84
IMAGE_HEIGHT = 84

WALL_MASS = 1000

MAX_NUM_COLLISIONS = 1

OBSERVATION_SPACE_SIZE = 4 + 3 + 1 + 1 + 1 + 1 + 1 + 1 + 1 + IMAGE_WIDTH * IMAGE_HEIGHT + IMAGE_WIDTH * IMAGE_HEIGHT * 4

tf1, tf, tfv = try_import_tf()
torch, nn = try_import_torch()

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
    "--stop-iters", type=int, default=MAX_ITERS, help="Number of iterations to train."
)
parser.add_argument(
    "--stop-timesteps", type=int, default=MAX_TIMESTEPS, help="Number of timesteps to train."
)
parser.add_argument(
    "--stop-reward", type=float, default=30, help="Reward at which we stop training."
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
        self.client_id = p.connect(p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())  # optionally
        p.setGravity(0, 0, -10)
        plane_id = p.loadURDF("plane.urdf")
        start_pos = [0, -5.3, 0.3]
        # p.getQuaternionFromAxisAngle(angle=np.pi/2, axis=[0,0,1])
        start_orientation_walls_vert = [np.sin(np.pi / 2), 1, 0, 0]
        start_orientation_walls_hori = [0, 0, 0, 1]
        start_orientation_robo = [0, 0, 0, 1]
        self.robo_id = p.loadURDF("r2d2/r2d2_short.urdf", start_pos, start_orientation_robo)
        self.num_joints = p.getNumJoints(self.robo_id)
        self.collision_count = 0
        self.robo_fell_over = 0
        self.prev_distance = 0
        self.distance_to_goal = 0

        print(self.num_joints)
        for jointIndex in range(0, self.num_joints):
            print(p.getJointInfo(self.robo_id, jointIndex))
        robo_base_box_index = 14

        # Creating the outer and inner walls of the testing grounds

        inner_wall_id_coll = p.createCollisionShape(p.GEOM_BOX, halfExtents=[5, 0.2, 1])
        inner_wall_id = p.createVisualShape(p.GEOM_BOX, halfExtents=[5.9, 0.2, 1])

        outer_wall_id_coll = p.createCollisionShape(p.GEOM_BOX, halfExtents=[6, 0.2, 1])
        outer_wall_id = p.createVisualShape(p.GEOM_BOX, halfExtents=[6, 0.2, 1])

        wall_ids = []

        wall_ids.append(p.createMultiBody(baseMass=WALL_MASS,
                                          baseInertialFramePosition=[0, 0, 0],
                                          baseCollisionShapeIndex=inner_wall_id_coll,
                                          baseVisualShapeIndex=inner_wall_id,
                                          basePosition=[-2, 0, 1],
                                          baseOrientation=start_orientation_walls_vert,
                                          useMaximalCoordinates=True))

        wall_ids.append(p.createMultiBody(baseMass=WALL_MASS,
                                          baseInertialFramePosition=[0, 0, 0],
                                          baseCollisionShapeIndex=inner_wall_id_coll,
                                          baseVisualShapeIndex=inner_wall_id,
                                          basePosition=[2, 0, 1],
                                          baseOrientation=start_orientation_walls_vert,
                                          useMaximalCoordinates=True))

        wall_ids.append(p.createMultiBody(baseMass=WALL_MASS,
                                          baseInertialFramePosition=[0, 0, 0],
                                          baseCollisionShapeIndex=outer_wall_id_coll,
                                          baseVisualShapeIndex=outer_wall_id,
                                          basePosition=[-6, 0, 1],
                                          baseOrientation=start_orientation_walls_vert,
                                          useMaximalCoordinates=True))

        wall_ids.append(p.createMultiBody(baseMass=WALL_MASS,
                                          baseInertialFramePosition=[0, 0, 0],
                                          baseCollisionShapeIndex=outer_wall_id_coll,
                                          baseVisualShapeIndex=outer_wall_id,
                                          basePosition=[6, 0, 1],
                                          baseOrientation=start_orientation_walls_vert,
                                          useMaximalCoordinates=True))

        wall_ids.append(p.createMultiBody(baseMass=WALL_MASS,
                                          baseInertialFramePosition=[0, 0, 0],
                                          baseCollisionShapeIndex=outer_wall_id_coll,
                                          baseVisualShapeIndex=outer_wall_id,
                                          basePosition=[0, -6, 1],
                                          baseOrientation=start_orientation_walls_hori,
                                          useMaximalCoordinates=True))

        wall_ids.append(p.createMultiBody(baseMass=WALL_MASS,
                                          baseInertialFramePosition=[0, 0, 0],
                                          baseCollisionShapeIndex=outer_wall_id_coll,
                                          baseVisualShapeIndex=outer_wall_id,
                                          basePosition=[0, 6, 1],
                                          baseOrientation=start_orientation_walls_hori,
                                          useMaximalCoordinates=True))

        # Creating the obstacles consisting of three cubes and three cylinders
        obstacle_cube_id_coll = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.5, 0.5, 0.5])
        obstacle_cube_id = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.5, 0.5, 0.5])

        obstacle_cyl_id_coll = p.createCollisionShape(p.GEOM_CYLINDER, radius=0.3)
        obstacle_cyl_id = p.createVisualShape(p.GEOM_CYLINDER, radius=0.3)

        obstacle_ids = []
        '''
        obstacle_ids.append(p.createMultiBody(baseMass=WALL_MASS,
                                              baseInertialFramePosition=[0, 0, 0],
                                              baseCollisionShapeIndex=obstacle_cube_id_coll,
                                              baseVisualShapeIndex=obstacle_cube_id,
                                              basePosition=[0.5, 0.5, 0.6],
                                              baseOrientation=start_orientation_walls_hori,
                                              useMaximalCoordinates=True))

        obstacle_ids.append(p.createMultiBody(baseMass=WALL_MASS,
                                              baseInertialFramePosition=[0, 0, 0],
                                              baseCollisionShapeIndex=obstacle_cube_id_coll,
                                              baseVisualShapeIndex=obstacle_cube_id,
                                              basePosition=[1, -2, 0.6],
                                              baseOrientation=start_orientation_walls_hori,
                                              useMaximalCoordinates=True))

        obstacle_ids.append(p.createMultiBody(baseMass=WALL_MASS,
                                              baseInertialFramePosition=[0, 0, 0],
                                              baseCollisionShapeIndex=obstacle_cube_id_coll,
                                              baseVisualShapeIndex=obstacle_cube_id,
                                              basePosition=[-0.8, 3, 0.6],
                                              baseOrientation=start_orientation_walls_hori,
                                              useMaximalCoordinates=True))

        obstacle_ids.append(p.createMultiBody(baseMass=WALL_MASS,
                                              baseInertialFramePosition=[0, 0, 0],
                                              baseCollisionShapeIndex=obstacle_cyl_id_coll,
                                              baseVisualShapeIndex=obstacle_cyl_id,
                                              basePosition=[0, -4, 0.6],
                                              baseOrientation=start_orientation_walls_hori,
                                              useMaximalCoordinates=True))

        obstacle_ids.append(p.createMultiBody(baseMass=WALL_MASS,
                                              baseInertialFramePosition=[0, 0, 0],
                                              baseCollisionShapeIndex=obstacle_cyl_id_coll,
                                              baseVisualShapeIndex=obstacle_cyl_id,
                                              basePosition=[-1, -2, 0.6],
                                              baseOrientation=start_orientation_walls_hori,
                                              useMaximalCoordinates=True))

        obstacle_ids.append(p.createMultiBody(baseMass=WALL_MASS,
                                              baseInertialFramePosition=[0, 0, 0],
                                              baseCollisionShapeIndex=obstacle_cyl_id_coll,
                                              baseVisualShapeIndex=obstacle_cyl_id,
                                              basePosition=[0.2, 3, 0.6],
                                              baseOrientation=start_orientation_walls_hori,
                                              useMaximalCoordinates=True))
        '''
        collision_pairs = []
        for i in range(0, len(wall_ids)):
            collision_pairs.append((self.robo_id, wall_ids[i]))

        for i in range(0, len(obstacle_ids)):
            collision_pairs.append((self.robo_id, obstacle_ids[i]))

        self.collision_detector_avoid = putils.collision.CollisionDetector(self.client_id, collision_pairs)

        goal_ball_id_coll = p.createCollisionShape(p.GEOM_SPHERE, radius=0.3)
        goal_ball_id = p.createVisualShape(p.GEOM_SPHERE, radius=0.3, rgbaColor=[0, 0, 1, 1])

        goal_ball_body_id = p.createMultiBody(baseMass=p.GEOM_SPHERE,
                                              baseInertialFramePosition=[0, 0, 0],
                                              baseCollisionShapeIndex=goal_ball_id_coll,
                                              baseVisualShapeIndex=goal_ball_id,
                                              basePosition=[0, 0, 0.6],
                                              baseOrientation=start_orientation_walls_hori,
                                              useMaximalCoordinates=True)

        self.collision_detector_goal = putils.CollisionDetector(self.client_id, [(self.robo_id, goal_ball_body_id)])

        self.goal_pos, goal_orientation = p.getBasePositionAndOrientation(goal_ball_body_id)
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
        self.above_camera.set_camera_pose([0, 0.1, 10], [0, 0, 0])

        head_link_state = p.getLinkState(self.robo_id, 13, computeForwardKinematics=True)
        link_position = np.array(head_link_state[0])
        link_orientation = head_link_state[1]

        camera_position = np.array(link_position)
        camera_position -= link_position
        camera_position = putils.quaternion_rotate(link_orientation, camera_position)
        point_in_front_of_camera = np.array(putils.quaternion_rotate(link_orientation, np.array([0, 0.2, 0])))
        camera_position += point_in_front_of_camera
        camera_position += link_position

        # print("Camera position:", camera_position)

        target_position = np.array(camera_position)
        target_position += point_in_front_of_camera

        # print("Camera and target position:", camera_position, target_position)

        self.robo_camera = putils.Camera.from_camera_position(
            camera_position=camera_position,
            target_position=target_position,
            near=0.1,
            far=20,
            width=IMAGE_WIDTH,
            height=IMAGE_HEIGHT,
        )
        self.robo_camera.set_camera_pose(camera_position, target_position)

        rgba, depth, seg = self.robo_camera.get_frame()
        self.rgba = np.array(rgba)
        self.depth = np.array(depth)
        print(self.depth.shape)
        self.robo_pos, self.robo_orn = p.getBasePositionAndOrientation(self.robo_id)
        # save the frame
        self.above_camera.save_frame("frame.png", rgba=rgba)

        self.data = []

        self.force_left_wheels = 0  # Force applied to left wheels
        self.force_right_wheels = 0  # Force applied to right wheels

        self.positions = []
        self.action_space = Discrete(5)
        self.passed_time_steps = 0

        spaces = {}
        spaces["orientation"] = gym.spaces.Box(low=np.full(4, -1), high=np.full(4, 1), dtype=np.float64)
        spaces["position"] = gym.spaces.Box(low=np.full(3, -6), high=np.full(3, 6), dtype=np.float64)
        spaces["wheels_force"] = gym.spaces.Box(low=np.full(2, -MAX_FORCE_WHEELS), high=np.full(2, MAX_FORCE_WHEELS), dtype=np.float64)
        spaces["collisions"] = gym.spaces.Box(low=np.array(0), high=np.array(MAX_NUM_COLLISIONS), dtype=np.int32)
        spaces["robo_fell_over"] = gym.spaces.Box(low=np.array(0), high=np.array(1), dtype=np.int32)
        spaces["distance_to_goal"] = gym.spaces.Box(low=np.array(-self.initial_distance * 1.5), high=np.array(self.initial_distance * 1.5), dtype=np.float64)
        spaces["passed_time_steps"] = gym.spaces.Box(low=np.array(0), high=np.array(MAX_TIMESTEPS_EPISODE), dtype=np.int32)
        spaces["rgba"] = gym.spaces.Box(low=np.full((IMAGE_HEIGHT, IMAGE_WIDTH, 4), 0), high=np.full((IMAGE_HEIGHT, IMAGE_WIDTH, 4), 255), dtype=np.float64)
        spaces["depth"] = gym.spaces.Box(low=np.full((IMAGE_HEIGHT, IMAGE_WIDTH), 0), high=np.full((IMAGE_HEIGHT, IMAGE_WIDTH), 1), dtype=np.float64)

        self.observation_space = gym.spaces.Dict(spaces)

        self.initial_state = p.saveState()
        # Set the seed. This is only used for the final (reach goal) reward.
        self.reset(seed=env_config.worker_index * env_config.num_workers)

    def get_observation_vector(self):
        obs = {}
        obs["orientation"] = np.array(self.robo_orn)
        obs["position"] = np.array(self.robo_pos)
        obs["wheels_force"] = np.array([self.force_left_wheels, self.force_right_wheels], dtype=np.float64)
        obs["collisions"] = np.array(self.collision_count, dtype=np.int32)
        obs["robo_fell_over"] = np.array(self.robo_fell_over, dtype=np.int32)
        obs["distance_to_goal"] = np.array(self.distance_to_goal, dtype=np.float64)
        obs["passed_time_steps"] = np.array(self.passed_time_steps, dtype=np.int32)
        obs["rgba"] = np.array(self.rgba, dtype=np.float64)
        obs["depth"] = np.array(self.depth, dtype=np.float64)
        #print(obs)
        return obs

    def reset(self, *, seed=None, options=None):
        random.seed(seed)
        p.restoreState(self.initial_state)
        self.force_left_wheels = 0
        self.force_right_wheels = 0
        self.collision_count = 0
        self.passed_time_steps = 0
        self.robo_fell_over = 0
        self.distance_to_goal = self.prev_distance = self.initial_distance
        self.robo_pos, self.robo_orn = p.getBasePositionAndOrientation(self.robo_id)
        self.rgba, self.depth, seg = self.robo_camera.get_frame()

        obs = self.get_observation_vector()

        return obs, {}

    def step(self, action):
        reward = 0
        done = truncated = False
        self.robo_pos, self.robo_orn = p.getBasePositionAndOrientation(self.robo_id)
        assert action in [0, 1, 2, 3, 4], action
        if action == 0:
            self.force_right_wheels = 100
            self.force_left_wheels = 100
        elif action == 1:
            self.force_right_wheels = -100
            self.force_left_wheels = -100
        elif action == 2:
            self.force_right_wheels = 50
            self.force_left_wheels = -50
        elif action == 3:
            self.force_right_wheels = -50
            self.force_left_wheels = 50
        elif action == 4:
            self.force_right_wheels = 0
            self.force_left_wheels = 0

        p.setJointMotorControlArray(bodyIndex=self.robo_id,
                                    jointIndices=[2, 3, 6, 7],
                                    controlMode=p.VELOCITY_CONTROL,
                                    targetVelocities=[self.force_right_wheels, self.force_right_wheels,
                                                      self.force_left_wheels, self.force_left_wheels],
                                    forces=[MAX_FORCE_WHEELS, MAX_FORCE_WHEELS, MAX_FORCE_WHEELS,
                                            MAX_FORCE_WHEELS])

        self.prev_distance = np.array(self.distance_to_goal)
        # print(1, "step")
        for i in range(4):
            p.stepSimulation()
            # print(10, "steps")

        head_link_state = p.getLinkState(self.robo_id, 13, computeForwardKinematics=True)
        head_link_position = np.array(head_link_state[0])
        head_link_orientation = head_link_state[1]

        camera_position, target_position = self.calculate_camera_position_and_orientation(head_link_orientation,
                                                                                          head_link_position)

        self.robo_camera.set_camera_pose(camera_position, target_position)

        self.rgba, self.depth, seg = self.robo_camera.get_frame()
        # save the frame
        # self.robo_camera.save_frame("frame.png", rgba=self.rgba)

        self.distance_to_goal = np.linalg.norm(self.goal_pos - self.robo_pos)

        # Reward the agent for getting closer to the goal
        if self.prev_distance > self.distance_to_goal:
            reward = 0.1

        # Calculate a factor that can be used to calculate a reward depending on how close the agent is to the goal
        reward_factor = self.distance_to_goal / self.initial_distance

        # Reset the simulation when the agent hasn't reached the goal in a certain time frame
        if self.passed_time_steps >= MAX_TIMESTEPS_EPISODE:
            print("Didn't reach goal. Distance left:", self.distance_to_goal)
            reward = -30
            truncated = True

        # Reset the simulation when the agent makes the robot fall over
        if head_link_position[2] < 0.3:
            print("Robo fell over.")
            self.robo_fell_over = 1
            reward = -30
            truncated = True

        # Check if the agent made the robot collide with a wall or obstacle and reset the simulation if that happened
        if self.collision_detector_avoid.in_collision():
            print("Collision detected!")
            self.collision_count += 1
            reward = -30
            truncated = True

        # Check if the agent reached the goal
        if self.collision_detector_goal.in_collision():
            print("Goal Reached!")
            reward = 100
            done = True

        if self.passed_time_steps % 500 == 0:
            rgba, depth, seg = self.above_camera.get_frame()

            # save the frame
            self.above_camera.save_frame("frame.png", rgba=rgba)

        self.passed_time_steps += 1

        obs = self.get_observation_vector()
        # print(obs[0:15])

        return (
            obs,
            reward,
            done,
            truncated,
            {},
        )

    # Calculates the position and orientation for the front camera of the robot
    def calculate_camera_position_and_orientation(self, head_link_orientation, head_link_position):
        # Rotate the camera in the direction the robot head link points
        camera_position = np.array(head_link_position)
        camera_position -= head_link_position
        camera_position = putils.quaternion_rotate(head_link_orientation, camera_position)
        # Vector used to move the camera in front of the head and the camera target in front of the camera
        point_in_front_of_camera = np.array(putils.quaternion_rotate(head_link_orientation, np.array([0, 0.2, 0])))
        camera_position += point_in_front_of_camera
        # Move the camera in front of the head and put the target in front of it, so it is oriented ahead of the robot
        camera_position += head_link_position
        target_position = np.array(camera_position)
        target_position += point_in_front_of_camera
        # print("Camera and target position:", camera_position, target_position)
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
        # Exploration sub-class by name or full path to module+class
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
        .training(double_q=False, lr_schedule=[[0, 1e-3], [50000, 1e-6]], v_min=-30, v_max=100, noisy=False,
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
