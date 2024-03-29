import os
import argparse
import random
from enum import Enum

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
from ray.tune.logger import pretty_print
from ray.rllib.algorithms.dqn import DQNConfig

tf1, tf, tfv = try_import_tf()
torch, nn = try_import_torch()


# Global variables used in the environment
class GlobalVars:
    passed_iterations = 0


# Contains all constants used in the environment
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


# Enum for how much of a circle should be drawn
class CirclePortion(Enum):
    HALF = "HALF"
    QUARTER = "QUARTER"
    FULL = "FULL"


def make_curved_line(radius=1.0, position=np.array([0, 0, 0]), rotation_angle=0, portion=CirclePortion.HALF):
    """
    This function generates a curved line in the environment.
    The curved line can be a full circle, half circle, or quarter circle.

    Parameters:
        radius (float): The radius of the curved line. Default is 1.0.
        position (np.array): The position of the curved line in the environment. Default is [0, 0, 0].
        rotation_angle (int): The rotation angle of the curved line. Default is 0.
        portion (CirclePortion): The portion of the circle to be drawn.
        Can be CirclePortion.HALF, CirclePortion.QUARTER, or CirclePortion.FULL. Default is CirclePortion.HALF.

    Returns:
        list: A list of object IDs for the created curved line.
    """
    if radius > 0.1:
        # Create the visual shapes for the outer and inner cylinders for the curved line
        outer_circle_cylinder_id = p.createVisualShape(p.GEOM_CYLINDER, radius=radius, length=0.01,
                                                       rgbaColor=[0, 0, 0, 1])
        inner_circle_cylinder_id = p.createVisualShape(p.GEOM_CYLINDER, radius=radius - 0.04, length=0.013,
                                                       rgbaColor=[1, 1, 1, 1])

        # Create the visual shape for the stopper for the curved line
        half_circle_stopper_id = p.createVisualShape(p.GEOM_BOX, halfExtents=[radius / 2, radius, 0.013],
                                                     rgbaColor=[1, 1, 1, 1])

        quarter_circle_stopper_id = p.createVisualShape(p.GEOM_BOX, halfExtents=[radius, radius / 2, 0.013],
                                                        rgbaColor=[1, 1, 1, 1])

        # Calculate the displacement for the stoppers
        stopper_displacement = putils.quaternion_rotate(p.getQuaternionFromAxisAngle([0, 0, 1], rotation_angle),
                                                        [radius / 2, 0, 0])
        second_stopper_displacement = putils.quaternion_rotate(p.getQuaternionFromAxisAngle([0, 0, 1], rotation_angle),
                                                               [0, radius / 2, 0])

        # Calculate the position for the stoppers
        stopper_position = position + stopper_displacement
        second_stopper_position = position + second_stopper_displacement

        obj_ids = []

        # Create the outer and inner cylinders
        obj_ids.append(p.createMultiBody(baseMass=0, baseInertialFramePosition=[0, 0, 0],
                                         baseVisualShapeIndex=outer_circle_cylinder_id, basePosition=position,
                                         baseOrientation=[0, 0, 0, 1], useMaximalCoordinates=True))
        obj_ids.append(p.createMultiBody(baseMass=0, baseInertialFramePosition=[0, 0, 0],
                                         baseVisualShapeIndex=inner_circle_cylinder_id, basePosition=position,
                                         baseOrientation=[0, 0, 0, 1], useMaximalCoordinates=True))

        # Create the stoppers based on the portion of the circle to be drawn
        if portion == CirclePortion.HALF or portion != CirclePortion.FULL:
            obj_ids.append(p.createMultiBody(baseMass=0, baseInertialFramePosition=[0, 0, 0],
                                             baseVisualShapeIndex=half_circle_stopper_id, basePosition=stopper_position,
                                             baseOrientation=p.getQuaternionFromAxisAngle([0, 0, 1], rotation_angle),
                                             useMaximalCoordinates=True))

        if portion == CirclePortion.QUARTER and portion != CirclePortion.FULL:
            obj_ids.append(p.createMultiBody(baseMass=0, baseInertialFramePosition=[0, 0, 0],
                                             baseVisualShapeIndex=quarter_circle_stopper_id,
                                             basePosition=second_stopper_position,
                                             baseOrientation=p.getQuaternionFromAxisAngle([0, 0, 1], rotation_angle),
                                             useMaximalCoordinates=True))

        return obj_ids
    else:
        return []


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
    "--stop-reward", type=float, default=160, help="Reward at which the training should be stopped."
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


class FollowLines(gym.Env):
    """
        This class represents the environment for the reinforcement learning agent.
        It consists of a 2D plane where the agent, represented by a robot, has to follow a line.
        The environment is built using the PyBullet physics engine and the Gym library.
    """
    def __init__(self, env_config: EnvContext):
        """
            Initializes the environment.

            Args:
                env_config (EnvContext): The configuration for the environment.
        """
        # Connect to the PyBullet physics engine
        self.client_id = p.connect(p.GUI)

        # Set the path for the PyBullet data
        p.setAdditionalSearchPath(pybullet_data.getDataPath())

        # Set the gravity for the environment
        p.setGravity(0, 0, -10)

        # Load the plane on that the robot will move
        plane_id = p.loadURDF("plane.urdf", globalScaling=0.5, basePosition=[2, 0, 0])

        # Set the starting position and orientation for the robot
        start_pos = [0, -5.3, 0.3]
        start_orientation_walls_vert = [np.sin(np.pi / 2), 1, 0, 0]
        start_orientation_robo = [0, 0, 0, 1]

        # Load the robot model
        self.robo_id = p.loadURDF("r2d2_without_arm/r2d2_short.urdf", start_pos, start_orientation_robo)

        # Get the number of joints of the robot
        self.num_joints = p.getNumJoints(self.robo_id)

        # Variable that will check if the robot fell over
        self.robo_fell_over = 0

        # List that will store the positions of the robot
        self.positions = []

        # List that will store the data
        self.data = []

        # Variable that will count the number of time steps that have passed since the start of the episode
        self.passed_time_steps = 0

        # Variables that will contain the forces applied to the wheels
        self.force_left_wheels = 0
        self.force_right_wheels = 0

        # Arrays that will contain the last three frames captured by the robot's camera
        self.rgba0 = np.zeros((Constants.IMAGE_HEIGHT, Constants.IMAGE_WIDTH, 4), np.int32)
        self.rgba1 = np.zeros((Constants.IMAGE_HEIGHT, Constants.IMAGE_WIDTH, 4), np.int32)
        self.rgba2 = np.zeros((Constants.IMAGE_HEIGHT, Constants.IMAGE_WIDTH, 4), np.int32)

        # Print the number of joints of the robot
        print(self.num_joints)

        # Print the information for each joint of the robot
        for jointIndex in range(0, self.num_joints):
            print(p.getJointInfo(self.robo_id, jointIndex))

        # Create two curved lines in the environment
        make_curved_line(1.75, np.array([3.03, -4, 0]), np.deg2rad(270), CirclePortion.HALF)
        make_curved_line(1.75, np.array([3.03, 1, 0]), np.deg2rad(90), CirclePortion.HALF)

        # Create the visual and collision shapes for the straight line in the environment
        line_id_coll = p.createCollisionShape(p.GEOM_BOX, halfExtents=[8, 0.02, 0])
        line_id = p.createVisualShape(p.GEOM_BOX, halfExtents=[8, 0.02, 0.01], rgbaColor=[0, 0, 0, 1])

        line_ids = []

        # Create the straight line in the environment
        line_ids.append(p.createMultiBody(baseMass=0,
                                          baseInertialFramePosition=[0, 0, 0],
                                          baseCollisionShapeIndex=line_id_coll,
                                          baseVisualShapeIndex=line_id,
                                          basePosition=[0, 2.5, 0.001],
                                          baseOrientation=start_orientation_walls_vert,
                                          useMaximalCoordinates=True))

        # Get the state of the head link of the robot
        head_link_state = p.getLinkState(self.robo_id, 13, computeForwardKinematics=True)

        # Get the position and orientation of the head link
        link_position = np.array(head_link_state[0])
        link_orientation = head_link_state[1]

        # Calculate the position of the camera and the target position
        point_in_front_of_camera = np.array(putils.quaternion_rotate(link_orientation, np.array([0, 0.2, 0.4])))
        camera_position = link_position + point_in_front_of_camera

        point_in_front_of_camera = np.array(putils.quaternion_rotate(link_orientation, np.array([0, 0.2, 0.3])))
        target_position = camera_position + point_in_front_of_camera

        # Initialize the camera of the robot
        self.robo_camera = putils.Camera.from_distance_rpy(target_position=target_position, distance=0.2, near=0.001,
                                                           far=20, width=Constants.IMAGE_WIDTH,
                                                           height=Constants.IMAGE_HEIGHT, fov=80)

        # Set the pose of the camera
        self.robo_camera.set_camera_pose(camera_position, target_position)

        # Get the initial frame from the camera
        rgba, depth, seg = self.robo_camera.get_frame()
        self.rgba = np.array(rgba, dtype=np.int32)

        # Get the position and orientation of the robot
        self.robo_pos, self.robo_orn = p.getBasePositionAndOrientation(self.robo_id)

        # Set up the action and observation spaces
        self.action_space = Discrete(3)

        spaces = {}

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

        # Save the initial state of the environment
        self.initial_state = p.saveState()

        self.reset()

    def get_observation_vector(self):
        """
        This method generates an observation vector for the reinforcement learning agent.
        The observation vector consists of the last three frames captured by the robot's camera.
        Each frame is represented as a 3D numpy array (height, width, RGBA),
        where RGBA stands for Red, Green, Blue, and Alpha channels of each pixel.

        Returns:
            dict: A dictionary containing the last three frames captured by the robot's camera.
            The newest frame is stored in the key "rgba2", the second newest frame is stored in the key "rgba1",
            the oldest frame is stored in the key "rgba0".
            The keys are "rgba0", "rgba1", and "rgba2".
        """
        obs = {}

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
        # Reset the simulation
        p.restoreState(self.initial_state)
        # Append the positions list to a file. It contains the positions of the robot during the last episode
        with open('positions.txt', 'a') as f:
            f.write(str(self.positions))
            f.write('\n')

        self.positions = []
        self.force_left_wheels = 0
        self.force_right_wheels = 0

        # Place the robot at a different starting position after 10 iterations
        if GlobalVars.passed_iterations > 9:
            p.resetBasePositionAndOrientation(self.robo_id, [1.25, -4.2, 0.3], [0, 0, 0, 1])
        self.robo_pos, self.robo_orn = p.getBasePositionAndOrientation(self.robo_id)

        # Get the initial frame from the camera
        self.rgba, self.depth, seg = self.robo_camera.get_frame()

        # Append the robots start position to the positions list
        self.positions.append([self.robo_pos[0], self.robo_pos[1]])
        obs = self.get_observation_vector()
        return obs, {}

    def step(self, action):
        reward = 0
        done = truncated = False

        # Set the force applied to the wheels based on the action
        assert action in [0, 1, 2], action
        if action == 0:
            # forward with force 40
            self.force_right_wheels = -40
            self.force_left_wheels = -40
        elif action == 1:
            # turn right
            self.force_right_wheels = 0
            self.force_left_wheels = -40
        elif action == 2:
            # turn left
            self.force_right_wheels = -40
            self.force_left_wheels = 0

        # Apply the forces to the wheels
        p.setJointMotorControlArray(bodyIndex=self.robo_id,
                                    jointIndices=[2, 3, 6, 7],
                                    controlMode=p.VELOCITY_CONTROL,
                                    targetVelocities=[self.force_right_wheels, self.force_right_wheels,
                                                      self.force_left_wheels, self.force_left_wheels],
                                    forces=[Constants.MAX_FORCE_WHEELS, Constants.MAX_FORCE_WHEELS,
                                            Constants.MAX_FORCE_WHEELS,
                                            Constants.MAX_FORCE_WHEELS])

        # Run the simulation for 24 steps with the chosen action
        for i in range(24):
            p.stepSimulation()

        self.robo_pos, self.robo_orn = p.getBasePositionAndOrientation(self.robo_id)
        self.positions.append([self.robo_pos[0], self.robo_pos[1]])

        head_link_state = p.getLinkState(self.robo_id, 13, computeForwardKinematics=True)
        head_link_position = np.array(head_link_state[0])
        head_link_orientation = head_link_state[1]

        camera_position, target_position = calculate_camera_position_and_orientation(head_link_orientation,
                                                                                     head_link_position)

        self.robo_camera.set_camera_pose(camera_position, target_position)

        self.rgba, self.depth, seg = self.robo_camera.get_frame()
        # save the frame
        # self.robo_camera.save_frame("frame.png", rgba=self.rgba)

        black_line_visible = False
        # Reward the agent for keeping the black line in the center of the screen
        # Iterate over the height of the image
        for i in range(len(self.rgba)):
            # Iterate over the width of the image
            for j in range(len(self.rgba[i])):
                # Check if the pixel at (i, j) is black
                if self.rgba[i][j][2] <= 0 and self.rgba[i][j][0] <= 0 >= self.rgba[i][j][1]:
                    black_line_visible = True
                    # Old reward calculation that uses a linear function
                    '''
                    if j < len(self.rgba[i]):
                        reward += np.abs((j / (len(self.rgba[i]) / 2)) / 1000)
                    else:
                        reward += np.abs((2 - j / (len(self.rgba[i]) / 2)) / 1000)
                    '''
                    # Calculate a reward based on the position of the black pixel in the image
                    # The reward is calculated using a Gaussian function, that is centered in the middle of the image
                    # The closer the black pixel is to the center of the image, the higher the reward
                    # This encourages the agent to keep the black line in the center of the image
                    reward += np.exp(
                        -np.square(
                            (((j + 1) - len(self.rgba[i]) / 2) / len(self.rgba[i])) / 0.05
                        )
                    ) / 200

        # Agent lost the line
        if not black_line_visible:
            reward = -5
            truncated = True

        # Reset the simulation when the agent makes the robot fall over
        if head_link_position[2] < 0.3:
            print("Robo fell over.")
            self.robo_fell_over = 1
            reward = -1
            truncated = True

        self.passed_time_steps += 1

        obs = self.get_observation_vector()
        # print(reward)

        return (
            obs,
            reward,
            done,
            truncated,
            {},
        )


# Calculates the position and orientation for the front camera of the robot
def calculate_camera_position_and_orientation(head_link_orientation, head_link_position):
    # Calculate the camera position
    camera_position = head_link_position + putils.quaternion_rotate(head_link_orientation, np.array([0, 0.2, 0]))

    # Calculate the target position
    target_position = camera_position + putils.quaternion_rotate(head_link_orientation, np.array([0, 0.2, -0.4]))

    return camera_position, target_position


def plt_image(img):
    plt.axis('off')
    plt.imshow(img)
    plt.show()


def create_checkpoint():
    save_result = algo.save()
    path_to_checkpoint = save_result.checkpoint.path
    print(
        "An Algorithm checkpoint has been created inside directory: "
        f"'{path_to_checkpoint}'."
    )
    return path_to_checkpoint


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
        "epsilon_timesteps": 10000,  # Timesteps over which to anneal epsilon.
    }

    config = (
        DQNConfig(args.run)
        .environment(FollowLines, env_config={})
        .framework(args.framework)
        .rollouts(num_rollout_workers=0, create_env_on_local_worker=True)
        # Use GPUs iff `RLLIB_NUM_GPUS` env var set to > 0.
        .resources(num_gpus=int(os.environ.get("RLLIB_NUM_GPUS", "0")))
        .training(double_q=False, v_min=-30, lr_schedule=[[0, 1e-4], [10000, 1e-6]], v_max=100, noisy=False,
                  replay_buffer_config=replay_config)
        .exploration(explore=True, exploration_config=exploration_config)
        # lr=7e-5,
        # lr_schedule=[[0, 1e-4], [20000, 1e-6]],
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
            print(result['training_iteration'])
            GlobalVars.passed_iterations = result['training_iteration']
            print('Counted iterations + 1')
            print("Current Exploration Epsilon:", algo.get_policy().get_exploration_state()['cur_epsilon'])
            print(pretty_print(result))
            if GlobalVars.passed_iterations == 10:
                path = create_checkpoint()

                algo.stop()
                p.disconnect()

                new_exploration_config = {
                    # Exploration sub-class by name or full path to module+class
                    # (e.g. “ray.rllib.utils.exploration.epsilon_greedy.EpsilonGreedy”)
                    "type": "EpsilonGreedy",
                    # Parameters for the Exploration class' constructor:
                    "initial_epsilon": 0.5,
                    "final_epsilon": 0.02,
                    "epsilon_timesteps": 20000,  # Timesteps over which to anneal epsilon.
                }

                new_config = (
                    DQNConfig(args.run)
                    .environment(FollowLines, env_config={})
                    .framework(args.framework)
                    .rollouts(num_rollout_workers=0, create_env_on_local_worker=True)
                    # Use GPUs iff `RLLIB_NUM_GPUS` env var set to > 0.
                    .resources(num_gpus=int(os.environ.get("RLLIB_NUM_GPUS", "0")))
                    .training(double_q=False, v_min=-30, lr_schedule=[[10000, 1e-4], [20000, 1e-6]], v_max=100,
                              noisy=False,
                              replay_buffer_config=replay_config)
                    .exploration(explore=True, exploration_config=new_exploration_config)
                    # lr=7e-5,
                    # lr_schedule=[[0, 1e-4], [20000, 1e-6]],
                )
                # config.gamma = 0.999
                algo = new_config.build()
                algo.restore(path)

            # stop training of the target train steps or reward are reached
            if (
                    result["timesteps_total"] >= args.stop_timesteps
                    or result["episode_reward_mean"] >= args.stop_reward
            ):
                break

        path = create_checkpoint()
        algo.stop()

    ray.shutdown()

# Old way of creating a curved line. This is now done using the make_curved_line function.
'''
        small_line_id_coll = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.5, 0.02, 0])
        small_line_id = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.5, 0.02, 0.01], rgbaColor=[0, 0, 0, 1])

        # Curve Lines
        line_ids.append(p.createMultiBody(baseMass=0,
                                          baseInertialFramePosition=[0, 0, 0],
                                          baseCollisionShapeIndex=small_line_id_coll,
                                          baseVisualShapeIndex=small_line_id,
                                          basePosition=[4.71, -0.92, 0.001],
                                          baseOrientation=[np.sin(np.pi / 2), 1, 0, 0],
                                          useMaximalCoordinates=True))

        line_ids.append(p.createMultiBody(baseMass=0,
                                          baseInertialFramePosition=[0, 0, 0],
                                          baseCollisionShapeIndex=small_line_id_coll,
                                          baseVisualShapeIndex=small_line_id,
                                          basePosition=[4.85, 0, 0.001],
                                          baseOrientation=p.getQuaternionFromEuler([0, 0, np.pi/2.5]),
                                          useMaximalCoordinates=True))

        line_ids.append(p.createMultiBody(baseMass=0,
                                          baseInertialFramePosition=[0, 0, 0],
                                          baseCollisionShapeIndex=small_line_id_coll,
                                          baseVisualShapeIndex=small_line_id,
                                          basePosition=[5.26, 0.9, 0.001],
                                          baseOrientation=p.getQuaternionFromEuler([0, 0, np.pi/3]),
                                          useMaximalCoordinates=True))

        line_ids.append(p.createMultiBody(baseMass=0,
                                          baseInertialFramePosition=[0, 0, 0],
                                          baseCollisionShapeIndex=small_line_id_coll,
                                          baseVisualShapeIndex=small_line_id,
                                          basePosition=[5.8, 1.6, 0.001],
                                          baseOrientation=p.getQuaternionFromEuler([0, 0, np.pi/4]),
                                          useMaximalCoordinates=True))

        line_ids.append(p.createMultiBody(baseMass=0,
                                          baseInertialFramePosition=[0, 0, 0],
                                          baseCollisionShapeIndex=small_line_id_coll,
                                          baseVisualShapeIndex=small_line_id,
                                          basePosition=[6.6, 2.2, 0.001],
                                          baseOrientation=p.getQuaternionFromEuler([0, 0, np.pi / 6]),
                                          useMaximalCoordinates=True))

        line_ids.append(p.createMultiBody(baseMass=0,
                                          baseInertialFramePosition=[0, 0, 0],
                                          baseCollisionShapeIndex=small_line_id_coll,
                                          baseVisualShapeIndex=small_line_id,
                                          basePosition=[7.5, 2.45, 0.001],
                                          baseOrientation=p.getQuaternionFromEuler([0, 0, 0]),
                                          useMaximalCoordinates=True))

        line_ids.append(p.createMultiBody(baseMass=0,
                                          baseInertialFramePosition=[0, 0, 0],
                                          baseCollisionShapeIndex=small_line_id_coll,
                                          baseVisualShapeIndex=small_line_id,
                                          basePosition=[8.4, 2.25, 0.001],
                                          baseOrientation=p.getQuaternionFromEuler([0, 0, np.pi + (np.pi / 1.15)]),
                                          useMaximalCoordinates=True))

        line_ids.append(p.createMultiBody(baseMass=0,
                                          baseInertialFramePosition=[0, 0, 0],
                                          baseCollisionShapeIndex=small_line_id_coll,
                                          baseVisualShapeIndex=small_line_id,
                                          basePosition=[9.2, 1.70, 0.001],
                                          baseOrientation=p.getQuaternionFromEuler([0, 0, np.pi + (np.pi / 1.35)]),
                                          useMaximalCoordinates=True))

        line_ids.append(p.createMultiBody(baseMass=0,
                                          baseInertialFramePosition=[0, 0, 0],
                                          baseCollisionShapeIndex=small_line_id_coll,
                                          baseVisualShapeIndex=small_line_id,
                                          basePosition=[9.72, 0.90, 0.001],
                                          baseOrientation=p.getQuaternionFromEuler([0, 0, np.pi + (np.pi / 1.6)]),
                                          useMaximalCoordinates=True))

        line_ids.append(p.createMultiBody(baseMass=0,
                                          baseInertialFramePosition=[0, 0, 0],
                                          baseCollisionShapeIndex=small_line_id_coll,
                                          baseVisualShapeIndex=small_line_id,
                                          basePosition=[9.98, 0.0, 0.001],
                                          baseOrientation=p.getQuaternionFromEuler([0, 0, np.pi + (np.pi / 1.8)]),
                                          useMaximalCoordinates=True))

        line_ids.append(p.createMultiBody(baseMass=0,
                                          baseInertialFramePosition=[0, 0, 0],
                                          baseCollisionShapeIndex=small_line_id_coll,
                                          baseVisualShapeIndex=small_line_id,
                                          basePosition=[10.06, -0.98, 0.001],
                                          baseOrientation=p.getQuaternionFromEuler([0, 0, np.pi + (np.pi / 2)]),
                                          useMaximalCoordinates=True))

        goal_box_id_coll = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.2, 0.2, 0.2])
        goal_box_id = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.2, 0.2, 0.2], rgbaColor=[0, 0, 1, 0])

        self.goal_ball_body_id = p.createMultiBody(baseMass=0,
                                                   baseInertialFramePosition=[0, 0, 0],
                                                   baseCollisionShapeIndex=goal_box_id_coll,
                                                   baseVisualShapeIndex=goal_box_id,
                                                   basePosition=Constants.GOAL_POSITION,
                                                   baseOrientation=start_orientation_walls_hori,
                                                   useMaximalCoordinates=True)

        self.collision_detector_goal = putils.CollisionDetector(self.client_id,
                                                                [(self.robo_id, self.goal_ball_body_id)])
        '''
