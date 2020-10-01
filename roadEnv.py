# -*- coding: utf-8 -*-
"""Author: Haoran Su
Email: hs1854@nyu.edu
"""
import vehicle_env
import numpy as np
import generate

# Action is now defined that index of vehicle start to yield at given step
# Under this scheme, only one vehicle can yield at each step
# action = -1 if no vehicle is yielding at the end of this step

# # Utility function
# def decode_action(s_action):
#     res = []
#     str_list = f'{s_action:015b}'.split()
#     for char in str_list[0]:
#         res.append(int(char))
#     return res


def lane_change(observation):
    """
    :param observation:
    :return: a new observation putting all possible lane change into position
    """
    # Find all vehicles who are ready to change lane:
    veh_lower_lane = [veh for veh in observation if veh[1] == 1]
    po_index = []
    # index for vehicle who needs to pull over:
    for i in range(len(observation)):
        if (observation[i][1] == 0) and (observation[i][2] == 0):
            po_index.append(i)

    for elem in po_index:
        can_pull_over = True
        for veh_1 in veh_lower_lane:
            if observation[elem][0] >= veh_1[0]:
                leading_veh = observation[elem]
                following_veh = veh_1
            else:
                leading_veh = veh_1
                following_veh = observation[elem]

            if leading_veh[0] - leading_veh[3] < following_veh[0]:
                can_pull_over = False
        if can_pull_over:
            observation[elem][1] = 1

    return observation


def mapped_state(state):
    """
    Mapped a given state into the state as neural network input, insert trivial vehicles until vehs_num = 18
    :param state: given state
    :return: new state
    """

    num_diff = 18 - len(state)
    for i in range(num_diff):
        state.append([0, 0, 0, 0, 0, 2])
    return state


def random_deceleration(most_comfortable_deceleration, lane_pos):
    """
    Return a deceleration based on given attribute of the vehicle
    :param most_comfortable_deceleration: the given attribute of the vehicle
    :param lane_pos: y
    :return: the deceleration adopted by human driver
    """
    if lane_pos:
        sigma = 0.3
    else:
        sigma = 0.5
    return np.random.normal(most_comfortable_deceleration, sigma)


# Calculating rewards depends only on the state:
def calculate_reward(state, l_gap=0.25, road_length=200):
    """
    Calculate reward for the given state. Notice that this function doesnt account for status inconsistency, but it gets
    covered in the state_transition function.
    :param road_length: segment length
    :param l_gap: minimum safety gap
    :param state: given state
    :return: reward for this state
    """

    # First get the number of valid vehicles:
    num_valid_vehs = 0
    for veh in state:
        if veh[5] != 2:
            num_valid_vehs += 1

    # Initialize reward for this step
    reward = -1

    # Initialize collision indicator:
    has_collision = False

    # Initialize process completed:
    has_done = True

    # For the new state:
    # First, we want to check collision:
    for i in range(num_valid_vehs):
        for j in range(i + 1, num_valid_vehs):

            # Determine which vehicle is the leading vehicle by their front position:
            if state[i][0] >= state[j][0]:
                leading_veh = state[i]
                following_veh = state[j]
            else:
                leading_veh = state[j]
                following_veh = state[i]

            # Find out the back of leading vehicle and front of following vehicle:
            back_pos = leading_veh[0] - leading_veh[3]
            front_pos = following_veh[0]

            # Collision check: 1. both vehicles are on the same lane, 2: two vehicles have overlapped with minimum
            # safety gap.
            if (back_pos < 200) and (leading_veh[1] != 2) and (following_veh[1] != 2) and (leading_veh[1] == following_veh[1])  \
                    and (back_pos - l_gap < front_pos):
                has_collision = True

    # # If any vehicle is on lane 0 and vehicle position has not exceed the roadway length:
    # for veh in state[:num_valid_vehs - 1]:
    #     if veh[1] == 0 and veh[0] - veh[3] <= road_length:
    #         has_cleared = False

    # Summarize reward:
    # If there is collision, apply collision penalty and end the process:
    if not has_collision:
        vehs_left_counter = 0
        for veh in state:
            if (veh[1] == 0) and (veh[0] - veh[3]) < road_length:
                vehs_left_counter += 1
                has_done = False
        reward -= vehs_left_counter
    else:
        reward -= 2000
        has_done = True
    return has_done, reward


# Road Environment
class RoadEnv:
    def __init__(self, road_length=200, l_gap=0, delta_t=0.5):
        # self.state = mapped_state(vehicle_env.generate_env_nparray())
        self.state = mapped_state(generate.generate_road_env_nonOO())
        self.done = False
        self.road_length = road_length  # Length of the roadway segment
        self.l_gap = l_gap  # Minimum safety gap
        self.delta_t = delta_t  # length of the timestamp interval

    def reset(self):
        # new_state = vehicle_env.generate_env_nparray()
        new_state = generate.generate_road_env_nonOO()
        self.state = mapped_state(new_state)

        return self.state

    def step(self, observation, action):
        """
            State Transition function to compute the next state
            :param action: now is a integer representing the index of which vehicle needs to yield in this step
            :param observation: s(t)
            :return: observation_,(next_state) reward, done(whether the process has completed)
        """
        # Initialize next state with only valid vehicles
        observation_ = []

        # Initialize a repetitive yielding instruction error
        # repetitive_yielding = False

        # Find the number of valid vehicles and only iterate through valid vehicles
        num_valid_vehs = 0
        for veh in observation:
            if veh[5] != 2:
                num_valid_vehs += 1

        # print("number of valid vehicles is :"+str(num_valid_vehs))

        # Iterate through all valid vehicles:
        for i in range(num_valid_vehs):

            # # extract corresponding action
            if action == i:
                action_i = 1
            else:
                action_i = 0

            # Model vehicle kinetics here:

            old_x = observation[i][0]
            old_y = observation[i][1]
            old_velocity = observation[i][2]
            veh_len = observation[i][3]
            most_comfortable_deceleration = observation[i][4]
            old_status = observation[i][5]

            # Determine new_status and b_actual based on old_status and action_i:
            if (not old_status) and (not action_i):
                new_status = 0
                # b_actual = 0
            elif (not old_status) and action_i:
                new_status = 1
                # b_actual = random_deceleration(most_comfortable_deceleration, old_y)
            elif old_status and (not action_i):
                new_status = 1
                # b_actual = random_deceleration(most_comfortable_deceleration, old_y)
            else:
                new_status = 1
                # b_actual = random_deceleration(most_comfortable_deceleration, old_y)
                repetitive_yielding = True

            if new_status:
                # b_actual = random_deceleration(most_comfortable_deceleration, old_y)
                # b_actual = random_deceleration(most_comfortable_deceleration, old_y)
                b_actual = random_deceleration(most_comfortable_deceleration, old_y)
            else:
                b_actual = 0

            new_x = old_x + old_velocity * self.delta_t
            new_y = old_y
            if action_i and (old_y == 0):
                new_y = 2

            new_v = max(0, old_velocity - b_actual * self.delta_t)


            # If the vehicle has left the lane, we add it to a virtually
            # Assign new vehicle information and add it to the new observation
            new_veh = [new_x, int(new_y), new_v, veh_len, most_comfortable_deceleration, new_status]
            observation_.append(new_veh)

        # print("Before lane change the state is :" + str(observation_))
        # observation_ = lane_change(observation_)

        # Calculate reward without considering action-status inconsistency
        # At this step, observation_ only contains valid vehicles, and reward is only calculated for that:
        # Check if the process also completes in this step:
        done, reward = calculate_reward(observation_, road_length=self.road_length, l_gap=self.l_gap)

        # Additionally, if there is any repetitive yielding instruction, we applies penalty and end the game:
        # if repetitive_yielding:
        #     reward -= 1

        # # If there is a yielding instruction to valid vehicle, we should encourage it:
        # if action <= num_valid_vehs - 1:
        #     reward += 10

        # pad the next state to consistent state length:
        observation_ = mapped_state(observation_)

        return observation_, reward, done
