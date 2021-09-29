import math
import numpy as np


def reward_function(params):
    '''
    In @params object:
    {
        "all_wheels_on_track": Boolean,    # flag to indicate if the vehicle is on the track
        "x": float,                        # vehicle's x-coordinate in meters
        "y": float,                        # vehicle's y-coordinate in meters
        "distance_from_center": float,     # distance in meters from the track center
        "is_left_of_center": Boolean,      # Flag to indicate if the vehicle is on the left side to the track center or not.
        "heading": float,                  # vehicle's yaw in degrees
        "progress": float,                 # percentage of track completed
        "steps": int,                      # number steps completed
        "speed": float,                    # vehicle's speed in meters per second (m/s)
        "streering_angle": float,          # vehicle's steering angle in degrees
        "track_width": float,              # width of the track
        "waypoints": [[float, float], … ], # list of [x,y] as milestones along the track center
        "closest_waypoints": [int, int]    # indices of the two nearest waypoints.
    }
    '''

    closest_waypoints = params['closest_waypoints']

    reward = 1e-3

    # daniel's reward part
    if 135 <= closest_waypoints[1] or closest_waypoints[1] <= 17:
        reward = reward_function_daniel(reward, params)

    # ben's reward part
    if 18 <= closest_waypoints[1] or closest_waypoints[1] <= 56:
        reward = reward_function_ben(reward, params)

    # samuel's reward part
    if 57 <= closest_waypoints[1] or closest_waypoints[1] <= 95:
        reward = reward_function_samuel(reward, params)

    # june's reward part
    if 96 <= closest_waypoints[1] or closest_waypoints[1] <= 134:
        reward = reward_function_june(reward, params)

    return float(reward)


def reward_function_minimum(params):
    closest_waypoints = params['closest_waypoints']

    minimum_reward = 1e-3

    # daniel's reward part
    if 135 <= closest_waypoints[1] or closest_waypoints[1] <= 17:
        minimum_reward = reward_function_minimum_daniel(minimum_reward, params)

    # ben's reward part
    if 18 <= closest_waypoints[1] or closest_waypoints[1] <= 56:
        minimum_reward = reward_function_minimum_ben(minimum_reward, params)

    # samuel's reward part
    if 57 <= closest_waypoints[1] or closest_waypoints[1] <= 95:
        minimum_reward = reward_function_minimum_samuel(minimum_reward, params)

    # june's reward part
    if 96 <= closest_waypoints[1] or closest_waypoints[1] <= 134:
        minimum_reward = reward_function_minimum_june(minimum_reward, params)

    return float(minimum_reward)


def reward_function_maximum(params):
    closest_waypoints = params['closest_waypoints']

    maximum_reward = 1e-3

    # daniel's reward part
    if 135 <= closest_waypoints[1] or closest_waypoints[1] <= 17:
        maximum_reward = reward_function_maximum_daniel(maximum_reward, params)

    # ben's reward part
    if 18 <= closest_waypoints[1] or closest_waypoints[1] <= 56:
        maximum_reward = reward_function_maximum_ben(maximum_reward, params)

    # samuel's reward part
    if 57 <= closest_waypoints[1] or closest_waypoints[1] <= 95:
        maximum_reward = reward_function_maximum_samuel(maximum_reward, params)

    # june's reward part
    if 96 <= closest_waypoints[1] or closest_waypoints[1] <= 134:
        maximum_reward = reward_function_maximum_june(maximum_reward, params)

    return float(maximum_reward)


def reward_function_daniel(current_reward, params):
    reward = 1e-3

    limit_params = {
        'optimal_speed': 2.5,
        'standard_speed': 1.5,
        'minimum_speed': 1.0,
        'steering_limit': 10.0,
        'straight_line_direction_limit': 5.0,
        'optimal_reward_l1': 2.5,
        'optimal_reward_l2': 1.8,
        'standard_reward_l1': 1.5,
        'standard_reward_l2': 0.9,
        'minimum_reward_l1': 0.6,
        'minimum_reward_l2': 0.3,
        'steps_per_second': 15
    }

    x = params['x']
    y = params['y']
    on_track = params['all_wheels_on_track']
    distance_from_center = params['distance_from_center']
    track_width = params['track_width']
    steering = params['steering_angle']
    speed = params['speed']
    waypoints = params['waypoints']
    closest_waypoints = params['closest_waypoints']
    heading = params['heading']
    is_left_of_center = params['is_left_of_center']

    reward = 1e-3

    # get the next 3 points
    np0 = waypoints[closest_waypoints[1]]
    np1 = waypoints[(closest_waypoints[1] + 1) % len(waypoints)]
    np2 = waypoints[(closest_waypoints[1] + 2) % len(waypoints)]

    # distance checking to make sure the agent is heading to the right direction
    distance_p0 = distance_of_2points(waypoints[closest_waypoints[0]], [x, y])
    distance_p = distance_of_2points(np0, [x, y])
    distance_p1 = distance_of_2points(np1, [x, y])
    distance_p2 = distance_of_2points(np2, [x, y])

    # projected position of the agent
    new_x = x + speed / limit_params['steps_per_second'] * math.cos(
        math.radians(heading + steering / limit_params['steps_per_second']))
    new_y = y + speed / limit_params['steps_per_second'] * math.sin(
        math.radians(heading + steering / limit_params['steps_per_second']))

    new_distance_p0 = distance_of_2points(waypoints[closest_waypoints[0]], [new_x, new_y])
    new_distance_p = distance_of_2points(np0, [new_x, new_y])
    new_distance_p1 = distance_of_2points(np1, [new_x, new_y])
    new_distance_p2 = distance_of_2points(np2, [new_x, new_y])

    agent_going_to_right_direction = False
    if distance_p0 <= new_distance_p0 and distance_p1 >= new_distance_p1 and distance_p2 >= new_distance_p2:
        agent_going_to_right_direction = True

    # the agent direction to next 3 points
    direction_np = directions_of_2points([x, y], np0)
    direction_np1 = directions_of_2points([x, y], np1)
    direction_np2 = directions_of_2points([x, y], np2)

    direction_np_diff = abs(direction_np - heading - steering / limit_params['steps_per_second'])
    if direction_np_diff > 180:
        direction_np = 360 - direction_np

    direction_np1_diff = abs(direction_np1 - heading - steering / limit_params['steps_per_second'])
    if direction_np1_diff > 180:
        direction_np1 = 360 - direction_np1

    direction_np2_diff = abs(direction_np2 - heading - steering / limit_params['steps_per_second'])
    if direction_np2_diff > 180:
        direction_np2 = 360 - direction_np2

    direction_np2_np_diff = abs(direction_np2 - direction_np)
    if direction_np2_np_diff > 180:
        direction_np2_np_diff = 360 - direction_np2_np_diff

    if agent_going_to_right_direction and on_track:
        if speed >= limit_params['optimal_speed']:
            if (is_left_of_center and 0 >= steering / limit_params['steps_per_second'] >= -1 * limit_params[
                'steering_limit']) or \
                    (not is_left_of_center and 0 <= steering / limit_params['steps_per_second'] <= limit_params[
                        'steering_limit']):
                if direction_np1_diff <= limit_params['straight_line_direction_limit']:
                    reward = limit_params['optimal_reward_l1']
                elif direction_np1_diff <= direction_np2_np_diff:
                    reward = limit_params['optimal_reward_l2']
        elif speed >= limit_params['standard_speed']:
            if direction_np1_diff <= limit_params['straight_line_direction_limit']:
                reward = limit_params['standard_reward_l1']
            elif direction_np1_diff <= direction_np2_np_diff:
                reward = limit_params['standard_reward_l2']
        elif speed >= limit_params['minimum_speed']:
            if direction_np1_diff <= limit_params['straight_line_direction_limit']:
                reward = limit_params['minimum_reward_l1']
            elif direction_np1_diff <= direction_np2_np_diff:
                reward = limit_params['minimum_reward_l2']

    reward += current_reward
    return float(reward)


def reward_function_minimum_daniel(current_reward, params):
    reward = 1e-3
    return float(reward)


def reward_function_maximum_daniel(current_reward, params):
    reward = 2.5
    return float(reward)


def reward_function_ben(current_reward, params):
    reward = 1e-3
    '''
    your code here
    '''
    reward += current_reward
    return float(reward)


def reward_function_minimum_ben(current_reward, params):
    reward = 1e-3
    '''
    your code here
    '''
    reward += current_reward
    return float(reward)


def reward_function_maximum_ben(current_reward, params):
    reward = 1e-3
    '''
    your code here
    '''
    reward += current_reward
    return float(reward)


def reward_function_samuel(current_reward, params):
    reward = 1e-3

    # Parameters for Speed Incentive
    FUTURE_STEP = 6
    TURN_THRESHOLD_SPEED = 6  # degrees
    SPEED_THRESHOLD_SLOW = 1.8  # m/s
    SPEED_THRESHOLD_FAST = 2  # m/s

    # Parameters for Straightness Incentive
    FUTURE_STEP_STRAIGHT = 8
    TURN_THRESHOLD_STRAIGHT = 25  # degrees
    STEERING_THRESHOLD = 11  # degrees

    waypoints = params['waypoints']
    closest_waypoints = params['closest_waypoints']
    x = params['x']
    y = params['y']
    speed = params['speed']
    heading = params['heading']

    # Parameters for Progress Incentive
    TOTAL_NUM_STEPS = 675  # (15 steps per second, therefore < 45 secs)

    # get the next 3 points
    np0 = waypoints[closest_waypoints[1]]
    np1 = waypoints[(closest_waypoints[1] + 1) % len(waypoints)]
    np2 = waypoints[(closest_waypoints[1] + 2) % len(waypoints)]

    # distance checking to make sure the agent is heading to the right direction
    distance_p0 = distance_of_2points(waypoints[closest_waypoints[0]], [x, y])
    distance_p = distance_of_2points(np0, [x, y])
    distance_p1 = distance_of_2points(np1, [x, y])
    distance_p2 = distance_of_2points(np2, [x, y])

    # projected position of the agent
    new_x = x + speed / 15 * math.cos(math.radians(heading))
    new_y = y + speed / 15 * math.sin(math.radians(heading))

    new_distance_p0 = distance_of_2points(waypoints[closest_waypoints[0]], [new_x, new_y])
    new_distance_p = distance_of_2points(np0, [new_x, new_y])
    new_distance_p1 = distance_of_2points(np1, [new_x, new_y])
    new_distance_p2 = distance_of_2points(np2, [new_x, new_y])

    agent_going_to_right_direction = False
    if distance_p0 <= new_distance_p0 and distance_p1 >= new_distance_p1 and distance_p2 >= new_distance_p2:
        agent_going_to_right_direction = True

    def identify_corner(waypoints, closest_waypoints, future_step):

        # Identify next waypoint and a further waypoint
        point_prev = waypoints[closest_waypoints[0]]
        point_next = waypoints[closest_waypoints[1]]
        point_future = waypoints[min(len(waypoints) - 1,
                                     closest_waypoints[1] + future_step)]

        # Calculate headings to waypoints
        heading_current = math.degrees(math.atan2(point_prev[1] - point_next[1],
                                                  point_prev[0] - point_next[0]))
        heading_future = math.degrees(math.atan2(point_prev[1] - point_future[1],
                                                 point_prev[0] - point_future[0]))

        # Calculate the difference between the headings
        diff_heading = abs(heading_current - heading_future)

        # Check we didn't choose the reflex angle
        if diff_heading > 180:
            diff_heading = 360 - diff_heading

        # Calculate distance to further waypoint
        dist_future = np.linalg.norm([point_next[0] - point_future[0], point_next[1] - point_future[1]])

        return diff_heading, dist_future

    def select_speed(waypoints, closest_waypoints, future_step):

        # Identify if a corner is in the future
        diff_heading, dist_future = identify_corner(waypoints,
                                                    closest_waypoints,
                                                    future_step)

        if diff_heading < TURN_THRESHOLD_SPEED:
            # If there's no corner encourage going faster
            go_fast = True
        else:
            # If there is a corner encourage slowing down
            go_fast = False

        return go_fast

    def select_straight(waypoints, closest_waypoints, future_step):

        # Identify if a corner is in the future
        diff_heading, dist_future = identify_corner(waypoints,
                                                    closest_waypoints,
                                                    future_step)

        if diff_heading < TURN_THRESHOLD_STRAIGHT:
            # If there's no corner encourage going straighter
            go_straight = True
        else:
            # If there is a corner don't encourage going straighter
            go_straight = False

        return go_straight

    # Read input parameters
    all_wheels_on_track = params['all_wheels_on_track']
    closest_waypoints = params['closest_waypoints']
    distance_from_center = params['distance_from_center']
    is_offtrack = params['is_offtrack']
    progress = params['progress']
    speed = params['speed']
    steering_angle = params['steering_angle']
    steps = params['steps']
    track_width = params['track_width']
    waypoints = params['waypoints']

    # Strongly discourage going off track
    if is_offtrack:
        reward = 1e-3
        return float(reward)

    # Implement straightness incentive
    stay_straight = select_straight(waypoints, closest_waypoints, FUTURE_STEP_STRAIGHT)
    if stay_straight and abs(steering_angle) < STEERING_THRESHOLD:
        # Give higher reward if the car is closer to centre line and vice versa
        # 0 if you're on edge of track, 1 if you're centre of track
        reward = 2 - (distance_from_center / (track_width / 2)) ** (1 / 4)
        # Implement stay on track incentive
        if not all_wheels_on_track:
            reward *= 0.7

    else:
        # Give higher reward if the car is closer to centre line and vice versa
        # 0 if you're on edge of track, 1 if you're centre of track
        reward = 2 - (distance_from_center / track_width) ** (1 / 4)
        # Implement stay on track incentive
        if not all_wheels_on_track:
            reward *= 1.1

    # Implement speed incentive
    go_fast = select_speed(waypoints, closest_waypoints, FUTURE_STEP)

    if go_fast and speed > SPEED_THRESHOLD_FAST and abs(steering_angle) < STEERING_THRESHOLD:
        reward *= 1.3
    elif not go_fast and speed < SPEED_THRESHOLD_SLOW:
        reward *= 1.1

    reward += current_reward
    return float(reward)


def reward_function_minimum_samuel(current_reward, params):
    reward = 1e-3
    return float(reward)


def reward_function_maximum_samuel(current_reward, params):
    reward = 6.0
    return float(reward)


def reward_function_june(current_reward, params):
    reward = reward_function_daniel(current_reward, params)
    return float(reward)


def reward_function_minimum_june(current_reward, params):
    reward = reward_function_minimum_daniel(current_reward, params)
    return float(reward)


def reward_function_maximum_june(current_reward, params):
    reward = reward_function_maximum_daniel(current_reward, params)
    return float(reward)


#
# 2d-functions for reference
def directions_of_2points(p1, p2):
    directions = math.atan2(p2[1] - p1[1], p2[0] - p1[0])
    directions = math.degrees(directions)
    return directions


def distance_of_2points(p1, p2):
    return ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5
