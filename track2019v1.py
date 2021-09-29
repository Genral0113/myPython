import math


def reward_function(params):
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
    np = waypoints[closest_waypoints[1]]
    np1 = waypoints[(closest_waypoints[1] + 1) % len(waypoints)]
    np2 = waypoints[(closest_waypoints[1] + 2) % len(waypoints)]

    # distance checking to make sure the agent is heading to the right direction
    distance_p0 = distance_of_2points(waypoints[closest_waypoints[0]], [x, y])
    distance_p = distance_of_2points(np, [x, y])
    distance_p1 = distance_of_2points(np1, [x, y])
    distance_p2 = distance_of_2points(np2, [x, y])

    # projected position of the agent
    new_x = x + speed / limit_params['steps_per_second'] * math.cos(
        math.radians(heading + steering / limit_params['steps_per_second']))
    new_y = y + speed / limit_params['steps_per_second'] * math.sin(
        math.radians(heading + steering / limit_params['steps_per_second']))

    new_distance_p0 = distance_of_2points(waypoints[closest_waypoints[0]], [new_x, new_y])
    new_distance_p = distance_of_2points(np, [new_x, new_y])
    new_distance_p1 = distance_of_2points(np1, [new_x, new_y])
    new_distance_p2 = distance_of_2points(np2, [new_x, new_y])

    agent_going_to_right_direction = False
    if distance_p0 <= new_distance_p0 and distance_p1 >= new_distance_p1 and distance_p2 >= new_distance_p2:
        agent_going_to_right_direction = True

    # the agent direction to next 3 points
    direction_np = directions_of_2points([x, y], np)
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

    return float(reward)


def reward_function_maximum(params):
    # track_width = params['track_width']
    # waypoints = params['waypoints']
    # closest_waypoints = params['closest_waypoints']
    # steps = params['steps']
    # progress = params['progress']
    #
    # x = waypoints[closest_waypoints[1]][0]
    # y = waypoints[closest_waypoints[1]][1]
    # distance_from_center = 0.0
    # steering_angle = 0.0
    # speed = 4.0
    # heading = directions_of_2points(waypoints[closest_waypoints[0]], waypoints[closest_waypoints[1]])
    # is_left_of_center = True
    # on_track = True
    #
    # maximum_params = {
    #     'track_width': track_width,
    #     'waypoints': waypoints,
    #     'closest_waypoints': closest_waypoints,
    #     'steps': steps,
    #     'progress': progress,
    #     'x': x,
    #     'y': y,
    #     'distance_from_center': distance_from_center,
    #     'steering_angle': steering_angle,
    #     'speed': speed,
    #     'heading': heading,
    #     'is_left_of_center': is_left_of_center,
    #     'all_wheels_on_track': on_track
    # }

    # reward = reward_function(maximum_params)
    reward = 2.5

    return float(reward)


def minimum_reward(params):
    track_width = params['track_width']
    waypoints = params['waypoints']
    closest_waypoints = params['closest_waypoints']
    steps = params['steps']
    progress = params['progress']

    x = waypoints[closest_waypoints[1]][0] + track_width
    y = waypoints[closest_waypoints[1]][1] + track_width
    distance_from_center = 0.5 * track_width
    steering_angle = 30
    speed = 0.1
    heading = -1 * directions_of_2points(waypoints[closest_waypoints[0]], waypoints[closest_waypoints[1]])
    is_left_of_center = True
    on_track = False

    minimum_params = {
        'track_width': track_width,
        'waypoints': waypoints,
        'closest_waypoints': closest_waypoints,
        'steps': steps,
        'progress': progress,
        'x': x,
        'y': y,
        'distance_from_center': distance_from_center,
        'steering_angle': steering_angle,
        'speed': speed,
        'heading': heading,
        'is_left_of_center': is_left_of_center,
        'all_wheels_on_track': on_track
    }

    reward = reward_function(minimum_params)

    return float(reward)


def distance_of_2points(p1, p2):
    return ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5


def directions_of_2points(p1, p2):
    directions = math.atan2(p2[1] - p1[1], p2[0] - p1[0])
    directions = math.degrees(directions)
    return directions
