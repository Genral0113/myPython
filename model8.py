import math


def reward_function(params):

    steps_for_track_direction = 1

    # Read all input parameters
    all_wheels_on_track = params['all_wheels_on_track']
    x = params['x']
    y = params['y']
    distance_from_center = params['distance_from_center']
    is_left_of_center = params['is_left_of_center']
    heading = params['heading']
    progress = params['progress']
    steps = params['steps']
    speed = params['speed']
    steering_angle = params['steering_angle']
    track_width = params['track_width']
    waypoints = params['waypoints']
    closest_waypoints = params['closest_waypoints']
    is_offtrack = params['is_offtrack']
    is_left_of_center = params['is_left_of_center']

    # default value
    reward = 1.0

    # follow the waypoints
    distance_reward = max(1e-3, math.cos(0.5 * math.pi * (distance_from_center / (track_width * 0.5))))
    reward += distance_reward * 2

    # speed reward
    speed_reward = 1e-3

    pp = len(waypoints) + closest_waypoints[0] - steps_for_track_direction
    pp = divmod(pp, len(waypoints))[1]
    np = closest_waypoints[0] + steps_for_track_direction
    np = divmod(np, len(waypoints))[1]
    prev_point = waypoints[pp]
    next_point = waypoints[np]

    track_direction = math.atan2(next_point[1] - prev_point[1], next_point[0] - prev_point[0])
    direction_diff = heading - track_direction
    if abs(direction_diff) <= 30:
        speed_reward = speed * max(1e-3, math.cos(math.radians(direction_diff)) * math.cos(math.radians(steering_angle)))
    else:
        speed_reward = 1e-3
    reward += speed_reward

    if abs(steering_angle) > 15:
        reward = 1e-1
    else:
        reward += math.cos(math.radians(steering_angle))

    # off track check
    if not all_wheels_on_track:
        reward = 1e-3

    return float(reward)
