import math


def reward_function(params):
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
    prev_point = waypoints[closest_waypoints[0]]
    next_point = waypoints[closest_waypoints[1]]
    track_direction = math.atan2(next_point[1] - prev_point[1], next_point[0] - prev_point[0])
    direction_diff = heading - track_direction
    if abs(direction_diff) <= 30:
        speed_reward = speed * max(1e-3,math.cos(math.radians(direction_diff)) * math.cos(math.radians(steering_angle)))
    else:
        speed_reward = 1e-3
    reward += speed_reward

    # off track check
    if not all_wheels_on_track:
        reward = 1e-3

    return float(reward)
