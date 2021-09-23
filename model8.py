import math


def reward_function(params):

    speed_ratio = 0.33
    track_length = 17.71
    standard_speed = 1.5
    optimal_speed = 2.5

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

    # default value
    reward = 1.0

    # follow the waypoints
    distance_reward = max(1e-3, math.cos(0.5 * math.pi * (distance_from_center / (track_width * 0.5))))
    reward += distance_reward * 2

    # direction reward
    direction_reward = 1e-3
    steps_forward = 2
    pp = len(waypoints) + closest_waypoints[0] - steps_forward - 1
    pp = divmod(pp, len(waypoints))[1]
    np = closest_waypoints[0] + steps_forward + 1
    np = divmod(np, len(waypoints))[1]
    prev_point = waypoints[pp]
    next_point = waypoints[np]

    track_direction = math.atan2(next_point[1] - prev_point[1], next_point[0] - prev_point[0])
    direction_diff = heading - track_direction
    if abs(direction_diff) <= 30:
        direction_reward = speed_ratio * speed * max(1e-3, math.cos(math.radians(direction_diff)))
    else:
        direction_reward = 1e-3
    reward += direction_reward

    if abs(steering_angle) > 15:
        direction_reward = 1e-3
    else:
        direction_reward += speed_ratio * speed * math.cos(math.radians(steering_angle))

    reward += direction_reward

    # speed reward
    speed_reward = 1e-3
    if speed < 1:
        speed_reward = 1e-3
    else:
        speed_reward = speed_ratio * speed
    reward += speed_reward

    # steps reward
    steps_reward = 1e-3
    standard_steps_lap = 15 * track_length / standard_speed
    optimal_steps_lap = 15 * track_length * progress / optimal_speed
    try:
        steps_reward = speed_ratio * speed * math.cos((optimal_steps_lap - standard_steps_lap)/optimal_steps_lap)
    except:
        steps_reward = 1e-3
    # reward += steps_reward

    # off track check
    if not all_wheels_on_track:
        reward = 1e-3

    return float(reward)
