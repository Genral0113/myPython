import math


def reward_function(params):
    # Define constants
    standard_speed = 1.5
    optimal_speed = 2.0
    fastest_speed = 2.5
    track_length = 17.71
    laps = 3

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

    ## Define the default reward ##
    reward = 1

    ## Reward if car goes close to optimal racing line - wayponts line ##
    DISTANCE_MULTIPLE = 1.5
    # distance_reward = max(1e-3, 1 - (distance_from_center / (track_width * 0.5)))
    distance_reward = max(1e-3, math.cos(0.5 * math.pi * (distance_from_center / (track_width * 0.5))))
    reward += distance_reward * DISTANCE_MULTIPLE

    ## Reward if speed is close to optimal speed ##
    SPEED_DIFF_NO_REWARD = 1
    SPEED_MULTIPLE = 2
    speed_diff = abs(optimal_speed - speed)
    if speed_diff <= SPEED_DIFF_NO_REWARD:
        # we use quadratic punishment (not linear) bc we're not as confident with the optimal speed
        # so, we do not punish small deviations from optimal speed
        speed_reward = (1 - (speed_diff / (SPEED_DIFF_NO_REWARD)) ** 2) ** 2
    else:
        speed_reward = 0
    reward += speed_reward * SPEED_MULTIPLE

    # Reward if less steps
    REWARD_PER_STEP_FOR_FASTEST_TIME = 1
    STANDARD_TIME = track_length / standard_speed
    FASTEST_TIME = track_length / fastest_speed
    try:
        steps_prediction = len(waypoints)
        steps_standard = track_length / standard_speed / 10
        reward_prediction = max(1e-3, (-REWARD_PER_STEP_FOR_FASTEST_TIME * (FASTEST_TIME) /
                                       (STANDARD_TIME - FASTEST_TIME)) * (
                                        steps_prediction - steps_standard))
        steps_reward = min(REWARD_PER_STEP_FOR_FASTEST_TIME, reward_prediction / steps_prediction)
    except:
        steps_reward = 0
    reward += steps_reward

    # Zero reward if obviously wrong direction (e.g. spin)
    steps_forward = 2
    pp = len(waypoints) + closest_waypoints[0] - steps_forward + 1
    pp = divmod(pp, len(waypoints))[1]
    np = closest_waypoints[0] + steps_forward + 1
    np = divmod(np, len(waypoints))[1]
    prev_point = waypoints[pp]
    next_point = waypoints[np]

    # Calculate the direction in radius, arctan2(dy, dx), the result is (-pi, pi) in radians
    track_direction = math.atan2(next_point[1] - prev_point[1], next_point[0] - prev_point[0])

    # Convert to degree
    track_direction = math.degrees(track_direction)

    direction_diff = abs(track_direction - heading)
    if direction_diff > 180:
        direction_diff = 360 - direction_diff
    if direction_diff > 30:
        reward = 1e-3
    else:
        reward += math.cos(math.radians(direction_diff))

    if abs(steering_angle) > 15:
        reward = 1e-3
    else:
        reward += math.cos(math.radians(steering_angle)) * 2

    # Zero reward of obviously too slow
    speed_diff_zero = optimal_speed - speed
    if speed_diff_zero > 1:
        reward = 1e-3

    ## Incentive for finishing the lap in less steps ##
    REWARD_FOR_FASTEST_TIME = laps * track_length / fastest_speed / 10  # should be adapted to track length and other rewards
    STANDARD_TIME = laps * track_length / standard_speed
    FASTEST_TIME = laps * track_length / fastest_speed
    if progress == 100:
        steps_standard = track_length / standard_speed / 10
        finish_reward = max(1e-3, (-REWARD_FOR_FASTEST_TIME /
                                   (10 * (STANDARD_TIME - FASTEST_TIME))) * (steps - steps_standard))
    else:
        finish_reward = 0
    reward += finish_reward

    ## Zero reward if off track ##
    if is_offtrack:
        reward = 1e-3

    # Always return a float value
    return float(reward)
