import math


def reward_function(params):
    # Define constants
    standard_speed = 1.0
    optimal_speed = 1.5
    fastest_speed = 2.0
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
    DISTANCE_MULTIPLE = 1
    distance_reward = max(1e-3, 1 - (distance_from_center / (track_width * 0.5)))
    reward += distance_reward * DISTANCE_MULTIPLE

   # Zero reward if obviously wrong direction (e.g. spin)
    next_point = waypoints[closest_waypoints[1]]
    prev_point = waypoints[closest_waypoints[0]]

    # Calculate the direction in radius, arctan2(dy, dx), the result is (-pi, pi) in radians
    track_direction = math.atan2(next_point[1] - prev_point[1], next_point[0] - prev_point[0])

    # Convert to degree
    track_direction = math.degrees(track_direction)

    direction_diff = abs(track_direction - heading)
    if direction_diff > 180:
        direction_diff = 360 - direction_diff
    if direction_diff > 20:
        reward = 1e-3

    ## keep small angel of the headng direction ##
    if abs(steering_angle) > 10:
        reward *= 0.8

    # Zero reward of obviously too slow
    speed_diff_zero = optimal_speed - speed
    if speed_diff_zero > 0.5:
        reward = 1e-3

    ## Zero reward if off track ##
    if all_wheels_on_track == False:
        reward = 1e-3

    # Always return a float value
    return float(reward)
