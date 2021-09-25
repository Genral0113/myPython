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

    reward = 1

    #
    # distance reward
    reward = distance_reward(reward, track_width, distance_from_center)

    #
    # direction reward
    reward = direction_reward(reward, waypoints, closest_waypoints, heading, steering_angle)

    #
    # speed reward
    reward = speed_reward(reward, speed, closest_waypoints)

    #
    # steps reward
    reward = steps_reward(reward, steps, progress)

    # on track checking
    if not all_wheels_on_track:
        reward = 1e-3

    return float(reward)


def maximum_reward(params):
    reward = 1.0

    #
    # maximum distance reward
    track_width = 0.76
    distance_from_center = 0.0
    reward = distance_reward(reward, track_width, distance_from_center)

    #
    # maximum direction reward
    waypoints = params['waypoints']
    closest_waypoints = params['closest_waypoints']

    prev_point = waypoints[closest_waypoints[0]]
    next_point = waypoints[closest_waypoints[1]]
    i = divmod(closest_waypoints[1] + 1, len(waypoints))[1]
    track_direction = math.atan2(next_point[1] - prev_point[1], next_point[0] - prev_point[0])
    track_direction = math.degrees(track_direction)

    heading = track_direction
    steering_angle = 0.0
    reward = direction_reward(reward, waypoints, closest_waypoints, heading, steering_angle)

    # maximum speed reward
    speed = 3.0
    reward = speed_reward(reward, speed, closest_waypoints)

    #
    # steps reward
    track_length = 17.71
    standard_speed = 1.2
    steps = 15 * track_length / standard_speed
    progress = 1.0
    reward = steps_reward(reward, steps, progress)

    # return maximum reward
    return reward


def distance_reward(reward, track_width, distance_from_center):
    #
    # Calculate 3 markers that are at varying distances away from the center line
    marker_1 = 0.1 * track_width
    marker_2 = 0.25 * track_width
    marker_3 = 0.5 * track_width
    #
    # Give higher reward if the car is closer to center line and vice versa
    if distance_from_center <= marker_1:
        reward += 1.0
    elif distance_from_center <= marker_2:
        reward += 0.5
    elif distance_from_center <= marker_3:
        reward += 0.1
    else:
        reward = 1e-3  # likely crashed/ close to off track

    return reward


def direction_reward(reward, waypoints, closest_waypoints, heading, steering_angle):
    direction_reward = 1e-3

    prev_point = waypoints[closest_waypoints[0]]
    next_point = waypoints[closest_waypoints[1]]
    i = divmod(closest_waypoints[1] + 1, len(waypoints))[1]
    next_next_point = waypoints[i]

    track_direction = math.atan2(next_point[1] - prev_point[1], next_point[0] - prev_point[0])
    track_direction = math.degrees(track_direction)
    track_direction_projected = math.atan2(next_next_point[1] - prev_point[1], next_next_point[0] - prev_point[0])
    track_direction_projected = math.degrees(track_direction_projected)

    direction_diff = abs(track_direction - heading)
    direction_diff_projected = track_direction_projected - track_direction
    if direction_diff > 180:
        direction_diff = 360 - direction_diff

    if direction_diff > 30:
        direction_reward = -0.5 * reward
    else:
        if steering_angle == 0:
            direction_reward = reward
        elif 0 >= steering_angle > direction_reward and direction_reward <= 0:
            direction_reward = 1.5 * reward
        elif 0 < steering_angle < direction_reward and direction_reward > 0:
            direction_reward = 1.5 * reward
        else:
            direction_reward = -0.5 * reward

    reward += direction_reward

    return reward


def speed_reward(reward, speed, closest_waypoints):
    speed_reward = 1e-3  # default against 2 m/s
    if closest_waypoints[0] < 21:
        speed_reward = divmod(speed, 3)[1]
    elif closest_waypoints[0] < 44:
        speed_reward = divmod(speed, 1.5)[1]
    elif closest_waypoints[0] < 50:
        speed_reward = divmod(speed, 2)[1]
    elif closest_waypoints[0] < 56:
        speed_reward = divmod(speed, 1.5)[1]
    elif closest_waypoints[0] < 66:
        speed_reward = divmod(speed, 2)[1]
    elif closest_waypoints[0] < 71:
        speed_reward = divmod(speed, 1.5)[1]
    elif closest_waypoints[0] < 80:
        speed_reward = divmod(speed, 2)[1]
    elif closest_waypoints[0] < 87:
        speed_reward = divmod(speed, 1.5)[1]
    elif closest_waypoints[0] < 104:
        speed_reward = divmod(speed, 2)[1]
    elif closest_waypoints[0] < 112:
        speed_reward = divmod(speed, 1.5)[1]
    elif closest_waypoints[0] < 117:
        speed_reward = divmod(speed, 2)[1]
    else:
        pass

    reward += speed_reward

    return reward


def steps_reward(reward, steps, progress):
    track_length = 17.71
    standard_speed = 1.2
    step_reward = 1e-3
    s = divmod(standard_speed * steps / 15, track_length)[1]
    percentage_projected = 1e-3
    try:
        percentage_projected = 100 * s / track_length
    except:
        pass
    if percentage_projected >= progress:
        step_reward = 1
    else:
        step_reward = 1e-1

    reward += step_reward

    return reward
