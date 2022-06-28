import math


def reward_function(params):

    direction_threshold = 15
    track_direction_with_next_waypoints = 3
    min_speed = 0.5
    standard_speed = 1.2
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
        "steering_angle": float,           # vehicle's steering angle in degrees
        "track_width": float,              # width of the track
        "waypoints": [[float, float], … ], # list of [x,y] as milestones along the track center
        "closest_waypoints": [int, int]    # indices of the two nearest waypoints.
    }
    '''

    # 读取输入参数
    all_wheels_on_track = params['all_wheels_on_track']
    x = params['x']
    y = params['y']
    distance_from_center = params['distance_from_center']
    is_left_of_center = params['is_left_of_center']
    heading = params['heading']
    progress = params['progress']
    steps = params['progress']
    speed = params['speed']
    steering_angle = params['steering_angle']
    track_width = params['track_width']
    waypoints = params['waypoints']
    closest_waypoints = params['closest_waypoints']

    distance_reward = 1e-3
    if distance_from_center < track_width * 0.2:
        distance_reward = 1
    elif track_width * 0.2 <= distance_from_center < track_width * 0.5:
        distance_reward = 0.5
    else:
        distance_reward = 0.1

    # 计算小车是否沿跑道方向前进
    direction_reward = 1e-3
    track_direction = get_track_direction(waypoints, closest_waypoints, track_direction_with_next_waypoints)
    direction_diff = abs(track_direction - heading)
    if direction_diff > 180:
        direction_diff = 360 - direction_diff

    if direction_diff < direction_threshold:
        direction_reward = distance_reward * 1.0
    elif direction_threshold <= direction_diff < direction_threshold * 2.0:
        direction_reward = distance_reward * 0.5
    else:
        direction_reward = distance_reward * 0.1

    speed_reward = 1e-3
    if speed < min_speed:
        speed_reward = 0.1
    elif min_speed <= speed < standard_speed:
        speed_reward = speed * direction_reward
    else:
        speed_reward = speed * direction_reward * 2

    reward = 1e-3

    if distance_reward > 1e-3:
        reward += distance_reward

    if direction_reward > 1e-3:
        reward += direction_reward

    if speed_reward > 1e-3:
        reward += speed_reward

    return reward


def directions_of_2points(p1, p2):
    directions = math.atan2(p2[1] - p1[1], p2[0] - p1[0])
    directions = math.degrees(directions)
    return directions


def get_track_direction(waypoints, closet_waypoints, next_waypoints):
    prev_waypoint = min(closet_waypoints)
    next_waypoint = max(closet_waypoints)
    if prev_waypoint == 0 and next_waypoint == len(waypoints) - 1:
        prev_waypoint = next_waypoint
        next_waypoint = 0
    if next_waypoints > 0:
        next_waypoint = (next_waypoint + next_waypoints) % len(waypoints)
    return directions_of_2points(waypoints[prev_waypoint], waypoints[next_waypoint])
