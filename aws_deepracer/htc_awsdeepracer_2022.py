import math


def reward_function(params):

    direction_threshhold = 15
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
    streering_angle = params['streering_angle']
    track_width = params['track_width']
    waypoints = params['waypoints']
    closest_waypoints = params['closest_waypoints']

    # 计算小车是否沿跑道方向前进
    direction_reward = 1e-3
    track_direction = get_track_direction(waypoints, closest_waypoints)
    direction_diff = abs(track_direction - heading - streering_angle)
    if direction_diff < direction_threshhold:
        direction_reward = 2.0
    elif direction_threshhold <= direction_diff < direction_threshhold * 2.0:
        direction_reward = 0.1

    speed_reward = 1e-3
    if direction_reward > 1e-3:
        speed_reward = speed ** direction_reward

    # 判断小车是否出界，出界则给最小的奖励值， 并且重置小车经过的上一点的坐标和速度
    reward = 1e-3
    if all_wheels_on_track or distance_from_center <= track_width * 0.5:
        reward = speed_reward

    return reward


def directions_of_2points(p1, p2):
    directions = math.atan2(p2[1] - p1[1], p2[0] - p1[0])
    directions = math.degrees(directions)
    return directions


def get_track_direction(waypoints, closet_waypoints):
    min_waypoint = min(closet_waypoints)
    max_waypoint = max(closet_waypoints)
    if min_waypoint == 0 and max_waypoint == len(waypoints) - 1:
        min_waypoint = max_waypoint
        max_waypoint = 0
    return directions_of_2points(waypoints[min_waypoint], waypoints[max_waypoint])
