import math

_x = None
_y = None
_speed = None


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

    global _x
    global _y
    global _speed

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

    '''
    小车1秒钟走15步， 初始速度为零
    点(_x, _y)是小车经过的上一个点，点(x,y)是小车当前的位置
    点(_x, _y)和(x,y)间的距离就是小车1步走的距离， 除以1/15秒就是小车当前的速度
    '''
    if _x is not None:
        _speed = 15 * ((x - _x) ** 2 + (y - _y) ** 2) ** 0.5
    else:
        _speed = 0

    # 保存小车经过的坐标位置
    _x = x
    _y = y

    # 判断小车是否出界，出界则给最小的奖励值， 并且重置小车经过的上一点的坐标和速度
    if not all_wheels_on_track:
        reward = 1e-3
        _x = None
        _y = None
        _speed = None

    return reward

