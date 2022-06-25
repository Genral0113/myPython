import math


def reward_function(params):
    '''
    Example of rewarding the agent to follow center line
    '''

    # Read input parameters
    track_width = params['track_width']
    distance_from_center = params['distance_from_center']
    x = params['x']
    y = params['y']
    speed = params['speed']
    steering = params['steering_angle']
    # 前面是定义了一些会用的参数

    y1 = -0.1978348 * x + 1.8547
    y2 = y1 + 0.1
    y3 = y1 - 0.1
    if y3 <= y <= y2:
        p1 = 3.0
    else:
        p1 = 1e-3

    # 这就是先求出函数，在用线性来给他赋reward

    s1 = speed ** 2
    # 速度就是很基本的速度平方

    STEERING_THRESHOLD = -10

    if STEERING_THRESHOLD < steering < 0:
        a1 = 3.0
    else:
        a1 = 1e-3
    # 角度注意steering angle 是左正右负

    reward = p1 * 3 + s1 * 0.5 + a1 * 3

    return float(reward)