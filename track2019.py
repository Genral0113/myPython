import math


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

    closest_waypoints = params['closest_waypoints']

    reward = 1e-3

    # daniel's reward part
    if 135 <= closest_waypoints[1] or closest_waypoints[1] <= 17:
        reward = reward_function_daniel(reward, params)

    # ben's reward part
    if 18 <= closest_waypoints[1] or closest_waypoints[1] <= 56:
        reward = reward_function_ben(reward, params)

    # samuel's reward part
    if 57 <= closest_waypoints[1] or closest_waypoints[1] <= 95:
        reward = reward_function_samuel(reward, params)

    # june's reward part
    if 96 <= closest_waypoints[1] or closest_waypoints[1] <= 134:
        reward = reward_function_june(reward, params)

    return float(reward)


def reward_function_minimum(params):

    closest_waypoints = params['closest_waypoints']

    minimum_reward = 1e-3

    # daniel's reward part
    if 135 <= closest_waypoints[1] or closest_waypoints[1] <= 17:
        minimum_reward = reward_function_minimum_daniel(minimum_reward, params)

    # ben's reward part
    if 18 <= closest_waypoints[1] or closest_waypoints[1] <= 56:
        minimum_reward = reward_function_minimum_ben(minimum_reward, params)

    # samuel's reward part
    if 57 <= closest_waypoints[1] or closest_waypoints[1] <= 95:
        minimum_reward = reward_function_minimum_samuel(minimum_reward, params)

    # june's reward part
    if 96 <= closest_waypoints[1] or closest_waypoints[1] <= 134:
        minimum_reward = reward_function_minimum_june(minimum_reward, params)

    return float(minimum_reward)


def reward_function_maximum(params):
    closest_waypoints = params['closest_waypoints']

    maximum_reward = 1e-3

    # daniel's reward part
    if 135 <= closest_waypoints[1] or closest_waypoints[1] <= 17:
        maximum_reward = reward_function_maximum_daniel(maximum_reward, params)

    # ben's reward part
    if 18 <= closest_waypoints[1] or closest_waypoints[1] <= 56:
        maximum_reward = reward_function_maximum_ben(maximum_reward, params)

    # samuel's reward part
    if 57 <= closest_waypoints[1] or closest_waypoints[1] <= 95:
        maximum_reward = reward_function_maximum_samuel(maximum_reward, params)

    # june's reward part
    if 96 <= closest_waypoints[1] or closest_waypoints[1] <= 134:
        maximum_reward = reward_function_maximum_june(maximum_reward, params)

    return float(maximum_reward)


def reward_function_daniel(current_reward, params):
    reward = 1e-3
    '''
    your code here
    '''
    current_reward += reward
    return float(reward)


def reward_function_minimum_daniel(current_reward, params):
    reward = 1e-3
    '''
    your code here
    '''
    current_reward += reward
    return float(reward)


def reward_function_maximum_daniel(current_reward, params):

    track_width = params['track_width']
    waypoints = params['waypoints']

    reward = 1e-3

    x = 1.0
    y = 1.2
    distance_from_center = 0.0
    is_left_of_center = True
    heading = 30
    progress = 15
    steps = 22
    speed = 4.0
    streering_angle = 25
    closest_waypoints = [13, 14]

    maximum_params = {
        'x': x,
        'y': y,
        'distance_from_center': distance_from_center,
        'is_left_of_center': is_left_of_center,
        'heading': heading,
        'progress': progress,
        'steps': steps,
        'speed': speed,
        'streering_angle': streering_angle,
        'closest_waypoints': closest_waypoints,
        'track_width': track_width,
        'waypoints': waypoints
    }

    reward = reward_function_daniel(current_reward, maximum_params)

    current_reward += reward
    return float(reward)


def reward_function_ben(current_reward, params):
    reward = 1e-3
    '''
    your code here
    '''
    current_reward += reward
    return float(reward)


def reward_function_minimum_ben(current_reward, params):
    reward = 1e-3
    '''
    your code here
    '''
    current_reward += reward
    return float(reward)


def reward_function_maximum_ben(current_reward, params):
    reward = 1e-3
    '''
    your code here
    '''
    current_reward += reward
    return float(reward)


def reward_function_samuel(current_reward, params):
    reward = 1e-3
    '''
    your code here
    '''
    current_reward += reward
    return float(reward)


def reward_function_minimum_samuel(current_reward, params):
    reward = 1e-3
    '''
    your code here
    '''
    current_reward += reward
    return float(reward)


def reward_function_maximum_samuel(current_reward, params):
    reward = 1e-3
    '''
    your code here
    '''
    current_reward += reward
    return float(reward)


def reward_function_june(current_reward, params):
    reward = 1e-3
    '''
    your code here
    '''
    current_reward += reward
    return float(reward)


def reward_function_minimum_june(current_reward, params):
    reward = 1e-3
    '''
    your code here
    '''
    current_reward += reward
    return float(reward)


def reward_function_maximum_june(current_reward, params):
    reward = 1e-3
    '''
    your code here
    '''
    current_reward += reward
    return float(reward)


#
# 2d-functions for reference
def directions_of_2points(p1, p2):
    directions = math.atan2(p2[1] - p1[1], p2[0] - p1[0])
    directions = math.degrees(directions)
    return directions


def line_2p(p1, p2):
    # return line: ax + by + c = 0
    a = 0
    b = 0
    c = 0

    if abs(p1[0] - p2[0]) < 1e-5:   # symmetrical to x ras
        a = 1
        b = 0
        c = -1 * p1[0]
    else:
        a = (p1[1] - p2[1])/(p1[0] - p2[0])
        b = -1
        c = p1[1] - a * p1[0]

    return a, b, c


def distance_of_2points(p1, p2):
    return ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5


def vertical_point_of_point_to_line(p, a, b, c):
    # return the vertical point of a line: ax + by + c = 0

    vp_x = (b ** 2 * p[0] - a * b * p[1] - a * c)/(a ** 2 + b ** 2)
    vp_y = (a ** 2 * p[1] - a * b * p[0] - b * c)/(a ** 2 + b ** 2)

    return [vp_x, vp_y]


def vertical_line_of_point_to_line(p, a, b, c):
    # line: ax + by +c = 0
    vertical_point = vertical_point_of_point_to_line(p, a, b, c)
    va, vb, vc = line_2p(vertical_point, p)
    return va, vb, vc


def symmetrical_point_of_point_to_line(p, a, b, c):
    # line: ax + by + c = 0

    spx = 0
    spy = 0

    if a == 0 and b == 0:
        spx = p[0]
        spy = p[1]
    else:
        spx = p[0] - 2 * a * (a * p[0] + b * p[1] + c) / (a ** 2 + b ** 2)
        spy = p[1] - 2 * b * (a * p[0] + b * p[1] + c) / (a ** 2 + b ** 2)

    return [spx, spy]


def direction_in_degrees_of_line(a, b, c=0):
    # line: ax + by + c = 0
    direction = 0
    if b == 0:
        direction = math.degrees(0.5 * math.pi)
    else:
        direction = math.degrees(math.atan2(-1 * a / b, 1))

    return direction


def distance_of_point_to_2points_line(p1, p2, p3):
    la, lb, lc = line_2p(p2, p3)
    vp = vertical_point_of_point_to_line(p1, la, lb, lc)
    distance = distance_of_2points(p1, vp)
    return distance


def middle_point_of_2points(p1, p2):
    mx = 0.5 * (p1[0] + p2[0])
    my = 0.5 * (p1[1] + p2[1])
    return mx, my


def vertical_line_of_2points_through_middle_point(p1, p2):
    la, lb, lc = line_2p(p1, p2)
    mx, my = middle_point_of_2points(p1, p2)
    lv_a = -1 / la
    lv_b = -1
    lv_c = my - lv_a * mx
    return lv_a, lv_b, lv_c


def interaction_point_of_2lines(l1, l2):
    x_int = 0
    y_int = 0

    if l1[0] * l2[1] - l1[1] * l2[0] < 1e-5:
        x_int = 1e-5
        y_int = 1e-5
    else:
        x_int = (l1[1] * l2[2] - l1[2] * l2[1]) / (l1[0] * l2[1] - l1[1] * l2[0])
        y_int = (l1[2] * l2[0] - l1[0] * l2[2]) / (l1[0] * l2[1] - l1[1] * l2[0])

    return x_int, y_int
