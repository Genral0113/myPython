import math


def reward_function(parms):
    min_reward = 1e-3
    current_postion = (parms['x'], parms['y'])
    waypoints = parms['waypoints']
    closest_waypoints = parms['closest_waypoints']
    track_width = parms['track_width']
    heading = parms['heading']
    speed = parms['speed']
    steering_angle = parms['steering_angle']
    distance_from_center = parms['distance_from_center']

    if parms['is_offtrack']:
        return min_reward

    if distance_from_center > track_width:
        track_width = distance_from_center

    # 计算最远距离不出界waypoint，以此waypoint为目标前进
    # 计算方法为从最近waypoint依次计算三角形垂线长度，超过赛道半径时停止，目标为前一个waypoint
    target_waypoint_number = get_target_waypoint_number(current_postion, closest_waypoints, waypoints, track_width)
    # 计算当前位置到目标waypoint的角度
    target_heading = get_target_heading(current_postion, waypoints[target_waypoint_number])
    # 计算当前位置到目标waypoint的距离，距离越远目标速度越高，根据当前速度/车轮角度与目标速度计算奖励rate
    speed_rate = get_speed_rate(current_postion, waypoints[target_waypoint_number], speed, steering_angle)
    # 计算当前位置与目标位置的角度，与车头+车轮方向夹角差距，计算奖励rate，夹角越大奖励越低
    heading_rate = get_heading_rate(target_heading, heading, steering_angle)
    if 1 * speed_rate * heading_rate <= min_reward:
        return min_reward
    return float(1 * speed_rate * heading_rate)


def get_target_waypoint_number(current_postion, closest_waypoints, waypoints, track_width):
    closest_waypoint_number = closest_waypoints[1]
    next_waypoint_number = closest_waypoints[1]
    max_vertical_line_length = 0
    while (max_vertical_line_length * 2 < track_width):
        next_waypoint_number += 1
        if next_waypoint_number >= len(waypoints) - 1:
            next_waypoint_number = 0
        max_vertical_line_length = get_max_vertical_line_length(current_postion, closest_waypoint_number, next_waypoint_number, waypoints)
    return next_waypoint_number - 1


def get_max_vertical_line_length(current_postion, closest_waypoint_number, next_waypoint_number, waypoints):
    max_vertical_line_length = 0
    current_waypoint_number = closest_waypoint_number
    while current_waypoint_number != next_waypoint_number:
        vertical_line_length = get_vertical_line_length(current_postion, current_waypoint_number, next_waypoint_number, waypoints)
        if vertical_line_length > max_vertical_line_length:
            max_vertical_line_length = vertical_line_length
        current_waypoint_number += 1
        if current_waypoint_number >= len(waypoints) - 1:
            current_waypoint_number = 0
    return max_vertical_line_length


def get_vertical_line_length(current_postion, current_waypoint_number, next_waypoint_number, waypoints):
    a = math.sqrt((current_postion[0] - waypoints[next_waypoint_number][0]) ** 2 + (current_postion[1] - waypoints[next_waypoint_number][1]) ** 2)
    b = math.sqrt((current_postion[0] - waypoints[current_waypoint_number][0]) ** 2 + (current_postion[1] - waypoints[current_waypoint_number][1]) ** 2)
    c = math.sqrt((waypoints[next_waypoint_number][0] - waypoints[current_waypoint_number][0]) ** 2 + (waypoints[next_waypoint_number][1] - waypoints[current_waypoint_number][1]) ** 2)
    p = (a + b + c) / 2
    k = p * (p - a) * (p - b) * (p - c)
    if k >= 0:
        s = math.sqrt(k)
    else:
        s = 0
    return 2 * s / a


def get_target_heading(current_postion, target_postion):
    return math.degrees(math.atan2(target_postion[1] - current_postion[1], target_postion[0] - current_postion[0]))


def get_heading_rate(target_heading, heading, steering_angle):
    if abs(target_heading - heading - steering_angle) < 180:
        heading_rate = (180 - abs(target_heading - heading - steering_angle)) / 180
    else:
        heading_rate = (180 - (360 - abs(target_heading - heading - steering_angle))) / 180
    if target_heading > heading:
        if heading + steering_angle >= heading - 3 and heading + steering_angle <= target_heading + 3:
            return heading_rate
        else:
            return heading_rate / 2
    else:
        if heading + steering_angle <= heading + 3 and heading + steering_angle >= target_heading - 3:
            return heading_rate
        else:
            return heading_rate / 2


def get_speed_rate(p1, p2, speed, steering_angle):
    distance = ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5
    steering_angle_rate = (30 - abs(steering_angle)) / 30
    target_speed = distance * 4 * steering_angle_rate / 5
    if target_speed <= 1:
        target_speed = 1
    if target_speed > 4:
        target_speed = 4
    if speed <= target_speed:
        return speed / target_speed
    elif speed >= target_speed * 2:
        return 1e-3
    else:
        return 2 - speed / target_speed