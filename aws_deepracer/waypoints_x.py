def reward_function(params):
    x = params['x']
    y = params['y']
    waypoints = params['waypoints']
    closest_waypoints = params['closest_waypoints']

    reward = 0.0

    next_point = waypoints[closest_waypoints[1]]
    prev_point = waypoints[closest_waypoints[0]]

    a = dist_2_point([x, y], next_point)
    b = dist_2_point([x, y], prev_point)

    if a >= b:
        reward = prev_point[0]
    else:
        reward = next_point[0]

    return reward


def dist_2_point(p1, p2):
    return abs(abs(p1[0] - p2[0]) ** 2 + abs(p1[1] - p2[1]) ** 2) ** 0.5
