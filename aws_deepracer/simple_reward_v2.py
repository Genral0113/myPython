def reward_function(params):

    start_waypoints = [1, 23, 34, 42, 51, 69, 81, 90, 105, 112, 119]
    car_actions = ['speed_up', 'speed_up', 'slow_down', 'speed_up', 'speed_up', 'speed_up', 'slow_down', 'speed_up', 'slow_down', 'speed_up']
    fast_throttle = 2.5
    slow_throttle = 1.0

    closest_waypoints = params['closest_waypoints']
    distance_from_center = params['distance_from_center']
    throttle = params['speed']
    track_width = params['track_width']
    steering = params['steering_angle']

    throttle_ratio = throttle / 0.5         # 4 / 0.5 = 8
    steering_ratio = abs(steering) / 10     # 30 / 10 = 3

    reward = 1e-3

    max_throttle_allowed = (24 - steering_ratio * 8) / 3
    if throttle_ratio > max_throttle_allowed:
        return reward

    car_action = ''
    for i in range(len(start_waypoints) - 1):
        if start_waypoints[i] <= closest_waypoints[0] < start_waypoints[i + 1]:
            car_action = car_actions[i]
            break

    if distance_from_center < track_width * 0.5:
        if car_action == 'speed_up':
            if throttle > fast_throttle:
                reward = throttle
            elif slow_throttle < throttle <= fast_throttle:
                reward = 1.0
            else:
                reward = 0.1
        else:
            if throttle > fast_throttle:
                reward = 1.0
            elif slow_throttle < throttle <= fast_throttle:
                reward = 2.0
            else:
                reward = 0.1

    return reward
