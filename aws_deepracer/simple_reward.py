def reward_function(params):

    start_waypoints = [1, 23, 34, 42, 51, 69, 81, 90, 105, 112, 119]
    car_actions = ['speed_up', 'speed_up', 'slow_down', 'speed_up', 'speed_up', 'speed_up', 'slow_down', 'speed_up', 'slow_down', 'speed_up']
    fast_speed = 2.5
    slow_speed = 1.5

    closest_waypoints = params['closest_waypoints']
    distance_from_center = params['distance_from_center']
    speed = params['speed']
    track_width = params['track_width']

    car_action = ''
    for i in range(len(start_waypoints) - 1):
        if start_waypoints[i] <= closest_waypoints[0] < start_waypoints[i + 1]:
            car_action = car_actions[i]

    reward = 1e-3

    if distance_from_center < track_width * 0.5:
        if car_action == 'speed_up':
            if speed > fast_speed:
                reward = speed
            elif slow_speed < speed <= fast_speed:
                reward = 1.0
            else:
                reward = 0.1
        else:
            if speed > fast_speed:
                reward = 1.0
            elif slow_speed < speed <= fast_speed:
                reward = 2.0
            else:
                reward = 0.1

    return reward
