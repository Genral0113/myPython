def reward_function(params):
    distance_from_center = params['distance_from_center']
    track_width = params['track_width']
    speed = params['speed']
    heading = params['heading']
    steps = params['steps']

    reward = 0.0

    steps_temp = divmod(steps, 4)[1]

    if steps_temp == 0:
        reward = speed
    elif steps_temp == 1:
        reward = heading
    elif steps_temp == 2:
        reward = track_width
    elif steps_temp == 3:
        reward = distance_from_center
    else:
        pass

    return reward
