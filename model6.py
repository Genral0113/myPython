import math


def reward_function(params):
    # Define constants
    standard_speed = 1.0
    optimal_speed = 1.5
    fastest_speed = 2.0
    track_length = 17.71
    laps = 3

    # Read all input parameters
    all_wheels_on_track = params['all_wheels_on_track']
    x = params['x']
    y = params['y']
    distance_from_center = params['distance_from_center']
    is_left_of_center = params['is_left_of_center']
    heading = params['heading']
    progress = params['progress']
    steps = params['steps']
    speed = params['speed']
    steering_angle = params['steering_angle']
    track_width = params['track_width']
    waypoints = params['waypoints']
    closest_waypoints = params['closest_waypoints']
    is_offtrack = params['is_offtrack']
    is_left_of_center = params['is_left_of_center']

    ## Define the default reward ##
    reward = 1

    ## Reward if car goes close to optimal racing line ##
    DISTANCE_MULTIPLE = 1
    distance_reward = max(1e-3, 1 - (distance_from_center / (track_width * 0.5)))
    reward += distance_reward * DISTANCE_MULTIPLE

    ## Reward if speed is close to optimal speed ##
    SPEED_DIFF_NO_REWARD = 1
    SPEED_MULTIPLE = 2
    speed_diff = abs(optimal_speed - speed)
    if speed_diff <= SPEED_DIFF_NO_REWARD:
        # we use quadratic punishment (not linear) bc we're not as confident with the optimal speed
        # so, we do not punish small deviations from optimal speed
        speed_reward = (1 - (speed_diff / (SPEED_DIFF_NO_REWARD)) ** 2) ** 2
    else:
        speed_reward = 0
    reward += speed_reward * SPEED_MULTIPLE

    # Reward if less steps
    REWARD_PER_STEP_FOR_FASTEST_TIME = 1
    STANDARD_TIME = track_length / standard_speed
    FASTEST_TIME = track_length / fastest_speed
    try:
        steps_prediction = len(waypoints)
        steps_standard = track_length / standard_speed / 10
        reward_prediction = max(1e-3, (-REWARD_PER_STEP_FOR_FASTEST_TIME * (FASTEST_TIME) /
                                       (STANDARD_TIME - FASTEST_TIME)) * (
                                        steps_prediction - steps_standard))
        steps_reward = min(REWARD_PER_STEP_FOR_FASTEST_TIME, reward_prediction / steps_prediction)
    except:
        steps_reward = 0
    reward += steps_reward

    # Zero reward if obviously wrong direction (e.g. spin)
    next_point = waypoints[closest_waypoints[1]]
    prev_point = waypoints[closest_waypoints[0]]

    # Calculate the direction in radius, arctan2(dy, dx), the result is (-pi, pi) in radians
    track_direction = math.atan2(next_point[1] - prev_point[1], next_point[0] - prev_point[0])

    # Convert to degree
    track_direction = math.degrees(track_direction)

    direction_diff = abs(track_direction - heading + steering_angle)
    if direction_diff > 180:
        direction_diff = 360 - direction_diff
    if direction_diff > 30:
        reward = 1e-3

    # Zero reward of obviously too slow
    speed_diff_zero = optimal_speed - speed
    if speed_diff_zero > 0.5:
        reward = 1e-3

    ## Incentive for finishing the lap in less steps ##
    REWARD_FOR_FASTEST_TIME = laps * track_length / fastest_speed / 10  # should be adapted to track length and other rewards
    STANDARD_TIME = laps * track_length / standard_speed
    FASTEST_TIME = laps * track_length / fastest_speed
    if progress == 100:
        steps_standard = track_length / standard_speed / 10
        finish_reward = max(1e-3, (-REWARD_FOR_FASTEST_TIME /
                                   (10 * (STANDARD_TIME - FASTEST_TIME))) * (steps - steps_standard))
    else:
        finish_reward = 0
    reward += finish_reward

    ## Zero reward if off track ##
    if all_wheels_on_track == False:
        reward = 1e-3

    # Always return a float value
    return float(reward)


waypoints = [
    [3.05, 0.70],       # 1st point
    [3.20, 0.70],       # 2nd point
    [3.35, 0.70],       # 3rd point
    [3.50, 0.70],       # 4th point
    [3.65, 0.70],       # 5th point
    [3.80, 0.70],       # 6th point
    [3.95, 0.70],       # 7th point
    [4.10, 0.70],       # 8th point
    [4.25, 0.70],       # 9th point
    [4.40, 0.70],       # 10th point
    [4.55, 0.70],       # 11th point
    [4.70, 0.70],       # 12th point
    [4.85, 0.70],       # 13th point
    [5.00, 0.70],       # 14th point
    [5.15, 0.70],       # 15th point
    [5.30, 0.70],       # 16th point
    [5.45, 0.70],       # 17th point
    [5.60, 0.70],       # 18th point
    [5.75, 0.70],       # 19th point
    [5.90, 0.70],       # 20th point
    [6.05, 0.70],       # 21th point

    [0.00, 0.00],       # 22th point
    [0.00, 0.00],       # 23th point
    [0.00, 0.00],       # 24th point
    [0.00, 0.00],       # 25th point
    [0.00, 0.00],       # 26th point
    [0.00, 0.00],       # 27th point
    [0.00, 0.00],       # 28th point
    [0.00, 0.00],       # 29th point
    [0.00, 0.00],       # 30th point
    [0.00, 0.00],       # 31th point
    [0.00, 0.00],       # 32th point
    [0.00, 0.00],       # 33th point
    [0.00, 0.00],       # 34th point
    [0.00, 0.00],       # 35th point
    [0.00, 0.00],       # 36th point
    [0.00, 0.00],       # 37th point
    [0.00, 0.00],       # 38th point
    [0.00, 0.00],       # 39th point
    [0.00, 0.00],       # 40th point
    [0.00, 0.00],       # 41th point
    [0.00, 0.00],       # 42th point

    [6.20, 2.85],       # 43th point
    [6.05, 2.85],       # 44th point
    [5.90, 2.85],       # 45th point
    [5.75, 2.85],       # 46th point
    [5.60, 2.85],       # 47th point
    [5.45, 2.85],       # 48th point
    [5.30, 2.85],       # 49th point
    [5.15, 2.85],       # 50th point
    [5.00, 2.85],       # 51th point
    [4.85, 2.85],       # 52th point

    [0.00, 0.00],       # 53th point
    [0.00, 0.00],       # 54th point
    [0.00, 0.00],       # 55th point
    [0.00, 0.00],       # 56th point
    [0.00, 0.00],       # 57th point

    [0.00, 0.00],       # 58th point
    [0.00, 0.00],       # 59th point
    [0.00, 0.00],       # 60th point
    [0.00, 0.00],       # 61th point
    [0.00, 0.00],       # 62th point
    [0.00, 0.00],       # 63th point
    [0.00, 0.00],       # 64th point
    [0.00, 0.00],       # 65th point
    [0.00, 0.00],       # 66th point

    [0.00, 0.00],       # 67th point
    [0.00, 0.00],       # 68th point
    [0.00, 0.00],       # 69th point
    [0.00, 0.00],       # 70th point

    [3.05, 4.50],       # 71th point
    [2.90, 4.50],       # 72th point
    [2.75, 4.50],       # 73th point
    [2.60, 4.50],       # 74th point
    [2.45, 4.50],       # 75th point
    [2.30, 4.50],       # 76th point
    [2.15, 4.50],       # 77th point
    [2.00, 4.50],       # 78th point
    [1.85, 4.50],       # 79th point
    [1.70, 4.50],       # 80th point
    [1.55, 4.50],       # 81th point

    [0.00, 0.00],       # 82th point
    [0.00, 0.00],       # 83th point
    [0.00, 0.00],       # 84th point
    [0.00, 0.00],       # 85th point
    [0.00, 0.00],       # 86th point
    [0.00, 0.00],       # 87th point
    [0.00, 0.00],       # 88th point
    [0.00, 0.00],       # 89th point
    [0.00, 0.00],       # 90th point

    [0.00, 0.00],       # 91th point
    [0.00, 0.00],       # 92th point
    [0.00, 0.00],       # 93th point
    [0.00, 0.00],       # 94th point
    [0.00, 0.00],       # 95th point
    [0.00, 0.00],       # 96th point
    [0.00, 0.00],       # 97th point
    [0.00, 0.00],       # 98th point
    [0.00, 0.00],       # 99th point
    [0.00, 0.00],       # 100th point
    [0.00, 0.00],       # 101th point
    [0.00, 0.00],       # 102th point
    [0.00, 0.00],       # 103th point
    [0.00, 0.00],       # 104th point

    [0.00, 0.00],       # 105th point
    [0.00, 0.00],       # 106th point
    [0.00, 0.00],       # 107th point
    [0.00, 0.00],       # 108th point
    [0.00, 0.00],       # 109th point
    [0.00, 0.00],       # 110th point
    [0.00, 0.00],       # 111th point

    [2.05, 0.70],       # 112th point
    [2.20, 0.70],       # 113th point
    [2.35, 0.70],       # 114th point
    [2.50, 0.70],       # 115th point
    [2.65, 0.70],       # 116th point
    [2.80, 0.70],       # 117th point
    [2.95, 0.70]        # 118th point
]
def get_params(x, y, closest_waypoints, distance_from_center, speed, steering_angle, heading=0,
               all_wheels_on_track=True, track_width=0.17, is_offtrack=False, is_left_of_center=True):
    params = {}

    waypoints = [[4.0, 2.0], [5.0, 2.0], [6.0, 2.0], [7.0, 2.0], [8.0, 2.0], [9.0, 2.0], [10.0, 2.0], [10.5, 2.1]]

    params = {
        'x': x,
        'y': y,
        'closest_waypoints': closest_waypoints,
        'distance_from_center': distance_from_center,
        'speed': speed,
        'steering_angle': steering_angle,
        'heading': heading,
        'all_wheels_on_track': all_wheels_on_track,
        'is_offtrack': is_offtrack,
        'is_left_of_center': is_left_of_center,
        'track_width': track_width,
        'progress': 0.0,
        'steps': 1.0,
        'waypoints': waypoints
    }

    return params


def verify_reward_function():
    maximum_speed = 2.0
    speed_step = 0.5
    heading_degrees_step = 10

    x = 4.1
    y = 2.0
    closest_waypoints = [0, 1]
    distance_from_center = 0.01
    steering_angle = 0

    for i in range(1, int(maximum_speed / speed_step) + 1):
        speed = i * speed_step
        for j in range(int(360 / heading_degrees_step)):
            heading = j * heading_degrees_step
            if heading > 180:
                heading -= 360

            params = get_params(x, y, closest_waypoints, distance_from_center, speed, steering_angle, heading)
            reward = reward_function(params)
            print(
                'The reward of agent at ({0:.2f}, {1:.2f}) at speed {2:.2f} m/s with agent direction {3} degrees is {4}'.format(
                    params['x'], params['y'], params['speed'], params['heading'] + params['steering_angle'], reward))


if __name__ == '__main__':
    verify_reward_function()
