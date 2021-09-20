import math


def reward_function(params):
    degrees_of_pi = math.pi / 180
    first_level_degrees = 15
    second_level_degrees = 45
    cos_value_of_1l_degrees = math.cos(first_level_degrees * degrees_of_pi)
    cos_value_of_2l_degrees = math.cos(second_level_degrees * degrees_of_pi)

    def normal_function(x, mu=4, sigma=1.1):
        return 10 * math.exp(-1 * ((x - mu) ** 2) / (2 * sigma ** 2)) / (((2 * math.pi) ** 0.5) * sigma)

    # Read input variables
    all_wheels_on_track = params['all_wheels_on_track']
    distance_from_center = params['distance_from_center']
    track_width = params['track_width']
    agent_x = params['x']
    agent_y = params['y']
    heading = params['heading']
    steering_angle = params['steering_angle']
    waypoints = params['waypoints']
    closest_waypoints = params['closest_waypoints']
    speed = params['speed']
    is_offtrack = params['is_offtrack']
    is_left_of_center = params['is_left_of_center']

    # Initialize the reward with typical value
    reward = 1.0

    # Calculate 3 markers that are at varying distances away from the center line
    marker_1 = 0.1 * track_width
    marker_2 = 0.25 * track_width
    marker_3 = 0.5 * track_width

    # Give higher reward if the car is closer to center line and vice versa
    if distance_from_center <= marker_1:
        reward = 1.0
    elif distance_from_center <= marker_2:
        reward = 0.5
    elif distance_from_center <= marker_3:
        reward = 0.1
    else:
        reward = 1e-3  # likely crashed/ close to off track

    # Calculate the direction of the center line based on the closest waypoints
    next_point = waypoints[closest_waypoints[1]]
    prev_point = waypoints[closest_waypoints[0]]

    # Calculate the direction in radius, arctan2(dy, dx), the result is (-pi, pi) in radians
    track_direction = math.atan2(next_point[1] - prev_point[1], next_point[0] - prev_point[0])

    # Convert to degree
    track_direction = math.degrees(track_direction)

    # calculate agent direction
    agent_direction = heading + steering_angle

    direction_diff = track_direction - agent_direction
    if direction_diff > 180:
        direction_diff = 360 - direction_diff

    # normalize speed and
    cos_value_of_direction_diff = math.cos(direction_diff * degrees_of_pi)

    # reward update on direction and speed
    if cos_value_of_direction_diff >= cos_value_of_1l_degrees:
        reward += normal_function(speed, sigma=5)
    elif cos_value_of_direction_diff >= cos_value_of_2l_degrees:
        reward += normal_function(speed, sigma=0.5)
    else:
        reward = 1e-3

    return reward


def get_params(x, y, closest_waypoints, distance_from_center, speed, steering_angle, heading=0,
               all_wheels_on_track=True, track_width=2, is_offtrack=False, is_left_of_center=True):
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
        'waypoints': waypoints
    }

    return params


def generate_waypoints(x0=3.05, y0=0.70):
    track_length = 17.71
    steps = 118
    step_length = track_length / steps

    waypoints = []

    left_to_right_straight_line_steps = 23
    right_semi_cycle_steps = 43

    for i in range(steps):
        x = x0
        y = y0
        if i < left_to_right_straight_line_steps:
            x += i * step_length
        elif i < right_semi_cycle_steps:
            pass
        waypoints.append([x, y])

    return waypoints


def normal_function(x, mu=10, sigma=1.1):
    return 5 * math.exp(-1 * ((x - mu) ** 2) / (2 * sigma ** 2)) / (((2 * math.pi) ** 0.5) * sigma)


def plot_speed_degrees_diff():
    import matplotlib.pyplot as plt
    import numpy as np

    # constants
    direction_limit = 10.0

    mu = 4  # maximum speed
    sigma = 1.1  # direction difference limit
    x = np.arange(0, mu, 0.01)

    y = []
    for x1 in x:
        y.append(normal_function(x1, mu, sigma))
    plt.plot(x, y)
    plt.grid(True)
    plt.xlabel('speed')
    plt.ylabel('Reward')
    plt.legend(['sigma=' + str(sigma)])
    plt.show()


def verify_reward_function():
    maximum_speed = 3.0
    speed_step = 0.5
    heading_degrees_step = 10

    x = 4.1
    y = 2.0
    closest_waypoints = [0, 1]
    distance_from_center = 0.0
    steering_angle = 0

    for i in range(1, int(maximum_speed / speed_step) + 1):
        speed = i * speed_step
        for j in range(int(360 / heading_degrees_step)):
            heading = j * heading_degrees_step
            if heading > 180:
                heading -= 360

            params = get_params(x, y, closest_waypoints, distance_from_center, speed, steering_angle, heading,
                                all_wheels_on_track=False, is_left_of_center=False)
            reward = reward_function(params)
            print(
                'The reward of agent at ({0:.2f}, {1:.2f}) at speed {2:.2f} m/s with agent direction {3} degrees is {4}'.format(
                    params['x'], params['y'], params['speed'], params['heading'] + params['steering_angle'], reward))

    # params = get_params(x, y, closest_waypoints, distance_from_center, 0.50, steering_angle, -10)
    # reward = reward_function(params)
    # print('The reward of agent at ({0:.2f}, {1:.2f}) at speed {2:.2f} m/s with heading={3} degrees and steering angle={4} degrees is {5}'.format(params['x'], params['y'], params['speed'], params['heading'], params['steering_angle'], reward))


import os
import csv


def read_csv_file(file_name, episode_num=1):
    X = []
    Y = []
    with open(file_name) as input_file:
        csv_reader = csv.reader(input_file)
        header_row = next(csv_reader)
        # print(header_row)
        for row_data in csv_reader:
            # print(row_data)
            episode = int(row_data[0])
            steps = float(row_data[1])
            x = float(row_data[2])
            y = float(row_data[3])
            yam = float(row_data[4])
            steer = float(row_data[5])
            throttle = float(row_data[6])
            action = [steer, throttle]
            reward = float(row_data[9])
            done = row_data[10]
            if done == 'True':
                done = True
            else:
                done = False
            all_wheels_on_track = row_data[11]
            if all_wheels_on_track == 'True':
                all_wheels_on_track = True
            else:
                all_wheels_on_track = False
            progress = float(row_data[12])
            closest_waypoint = int(row_data[13])
            track_len = float(row_data[14])
            tstamp = float(row_data[15])
            episode_status = row_data[16]
            pause_duration = float(row_data[17])
            if episode == episode_num:
                # print(
                #     'steps= {} x= {} y= {} yam= {} reward= {} done= {} all_wheels_on_track= {} closest_waypoint = {} tstamp= {} episode_status = {} pause_duration= {}'.format(
                #         steps, x, y, yam, reward, done, all_wheels_on_track, closest_waypoint, tstamp, episode_status,
                #         pause_duration))
                X.append(x)
                Y.append(y)
    return X, Y


def plot_track_line():
    # evaluation_file = os.path.dirname(__file__) + r'\aws\evaluation-simtrace\0-iteration-model1.csv'
    evaluation_file = os.path.dirname(__file__) + r'\aws\evaluation-simtrace\ben-model3-envaluation.csv'
    waypoints = [
        [3.05, 0.70],  # 1st point
        [3.20, 0.70],  # 2nd point
        [3.35, 0.70],  # 3rd point
        [3.50, 0.70],  # 4th point
        [3.65, 0.70],  # 5th point
        [3.80, 0.70],  # 6th point
        [3.95, 0.70],  # 7th point
        [4.10, 0.70],  # 8th point
        [4.25, 0.70],  # 9th point
        [4.40, 0.70],  # 10th point
        [4.55, 0.70],  # 11th point
        [4.70, 0.70],  # 12th point
        [4.85, 0.70],  # 13th point
        [5.00, 0.70],  # 14th point
        [5.15, 0.70],  # 15th point
        [5.30, 0.70],  # 16th point
        [5.45, 0.70],  # 17th point
        [5.60, 0.70],  # 18th point
        [5.75, 0.70],  # 19th point
        [5.90, 0.70],  # 20th point
        [6.05, 0.70],  # 21th point
        [6.20, 0.70],  # 22th point

        [6.35, 0.76],  # 23th point
        [6.43, 0.80],  # 24th point
        [6.51, 0.87],  # 25th point
        [6.59, 0.92],  # 26th point
        [6.67, 1.02],  # 27th point
        [6.75, 1.13],  # 28th point
        [6.83, 1.33],  # 29th point
        [6.91, 1.45],  # 30th point
        [6.99, 1.60],  # 31th point
        [7.07, 1.69],  # 32th point
        [7.15, 1.80],  # 33th point *
        [0.00, 0.00],  # 34th point
        [0.00, 0.00],  # 35th point
        [0.00, 0.00],  # 36th point
        [0.00, 0.00],  # 37th point
        [0.00, 0.00],  # 38th point
        [0.00, 0.00],  # 39th point
        [0.00, 0.00],  # 40th point
        [0.00, 0.00],  # 41th point
        [0.00, 0.00],  # 42th point
        [6.20, 2.85],  # 43th point

        [6.05, 2.85],  # 44th point
        [5.90, 2.85],  # 45th point
        [5.75, 2.85],  # 46th point
        [5.60, 2.85],  # 47th point
        [5.45, 2.85],  # 48th point
        [5.30, 2.85],  # 49th point
        [5.15, 2.85],  # 50th point
        [5.00, 2.85],  # 51th point
        [4.85, 2.85],  # 52th point

        [0.00, 0.00],  # 53th point
        [0.00, 0.00],  # 54th point
        [0.00, 0.00],  # 55th point
        [0.00, 0.00],  # 56th point
        [0.00, 0.00],  # 57th point

        [0.00, 0.00],  # 58th point
        [0.00, 0.00],  # 59th point
        [0.00, 0.00],  # 60th point
        [0.00, 0.00],  # 61th point
        [0.00, 0.00],  # 62th point
        [0.00, 0.00],  # 63th point
        [0.00, 0.00],  # 64th point
        [0.00, 0.00],  # 65th point
        [0.00, 0.00],  # 66th point

        [0.00, 0.00],  # 67th point
        [0.00, 0.00],  # 68th point
        [0.00, 0.00],  # 69th point
        [0.00, 0.00],  # 70th point

        [3.05, 4.50],  # 71th point
        [2.90, 4.50],  # 72th point
        [2.75, 4.50],  # 73th point
        [2.60, 4.50],  # 74th point
        [2.45, 4.50],  # 75th point
        [2.30, 4.50],  # 76th point
        [2.15, 4.50],  # 77th point
        [2.00, 4.50],  # 78th point
        [1.85, 4.50],  # 79th point
        [1.70, 4.50],  # 80th point
        [1.55, 4.50],  # 81th point

        [0.00, 0.00],  # 82th point
        [0.00, 0.00],  # 83th point
        [0.00, 0.00],  # 84th point
        [0.00, 0.00],  # 85th point
        [0.00, 0.00],  # 86th point
        [0.00, 0.00],  # 87th point
        [0.00, 0.00],  # 88th point
        [0.00, 0.00],  # 89th point
        [0.00, 0.00],  # 90th point

        [0.00, 0.00],  # 91th point
        [0.00, 0.00],  # 92th point
        [0.00, 0.00],  # 93th point
        [0.00, 0.00],  # 94th point
        [0.00, 0.00],  # 95th point
        [0.00, 0.00],  # 96th point
        [0.00, 0.00],  # 97th point
        [0.00, 0.00],  # 98th point
        [0.00, 0.00],  # 99th point
        [0.00, 0.00],  # 100th point
        [0.00, 0.00],  # 101th point
        [0.00, 0.00],  # 102th point
        [0.00, 0.00],  # 103th point
        [0.00, 0.00],  # 104th point

        [0.00, 0.00],  # 105th point
        [0.00, 0.00],  # 106th point
        [0.00, 0.00],  # 107th point
        [0.00, 0.00],  # 108th point
        [0.00, 0.00],  # 109th point
        [0.00, 0.00],  # 110th point
        [0.00, 0.00],  # 111th point

        [2.05, 0.70],  # 112th point
        [2.20, 0.70],  # 113th point
        [2.35, 0.70],  # 114th point
        [2.50, 0.70],  # 115th point
        [2.65, 0.70],  # 116th point
        [2.80, 0.70],  # 117th point
        [2.95, 0.70]  # 118th point
    ]
    import matplotlib.pyplot as plt

    x = []
    y = []
    for i in range(len(waypoints)):
        x.append(waypoints[i][0])
        y.append((waypoints[i][1]))
    plt.plot(x, y)

    x, y = read_csv_file(evaluation_file, episode_num=0)
    # plt.plot(x, y)

    x, y = read_csv_file(evaluation_file, episode_num=1)
    # plt.plot(x, y)

    x, y = read_csv_file(evaluation_file, episode_num=2)
    # plt.plot(x, y)

    plt.grid(True)

    plt.legend(['waypoints', 'lap1', 'lap2', 'lap3'])
    plt.show()


if __name__ == '__main__':
    # training_file_dir = os.path.dirname(__file__) + r'\aws\training-simtrace'
    # evaluation_file_dir = os.path.dirname(__file__) + r'\aws\evaluation-simtrace'
    #
    # file_dir = evaluation_file_dir
    #
    # for file_name in os.listdir(file_dir):
    #     File_name_full_path = os.path.join(file_dir, file_name)
    #     read_csv_file(File_name_full_path)
    # plot_speed_degrees_diff()

    plot_track_line()
