import os
import csv
import matplotlib.pyplot as plt
import numpy as np
import math


def read_csv_file(file_name, episode_num=-1):
    episode = []
    steps = []
    x = []
    y = []
    yam = []
    steer = []
    throttle = []
    action = []
    reward = []
    done = []
    all_wheels_on_track = []
    progress = []
    closest_waypoint = []
    track_len = []
    tstamp = []
    episode_status = []
    pause_duration = []

    with open(file_name) as input_file:
        csv_reader = csv.reader(input_file)
        header_row = next(csv_reader)   # skip header row
        for row_data in csv_reader:
            if episode_num == -1 or int(row_data[0]) == episode_num:
                episode.append(int(row_data[0]))
                steps.append(float(row_data[1]))
                x.append(float(row_data[2]))
                y.append(float(row_data[3]))
                yam.append(float(row_data[4]))
                steer.append(float(row_data[5]))
                throttle.append(float(row_data[6]))
                action.append([steer, throttle])
                reward.append(float(row_data[9]))
                tmp = row_data[10]
                if tmp == 'True':
                    done.append(True)
                else:
                    done.append(False)
                tmp = row_data[11]
                if tmp == 'True':
                    all_wheels_on_track.append(True)
                else:
                    all_wheels_on_track.append(False)
                progress.append(float(row_data[12]))
                closest_waypoint.append(int(row_data[13]))
                track_len.append(float(row_data[14]))
                tstamp.append(float(row_data[15]))
                episode_status.append(row_data[16])
                pause_duration.append(float(row_data[17]))

    log_parmas = {
        'episode': episode,
        'steps': steps,
        'x': x,
        'y': y,
        'yam': yam,
        'steer': steer,
        'throttle': throttle,
        'action': action,
        'reward': reward,
        'done': done,
        'all_wheels_on_track': all_wheels_on_track,
        'progress': progress,
        'closest_waypoint': closest_waypoint,
        'track_len': track_len,
        'tstamp': tstamp,
        'episode_status': episode_status,
        'pause_duration': pause_duration
    }

    return log_parmas


def multiple_track_line(track_line, multiply_factor=2):
    new_track_line = []

    for i in range(len(track_line)):
        # get next point
        j = divmod(i+1, len(track_line))[1]     # get mod of (i + i) mod len

        prev_point = track_line[i]
        next_point = track_line[j]

        delta_x = (next_point[0] - prev_point[0]) / multiply_factor
        delta_y = (next_point[1] - prev_point[1]) / multiply_factor

        for j in range(multiply_factor):
            temp_point = [prev_point[0] + j * delta_x, prev_point[1] + j * delta_y]
            new_track_line.append(temp_point)

    if len(new_track_line) != multiply_factor * len(track_line):
        print('Error in multiplying track line dots .............. ')

    return new_track_line


def get_track_direction(track_line):
    track_direction = []

    for i in range(len(track_line)):
        # get next point
        j = divmod(i + 1, len(track_line))[1]    # get mod of (i + i) mod len

        prev_point = track_line[i]
        next_point = track_line[j]

        # calculate the track angel between [prev_point, middle_point]
        track_angel = math.atan2(next_point[1] - prev_point[1], next_point[0] - prev_point[0])
        track_angel = math.degrees(track_angel)
        track_direction.append(track_angel)

    if len(track_direction) != len(track_line):
        print('Error in calculating track line direction .............. ')

    return track_direction


def calculate_distance_of_track_line(track_line):
    track_distance = []

    for i in range(len(track_line)):
        # get next point
        j = divmod(i + 1, len(track_line))[1]  # get mod of (i + i) mod len

        prev_point = track_line[i]
        next_point = track_line[j]

        distance = dist_2_points(prev_point[0], next_point[0], prev_point[1], next_point[1])

        track_distance.append(distance)

    if len(track_distance) != len(track_line):
        print('Error in calculating track line distance .............. ')

    return track_distance


def dist_2_points(x1, x2, y1, y2):
    return abs(abs(x1 - x2) ** 2 + abs(y1 - y2) ** 2) ** 0.5


def closest_2_racing_points_index(racing_coords, car_coords):
    # Calculate all distances to racing points
    distances = []
    for i in range(len(racing_coords)):
        distance = dist_2_points(x1=racing_coords[i][0], x2=car_coords[0],
                                 y1=racing_coords[i][1], y2=car_coords[1])
        distances.append(distance)

    # Get index of the closest racing point
    closest_index = distances.index(min(distances))

    # Get index of the second closest racing point
    distances_no_closest = distances.copy()
    distances_no_closest[closest_index] = 999
    second_closest_index = distances_no_closest.index(
        min(distances_no_closest))

    return [closest_index, second_closest_index]


def dist_to_racing_line(closest_coords, second_closest_coords, car_coords):
    # Calculate the distances between 2 closest racing points
    a = abs(dist_2_points(x1=closest_coords[0],
                          x2=second_closest_coords[0],
                          y1=closest_coords[1],
                          y2=second_closest_coords[1]))

    # Distances between car and closest and second closest racing point
    b = abs(dist_2_points(x1=car_coords[0],
                          x2=closest_coords[0],
                          y1=car_coords[1],
                          y2=closest_coords[1]))
    c = abs(dist_2_points(x1=car_coords[0],
                          x2=second_closest_coords[0],
                          y1=car_coords[1],
                          y2=second_closest_coords[1]))

    # Calculate distance between car and racing line (goes through 2 closest racing points)
    # try-except in case a=0 (rare bug in DeepRacer)
    try:
        distance = abs(-(a ** 4) + 2 * (a ** 2) * (b ** 2) + 2 * (a ** 2) * (c ** 2) -
                       (b ** 4) + 2 * (b ** 2) * (c ** 2) - (c ** 4)) ** 0.5 / (2 * a)
    except:
        distance = b

    return distance


# Calculate which one of the closest racing points is the next one and which one the previous one
def next_prev_racing_point(closest_coords, second_closest_coords, car_coords, heading):
    # Virtually set the car more into the heading direction
    heading_vector = [math.cos(math.radians(
        heading)), math.sin(math.radians(heading))]
    new_car_coords = [car_coords[0] + heading_vector[0],
                      car_coords[1] + heading_vector[1]]

    # Calculate distance from new car coords to 2 closest racing points
    distance_closest_coords_new = dist_2_points(x1=new_car_coords[0],
                                                x2=closest_coords[0],
                                                y1=new_car_coords[1],
                                                y2=closest_coords[1])
    distance_second_closest_coords_new = dist_2_points(x1=new_car_coords[0],
                                                       x2=second_closest_coords[0],
                                                       y1=new_car_coords[1],
                                                       y2=second_closest_coords[1])

    if distance_closest_coords_new <= distance_second_closest_coords_new:
        next_point_coords = closest_coords
        prev_point_coords = second_closest_coords
    else:
        next_point_coords = second_closest_coords
        prev_point_coords = closest_coords

    return [next_point_coords, prev_point_coords]


def racing_direction_diff(closest_coords, second_closest_coords, car_coords, heading):
    # Calculate the direction of the center line based on the closest waypoints
    next_point, prev_point = next_prev_racing_point(closest_coords,
                                                    second_closest_coords,
                                                    car_coords,
                                                    heading)

    # Calculate the direction in radius, arctan2(dy, dx), the result is (-pi, pi) in radians
    track_direction = math.atan2(
        next_point[1] - prev_point[1], next_point[0] - prev_point[0])

    # Convert to degree
    track_direction = math.degrees(track_direction)

    # Calculate the difference between the track direction and the heading direction of the car
    direction_diff = abs(track_direction - heading)
    if direction_diff > 180:
        direction_diff = 360 - direction_diff

    return direction_diff


def plot_to_image(file_directory, plt_show=0):
    for file_name in os.listdir(file_directory):
        file_name_full_path = os.path.join(file_directory, file_name)

        if os.path.isfile(file_name_full_path) and file_name.split('.')[1] == 'csv':
            image_file_name_full_path = os.path.join(file_directory, file_name.split('.')[0] + '.jpg')

            # load data from log file
            log_parmas = read_csv_file(file_name_full_path)

            # retrieve useful data
            episode = np.array(log_parmas['episode'])
            x = np.array(log_parmas['x'])
            y = np.array(log_parmas['y'])
            steps = np.array(log_parmas['steps'])
            speed = np.array(log_parmas['throttle'])
            heading = np.array(log_parmas['yam'])
            steer_angel = np.array(log_parmas['steer'])
            reward = np.array(log_parmas['reward'])

            track_line = []

            legent = []

            plt.figure()

            for episode_num in range(min(episode), max(episode) + 1):
                pos = np.where(episode == episode_num)
                track_line.append([x[pos[0][0]], y[pos[0][0]]])
                if plt_show > 0:
                    plt.scatter(x[pos], y[pos], s=2)
                    legent.append('episode='+str(episode_num))
            plt.grid(True)
            if len(legent) < 6 and plt_show > 0:
                plt.legend(legent, loc='upper right')
            plt.title(file_name.split('.')[0])
            if plt_show == 2 or plt_show == 3:
                plt.savefig(image_file_name_full_path)
            if plt_show == 1 or plt_show == 3:
                plt.show()
            plt.close()

            if plt_show == 0:
                x1 = []
                y1 = []
                optimal_time = []
                optimal_speed = 2.5   # 3m/s
                new_track_line = multiple_track_line(track_line)
                track_direction = get_track_direction(new_track_line)
                track_distance = calculate_distance_of_track_line(new_track_line)
                new_closest_waypoints = closest_2_racing_points_index(new_track_line, [7.22, 1.6])
                print(new_closest_waypoints)
                for i in range(len(track_distance)):
                    optimal_time.append(track_distance[i]/optimal_speed)
                    x1.append(new_track_line[i][0])
                    y1.append(new_track_line[i][1])
                    plt.scatter(new_track_line[i][0], new_track_line[i][1], s=2)

                if len(new_track_line) != len(optimal_time):
                    print('Error in calculating optimal time .............. ')

                plt.title('track line')
                plt.show()
                print(x1)
                print(y1)
                print(track_direction)
                print(optimal_time)


if __name__ == '__main__':
    # log file list
    log_file1_dir = os.path.dirname(__file__) + r'\aws\evaluation-simtrace'
    log_file1 = os.path.dirname(__file__) + r'\aws\evaluation-simtrace\ben-model3.csv'
    log_file2 = os.path.dirname(__file__) + r'\aws\evaluation-simtrace\dlcf-htc-2021-model1.csv'
    log_file3 = os.path.dirname(__file__) + r'\aws\evaluation-simtrace\dlcf-htc-2021-model6.csv'
    log_file4 = os.path.dirname(__file__) + r'\aws\training-simtrace\ben-model4\0-iteration.csv'
    log_file4_dir = os.path.dirname(__file__) + r'\aws\training-simtrace\dlcf-htc-2021-model6'
    log_file5_dir = os.path.dirname(__file__) + r'\aws\training-simtrace\dlcf-htc-2021-model1'
    log_file6_dir = os.path.dirname(__file__) + r'\aws\training-simtrace\model9'

    # plt_show = 0 for calculation
    # plt_show = 1 for show
    # plt_show = 2 for save image
    # plt_show = 3 for both show and save image

    plot_to_image(log_file6_dir, plt_show=2)
