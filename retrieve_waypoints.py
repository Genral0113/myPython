import os
import csv
import math

import matplotlib.pyplot as plt


def read_csv_file(file_name, episode_num=-1):
    import csv

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


def get_waypoints():
    waypoints = []

    training_log = os.path.dirname(__file__) + r'\aws\training-simtrace\model-x'

    for file_name in os.listdir(training_log):
        if file_name.split('.')[1] == 'csv':
            waypoints = []
            file_name_full_path = os.path.join(training_log, file_name)

            log_parmas = read_csv_file(file_name_full_path)

            reward = log_parmas['reward']
            closest_waypoint = log_parmas['closest_waypoint']

            for i in range(max(closest_waypoint)):
                waypoints.append([0.0, 0.0])

            for i in range(len(closest_waypoint)):
                waypoints[closest_waypoint[i] - 1][0] = reward[i]

    training_log = os.path.dirname(__file__) + r'\aws\training-simtrace\model-y'

    for file_name in os.listdir(training_log):
        if file_name.split('.')[1] == 'csv':
            file_name_full_path = os.path.join(training_log, file_name)

            log_parmas = read_csv_file(file_name_full_path)

            reward = log_parmas['reward']
            closest_waypoint = log_parmas['closest_waypoint']

            for i in range(len(closest_waypoint)):
                waypoints[closest_waypoint[i] - 1][1] = reward[i]

    return waypoints


if __name__ == '__main__':

    waypoints = get_waypoints()
    print(waypoints)
    x = []
    y = []
    for i in range(len(waypoints)):
        x.append(waypoints[i][0])
        y.append(waypoints[i][1])

    plt.scatter(x, y, s=2)
    plt.title('waypoints')
    plt.show()
