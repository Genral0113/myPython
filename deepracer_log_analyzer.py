import os
import csv
import matplotlib.pyplot as plt
import numpy as np


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


def plot_to_image(file_directory, plt_show=False):
    for file_name in os.listdir(file_directory):
        if file_name.split('.')[1] == 'csv':
            file_name_full_path = os.path.join(file_directory, file_name)
            image_file_name_full_path = os.path.join(file_directory, file_name.split('.')[0] + '.jpg')

            # load data from log file
            log_parmas = read_csv_file(file_name_full_path)

            # retrieve useful data
            episode = np.array(log_parmas['episode'])
            x = np.array(log_parmas['x'])
            y = np.array(log_parmas['y'])

            legent = []
            plt.figure()
            for episode_num in range(max(episode)+1):
                pos = np.where(episode == episode_num)
                plt.scatter(x[pos], y[pos], s=2)
                legent.append('episode='+str(episode_num))
            plt.grid(True)
            if len(legent) < 6:
                plt.legend(legent, loc='upper right')
            plt.title(file_name.split('.')[0])
            plt.savefig(image_file_name_full_path)
            if plt_show:
                plt.show()


if __name__ == '__main__':
    # log file list
    log_file1_dir = os.path.dirname(__file__) + r'\aws\evaluation-simtrace'
    log_file1 = os.path.dirname(__file__) + r'\aws\evaluation-simtrace\ben-model3.csv'
    log_file2 = os.path.dirname(__file__) + r'\aws\evaluation-simtrace\dlcf-htc-2021-model1.csv'
    log_file3 = os.path.dirname(__file__) + r'\aws\evaluation-simtrace\dlcf-htc-2021-model6.csv'
    log_file4 = os.path.dirname(__file__) + r'\aws\training-simtrace\ben-model4\0-iteration.csv'
    log_file4_dir = os.path.dirname(__file__) + r'\aws\training-simtrace\dlcf-htc-2021-model6'
    log_file5_dir = os.path.dirname(__file__) + r'\aws\training-simtrace\dlcf-htc-2021-model1'

    plot_to_image(log_file1_dir, plt_show=True)
