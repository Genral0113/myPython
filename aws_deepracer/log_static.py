import pandas as pd
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from training_log_viewer import display_setup


def read_log(log_file, episode_num=-1, steps=-1):
    '''
    :param log_file: the Deepracer training log file in csv
    :param episode_num: the specified episode number for processing; it will process all episodes in a iteration if not specified
    :return: df - Data Frame of Deepracer training log file
    '''

    column_names = ['episode', 'steps', 'X', 'Y', 'yaw', 'steer', 'throttle', 'action1', 'action2', 'reward', 'done',
                    'all_wheels_on_track', 'progress', 'closest_waypoint', 'track_len', 'tstamp', 'episode_status',
                    'pause_duration']

    column_dtype = {'episode': int, 'steps': int, 'X': float, 'Y': float, 'yaw': float, 'steer': float,
                    'throttle': float, 'action1': str, 'action12': str, 'reward': float, 'done': bool,
                    'all_wheels_on_track': bool, 'progress': float, 'closest_waypoint': int, 'track_len': float,
                    'tstamp': float, 'episode_status': str, 'pause_duration': float}

    df = pd.read_csv(log_file, engine='python', skiprows=1, names=column_names, dtype=column_dtype)

    if episode_num >= 0:
        df = df[df.episode == episode_num]
        if steps > 0:
            df = df[df.steps == steps]

    return df


def distance_of_2points(p1, p2):
    dist = np.linalg.norm([p1[0] - p2[0], p1[1] - p2[1]])
    return dist


if __name__ == '__main__':
    log_file = r'C:\Users\asus\Desktop\2022 aws\autobus-final-v3-clone-training_job_EEgwPwzWTM-YyXYDegM5Hw_logs\51abc572-30e7-40e0-85db-d15ad31ccef6\sim-trace\training\training-simtrace\all-iterations.csv'
    log_file = r'C:\Users\asus\Desktop\2022 aws\autobus-v10-training_job_b3JGFL48Q-qUcDq3n1oXtA_logs\e48e476a-0ec0-44e5-894a-67c761dfd006\sim-trace\training\training-simtrace\all-iterations.csv'
    log_file = r'C:\Users\asus\Desktop\2022 aws\v12-training_job_moiGZao9TCeOeT1cArD0fw_logs\addab6f8-608d-472e-94e7-259aac85970c\sim-trace\training\training-simtrace\all-iterations.csv'
    df = read_log(log_file)

    episodes = []
    rewards = []
    episodes_distance = []
    episodes_time = []
    episodes_speed_avg = []

    i = 0
    prev_episode = -1
    first_step_timestamp = 0
    last_step_timestaap = 0
    distance = 0
    total_rewards = 0
    last_x = -99
    last_y = -99

    for episode, steps, x, y, yaw, steer, throttle, reward, progress, closest_waypoint, tstamp, episode_status \
            in zip(df['episode'], df['steps'], df['X'], df['Y'], df['yaw'], df['steer'], df['throttle'], df['reward'],
                   df['progress'], df['closest_waypoint'], df['tstamp'], df['episode_status']):

        if prev_episode == -1:
            prev_episode = episode
            last_x = x
            last_y = y

        if episode != prev_episode:
            episodes.append(prev_episode)
            rewards.append(total_rewards)
            episodes_distance.append(distance)
            episodes_time.append(last_step_timestaap - first_step_timestamp)
            episodes_speed_avg.append(10 * distance / (last_step_timestaap - first_step_timestamp))

            prev_episode = episode
            total_rewards = 0
            distance = 0
            last_x = x
            last_y = y

        if steps == 1:
            first_step_timestamp = tstamp

        distance += distance_of_2points([x, y], [last_x, last_y])
        last_x = x
        last_y = y
        total_rewards += reward
        last_step_timestaap = tstamp

        # next record
        i += 1

    episodes.append(prev_episode)
    rewards.append(total_rewards)
    episodes_distance.append(distance)
    episodes_time.append(last_step_timestaap - first_step_timestamp)
    episodes_speed_avg.append(10 * distance / (last_step_timestaap - first_step_timestamp))

    for i in range(len(episodes)):
        # if episodes_distance[i] > 16:
        print('{}the episode: Total rewards={}, distance={}, time={} and average speed={}'.format(episodes[i], rewards[i], episodes_distance[i], episodes_time[i], episodes_speed_avg[i]))

    fig = plt.figure(figsize=display_setup['figure_size'], dpi=display_setup['dpi'])
    mng = plt.get_current_fig_manager()
    ax = fig.add_subplot()

    legends = []

    ax.plot(episodes, rewards, c='b', linestyle='-.', linewidth=1)
    legends.append('training rewards')

    ax.plot(episodes, episodes_speed_avg, c='r', linestyle='-.', linewidth=1)
    legends.append('average speed(x10)')

    ax.plot(episodes, episodes_time, c='g', linestyle='-.', linewidth=1)
    legends.append('episode timestamp')

    ax.plot(episodes, episodes_distance, c='y', linestyle='-.', linewidth=1)
    legends.append('distance')

    mng.window.state("zoomed")
    plt.xlabel('episodes')
    plt.grid()
    plt.legend(legends)
    plt.show()
    plt.close()
