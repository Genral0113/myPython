import math
import numpy as np
import pandas
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as colors


display_setup = {
    'dot_size': 2,
    'fontsize': 4,
    'figure_size': (40, 30),
    'dpi': 240,
    'display_waypoints': True,
    'display_steps': True,
    'display_heading_arrow': True,
    'display_projected_track': False,
    'heading_arrow_width': 0.0005
}


def get_waypoints(npy_file):
    waypoints = np.load(npy_file)
    waypoints_mid = waypoints[:, 0:2]
    waypoints_inn = waypoints[:, 2:4]
    waypoints_out = waypoints[:, 4:6]
    return waypoints_mid, waypoints_inn, waypoints_out


def read_log(log_file, episode_num=-1):
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

    return df


def get_color_name(ind):
    i = 0
    for c in colors.cnames:
        if i == ind % len(colors.cnames):
            return c
        i += 1


def plot_waypoints(ax, waypoints_mid, waypoints_inn, waypoints_out, waypoints_length=154):
    # plot outer track
    i = 0
    for x, y in zip(waypoints_out[:, 0], waypoints_out[:, 1]):
        i = i % waypoints_length
        ax.scatter(x, y, c='r', s=display_setup['dot_size'])
        ax.text(x, y, str(i + 1), fontsize=display_setup['fontsize'])
        i += 1

    # plot middle track
    i = 0
    for x, y in zip(waypoints_mid[:, 0], waypoints_mid[:, 1]):
        i = i % waypoints_length
        ax.scatter(x, y, c='b', s=display_setup['dot_size'])
        ax.text(x, y, str(i + 1), fontsize=display_setup['fontsize'])
        i += 1

    # plot inner track
    i = 0
    for x, y in zip(waypoints_inn[:, 0], waypoints_inn[:, 1]):
        i = i % waypoints_length
        ax.scatter(x, y, c='r', s=display_setup['dot_size'])
        ax.text(x, y, str(i + 1), fontsize=display_setup['fontsize'])
        i += 1


def plot_dataframe(df, ax):
    df.plot(x='X', y='Y', kind='scatter', ax=ax, grid=True, color=[get_color_name(x) for x in df['episode']],
            s=display_setup['dot_size'])

    if display_setup['display_steps']:
        for x, y, t in zip(df['X'], df['Y'], df['steps']):
            ax.text(x, y, str(t), fontsize=display_setup['fontsize'])

    if display_setup['display_heading_arrow']:
        for x, y, omega, episode in zip(df['X'], df['Y'], df['yaw'], df['episode']):
            x1 = math.cos(math.radians(omega))
            y1 = math.sin(math.radians(omega))
            ax.quiver(x, y, x1, y1, color=get_color_name(episode), width=display_setup['heading_arrow_width'])

    if display_setup['display_projected_track']:
        tstamp = df['tstamp'].array
        for i in range(len(tstamp)):
            if i == 0 or i == len(tstamp) - 1:
                tstamp[i] = 0
            else:
                tstamp[i] = tstamp[i+1] - tstamp[i]

        for episode, x, y, omega, steer, steps, speed, t, episode_status in zip(df['episode'], df['X'], df['Y'], df['yaw'], df['steer'], df['steps'], df['throttle'], tstamp, df['episode_status']):
            if steps > 1 and episode_status != 'off_track':
                x1 = x + 0.5 * t ** 2 * speed * math.cos(math.radians(omega))
                y1 = y + 0.5 * t ** 2 * speed * math.sin(math.radians(omega))
                ax.scatter(x1, y1, s=display_setup['dot_size'], c='r')
                ax.text(x1, y1, str(steps + 1), fontsize=display_setup['fontsize'])


if __name__ == '__main__':
    waypoints_npy_file = r'npy\ChampionshipCup2019_track.npy'
    training_log_dir = r'aws\training-simtrace\2019\dlcf-test-clone'

    fig = plt.figure(figsize=display_setup['figure_size'], dpi=display_setup['dpi'])
    mng = plt.get_current_fig_manager()
    ax = fig.add_subplot()

    waypoints_mid, waypoints_inn, waypoints_out = get_waypoints(waypoints_npy_file)
    if display_setup['display_waypoints']:
        plot_waypoints(ax, waypoints_mid, waypoints_inn, waypoints_out)

    training_log = training_log_dir + r'\21-iteration.csv'
    df = read_log(training_log)

    plot_dataframe(df, ax)

    plt.grid(True)
    mng.window.state("zoomed")
    plt.show()
    plt.close()
