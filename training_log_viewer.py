import math
import numpy as np
import pandas
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from functions_2d import *

display_setup = {
    'dot_size': 2,
    'fontsize': 4,
    'figure_size': (40, 30),
    'dpi': 240,
    'display_waypoints_mid': True,
    'display_waypoints_inn': True,
    'display_waypoints_out': True,
    'display_steps': True,
    'display_heading_arrow': True,
    'display_action': True,
    'display_training_reward': True,
    'display_projected_track': False,
    'display_distance_to_closest_waypoint': True,
    'heading_arrow_width': 0.0005,
    'dist_line_width': 0.2
}


def get_waypoints(npy_file):
    waypoints = np.load(npy_file)
    waypoints_len = len(waypoints) - 1
    waypoints_mid = waypoints[0:waypoints_len][:, 0:2]
    waypoints_inn = waypoints[0:waypoints_len][:, 2:4]
    waypoints_out = waypoints[0:waypoints_len][:, 4:6]
    return waypoints_mid, waypoints_inn, waypoints_out


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


def get_color_name(ind):
    i = 0
    for c in colors.cnames:
        if i == ind % len(colors.cnames):
            return c
        i += 1


def plot_waypoints(ax, waypoints_mid, waypoints_inn, waypoints_out):
    waypoints_length = len(waypoints_mid)

    # plot outer track
    if display_setup['display_waypoints_out']:
        i = 0
        for x, y in zip(waypoints_out[:, 0], waypoints_out[:, 1]):
            i = i % waypoints_length
            ax.scatter(x, y, c='r', s=display_setup['dot_size'])
            ax.text(x, y, str(i), fontsize=display_setup['fontsize'])
            i += 1

    # plot middle track
    if display_setup['display_waypoints_mid']:
        i = 0
        for x, y in zip(waypoints_mid[:, 0], waypoints_mid[:, 1]):
            i = i % waypoints_length
            ax.scatter(x, y, c='b', s=display_setup['dot_size'])
            ax.text(x, y, str(i), fontsize=display_setup['fontsize'])
            i += 1

    # plot inner track
    if display_setup['display_waypoints_inn']:
        i = 0
        for x, y in zip(waypoints_inn[:, 0], waypoints_inn[:, 1]):
            i = i % waypoints_length
            ax.scatter(x, y, c='r', s=display_setup['dot_size'])
            ax.text(x, y, str(i), fontsize=display_setup['fontsize'])
            i += 1


def plot_dataframe(df, ax):
    df.plot(x='X', y='Y', kind='scatter', ax=ax, grid=True, color=[get_color_name(x) for x in df['episode']],
            s=display_setup['dot_size'])

    if display_setup['display_steps']:
        for x, y, steeps in zip(df['X'], df['Y'], df['steps']):
            ax.text(x, y - 0.005, str(steeps), fontsize=display_setup['fontsize'])

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
                tstamp[i] = tstamp[i + 1] - tstamp[i]

        for episode, x, y, omega, steer, steps, speed, t, episode_status in zip(df['episode'], df['X'], df['Y'],
                                                                                df['yaw'], df['steer'], df['steps'],
                                                                                df['throttle'], tstamp,
                                                                                df['episode_status']):
            if steps > 1 and episode_status != 'off_track':
                x1 = x + 0.5 * t ** 2 * speed * math.cos(math.radians(omega))
                y1 = y + 0.5 * t ** 2 * speed * math.sin(math.radians(omega))
                ax.scatter(x1, y1, s=display_setup['dot_size'], c='r')
                ax.text(x1, y1, str(steps + 1), fontsize=display_setup['fontsize'])

    if display_setup['display_action']:
        for episode, x, y, steer, throttle in zip(df['episode'], df['X'], df['Y'], df['steer'], df['throttle']):
            action = '[{:.1f},{:.2f}]'.format(steer, throttle)
            ax.text(x, y, action, fontsize=display_setup['fontsize'], color=get_color_name(episode))

    if display_setup['display_training_reward']:
        for episode, x, y, reward in zip(df['episode'], df['X'], df['Y'], df['reward']):
            ax.text(x, y + 0.005, '{:.3f}'.format(reward), fontsize=display_setup['fontsize'],
                    color=get_color_name(episode))


def plot_dataframe_new(df, ax, waypoints_mid, waypoints_inn, waypoints_out):
    for episode, steps, x, y, yaw, steer, throttle, reward, progress, closest_waypoint, tstamp, episode_status \
            in zip(df['episode'], df['steps'], df['X'], df['Y'], df['yaw'], df['steer'], df['throttle'], df['reward'],
                   df['progress'], df['closest_waypoint'], df['tstamp'], df['episode_status']):

        ax.scatter(x, y, c=get_color_name(episode), s=display_setup['dot_size'])

        if display_setup['display_steps']:
            ax.text(x, y - 0.005, str(steps), fontsize=display_setup['fontsize'])

        if display_setup['display_heading_arrow']:
            x1 = math.cos(math.radians(yaw))
            y1 = math.sin(math.radians(yaw))
            ax.quiver(x, y, x1, y1, color=get_color_name(episode), width=display_setup['heading_arrow_width'])

        if display_setup['display_projected_track']:
            if steps > 1 and episode_status != 'off_track':
                x1 = x + 0.5 * 1/15 ** 2 * throttle * math.cos(math.radians(yaw))
                y1 = y + 0.5 * 1/15 ** 2 * throttle * math.sin(math.radians(yaw))
                ax.scatter(x1, y1, s=display_setup['dot_size'], c='r')
                ax.text(x1, y1, str(steps + 1), fontsize=display_setup['fontsize'])

        if display_setup['display_action']:
            action = '[{:.1f},{:.2f}]'.format(steer, throttle)
            ax.text(x, y, action, fontsize=display_setup['fontsize'], color=get_color_name(episode))

        if display_setup['display_training_reward']:
            ax.text(x, y + 0.005, '{:.3f}'.format(reward), fontsize=display_setup['fontsize'],
                    color=get_color_name(episode))

        if display_setup['display_distance_to_closest_waypoint']:
            waypoints_length = len(waypoints_mid) - 1
            x1 = waypoints_mid[closest_waypoint % waypoints_length][0]
            y1 = waypoints_mid[closest_waypoint % waypoints_length][1]

            # closest waypoint
            dist = distance_of_2points([x, y], [x1, y1])
            ax.plot([x, x1], [y, y1], color=get_color_name(episode), linewidth=display_setup['dist_line_width'], linestyle='-.')
            a, b, c = line_2p([x, y], [x1, y1])
            x_m = 0.5 * (x + x1)
            y_m = -1 * (a * x_m + c)/b
            ax.text(x_m, y_m, '{:.2f}'.format(dist), fontsize=display_setup['fontsize'], color=get_color_name(episode))

            # 2nd closest waypoint
            second_closest_waypoint = (waypoints_length + closest_waypoint - 1) % waypoints_length
            x2 = waypoints_mid[second_closest_waypoint][0]
            y2 = waypoints_mid[second_closest_waypoint][1]
            dist2 = distance_of_2points([x, y], [x2, y2])

            second_closest_waypoint = (closest_waypoint + 1) % waypoints_length
            x3 = waypoints_mid[second_closest_waypoint][0]
            y3 = waypoints_mid[second_closest_waypoint][1]
            dist3 = distance_of_2points([x, y], [x2, y2])

            if dist3 < dist2:
                x2 = x3
                y2 = y3
                dist2 = dist3
            ax.plot([x, x2], [y, y2], color=get_color_name(episode + 1), linewidth=display_setup['dist_line_width'], linestyle='-.')
            a, b, c = line_2p([x, y], [x2, y2])
            x_m = 0.5 * (x + x2)
            y_m = -1 * (a * x_m + c)/b
            ax.text(x_m, y_m, '{:.2f}'.format(dist2), fontsize=display_setup['fontsize'], color=get_color_name(episode))


if __name__ == '__main__':
    waypoints_npy_file = r'npy\ChampionshipCup2019_track.npy'
    training_log_dir = r'aws\training-simtrace\2019\track2019'

    fig = plt.figure(figsize=display_setup['figure_size'], dpi=display_setup['dpi'])
    mng = plt.get_current_fig_manager()
    ax = fig.add_subplot()

    waypoints_mid, waypoints_inn, waypoints_out = get_waypoints(waypoints_npy_file)
    if display_setup['display_waypoints_mid'] or \
            display_setup['display_waypoints_inn'] or \
            display_setup['display_waypoints_out']:
        plot_waypoints(ax, waypoints_mid, waypoints_inn, waypoints_out)

    training_log = training_log_dir + r'\44-iteration.csv'
    df = read_log(training_log, episode_num=880, steps=4)

    plot_dataframe_new(df, ax, waypoints_mid, waypoints_inn, waypoints_out)

    plt.grid(True)
    mng.window.state("zoomed")
    plt.show()
    plt.close()
