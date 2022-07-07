import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from aws_deepracer.functions_2d import *
from csv_file_combine import input_file_dir, output_file_name

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
    'display_steering_arrow': True,
    'display_action': False,
    'display_training_reward': True,
    'display_projected_track': False,
    'display_distance_to_next_waypoint': False,
    'display_distance_to_prev_waypoint': False,
    'display_distance_to_center_line': False,
    'heading_arrow_width': 0.0005,
    'dist_line_width': 0.2
}


def get_waypoints(npy_file):
    waypoints = np.load(npy_file)
    waypoints_mid = waypoints[:, 0:2]
    waypoints_inn = waypoints[:, 2:4]
    waypoints_out = waypoints[:, 4:6]
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

    try:
        df = pd.read_csv(log_file, engine='python', skiprows=1, names=column_names, dtype=column_dtype)
    except ValueError:
        column_names = ['episode', 'steps', 'X', 'Y', 'yaw', 'steer', 'throttle', 'action', 'reward', 'done',
                        'all_wheels_on_track', 'progress', 'closest_waypoint', 'track_len', 'tstamp', 'episode_status',
                        'pause_duration']

        column_dtype = {'episode': int, 'steps': int, 'X': float, 'Y': float, 'yaw': float, 'steer': float,
                        'throttle': float, 'action': float, 'reward': float, 'done': bool,
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
    ind += 7
    for c in colors.cnames:
        # if i == ind % len(colors.cnames):
        if i == ind % 24:
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


def plot_dataframe_new(df, ax, waypoints_mid, waypoints_inn, waypoints_out):
    i = 0
    for episode, steps, x, y, yaw, steer, throttle, reward, progress, closest_waypoint, tstamp, episode_status \
            in zip(df['episode'], df['steps'], df['X'], df['Y'], df['yaw'], df['steer'], df['throttle'], df['reward'],
                   df['progress'], df['closest_waypoint'], df['tstamp'], df['episode_status']):

        ax.scatter(x, y, c=get_color_name(episode), s=display_setup['dot_size'])

        if display_setup['display_steps']:
            ax.text(x, y - 0.005, str(steps), fontsize=display_setup['fontsize'])

        if display_setup['display_heading_arrow']:
            x1 = math.cos(math.radians(yaw))
            y1 = math.sin(math.radians(yaw))
            ax.quiver(x, y, x1, y1, angles='xy', scale_units='xy', scale=10,
                      color=get_color_name(episode), width=display_setup['heading_arrow_width'])

        if display_setup['display_steering_arrow']:
            x1 = math.cos(math.radians(yaw + steer)) * 0.5
            y1 = math.sin(math.radians(yaw + steer)) * 0.5
            ax.quiver(x, y, x1, y1, angles='xy', scale_units='xy', scale=10,
                      color=get_color_name(episode + 1), width=display_setup['heading_arrow_width'])

        if display_setup['display_projected_track']:
            if steps > 1 and episode_status != 'off_track' and i + 1 < len(df):
                t = df.iloc[i + 1]['tstamp'] - df.iloc[i]['tstamp']
                x1 = x + t * throttle * math.cos(math.radians(yaw))
                y1 = y + t * throttle * math.sin(math.radians(yaw))
                ax.scatter(x1, y1, s=display_setup['dot_size'], c='r')
                ax.text(x1, y1, str(steps + 1), fontsize=display_setup['fontsize'])

        if display_setup['display_action']:
            action = '[{:.1f},{:.2f}]'.format(steer, throttle)
            ax.text(x, y, action, fontsize=display_setup['fontsize'], color=get_color_name(episode))

        if display_setup['display_training_reward']:
            ax.text(x, y + 0.005, '{:.3f}'.format(reward), fontsize=display_setup['fontsize'],
                    color=get_color_name(episode))

        if display_setup['display_distance_to_next_waypoint']:
            closest_waypoints = closest_2_racing_points_index(waypoints_mid, [x, y])
            closest_waypoints = next_prev_racing_point(waypoints_mid, closest_waypoints, [x, y], yaw)

            x1 = waypoints_mid[closest_waypoints[1]][0]
            y1 = waypoints_mid[closest_waypoints[1]][1]

            # closest waypoint
            dist = distance_of_2points([x, y], [x1, y1])
            ax.plot([x, x1], [y, y1], color=get_color_name(episode), linewidth=display_setup['dist_line_width'], linestyle='-.')
            a, b, c = line_2p([x, y], [x1, y1])
            x_m = 0.5 * (x + x1)
            y_m = -1 * (a * x_m + c)/b
            ax.text(x_m, y_m, '{:.2f}'.format(dist), fontsize=display_setup['fontsize'], color=get_color_name(episode))

            if display_setup['display_distance_to_prev_waypoint']:
                x1 = waypoints_mid[closest_waypoints[0]][0]
                y1 = waypoints_mid[closest_waypoints[0]][1]

                # closest waypoint
                dist = distance_of_2points([x, y], [x1, y1])
                ax.plot([x, x1], [y, y1], color=get_color_name(episode + 1), linewidth=display_setup['dist_line_width'], linestyle='-.')
                a, b, c = line_2p([x, y], [x1, y1])
                x_m = 0.5 * (x + x1)
                y_m = -1 * (a * x_m + c) / b
                ax.text(x_m, y_m, '{:.2f}'.format(dist), fontsize=display_setup['fontsize'], color=get_color_name(episode + 1))

        if display_setup['display_distance_to_center_line']:
            closest_waypoints = closest_2_racing_points_index(waypoints_mid, [x, y])
            closest_waypoints = next_prev_racing_point(waypoints_mid, closest_waypoints, [x, y], yaw)

            ax.plot([waypoints_mid[closest_waypoints[0]][0], waypoints_mid[closest_waypoints[1]][0]],
                    [waypoints_mid[closest_waypoints[0]][1], waypoints_mid[closest_waypoints[1]][1]],
                    color=get_color_name(episode), linewidth=display_setup['dist_line_width'], linestyle='-.')

            a, b, c = line_2p(waypoints_mid[closest_waypoints[0]], waypoints_mid[closest_waypoints[1]])
            vertical_point = vertical_point_of_point_to_line([x, y], a, b, c)

            ax.plot([x, vertical_point[0]], [y, vertical_point[1]], color=get_color_name(episode), linewidth=display_setup['dist_line_width'], linestyle='-.')

            distance = distance_of_2points([x, y], vertical_point)

            a, b, c = line_2p([x, y], vertical_point)
            x_m = 0.5 * (x + vertical_point[0])
            y_m = -1 * (a * x_m + c) / b

            ax.text(x_m, y_m, '{:.2f}'.format(distance), fontsize=display_setup['fontsize'], color=get_color_name(episode))

        i += 1


def closest_2_racing_points_index(racing_coords, car_coords):
    distances = []
    for i in range(len(racing_coords)):
        distance = distance_of_2points(racing_coords[i], car_coords)
        distances.append(distance)

    closest_index = distances.index(min(distances))

    distances_no_closest = distances.copy()
    distances_no_closest[closest_index] = 999
    second_closest_index = distances_no_closest.index(min(distances_no_closest))

    return [closest_index, second_closest_index]


def next_prev_racing_point(waypoints, closest_waypoints, car_coords, heading):
    closest_coords = waypoints[closest_waypoints[0]]
    second_closest_coords = waypoints[closest_waypoints[1]]

    heading_vector = [math.cos(math.radians(heading)), math.sin(math.radians(heading))]

    new_car_coords = [car_coords[0] + heading_vector[0], car_coords[1] + heading_vector[1]]

    distance_closest_coords_new = distance_of_2points(closest_coords, new_car_coords)
    distance_second_closest_coords_new = distance_of_2points(second_closest_coords, new_car_coords)

    if distance_closest_coords_new <= distance_second_closest_coords_new:
        next_point = closest_waypoints[0]
        prev_point = closest_waypoints[1]
    else:
        next_point = closest_waypoints[1]
        prev_point = closest_waypoints[0]

    return [prev_point, next_point]


if __name__ == '__main__':
    waypoints_npy_file = r'../npy/reinvent_base.npy'
    #
    fig = plt.figure(figsize=display_setup['figure_size'], dpi=display_setup['dpi'])
    mng = plt.get_current_fig_manager()
    ax = fig.add_subplot()
    #
    waypoints_mid, waypoints_inn, waypoints_out = get_waypoints(waypoints_npy_file)
    if display_setup['display_waypoints_mid'] or \
            display_setup['display_waypoints_inn'] or \
            display_setup['display_waypoints_out']:
        plot_waypoints(ax, waypoints_mid, waypoints_inn, waypoints_out)
    #
    training_log = input_file_dir + output_file_name
    df = read_log(training_log, episode_num=3, steps=0)

    #增加选择条件参看特定的点
    # df = df[df.throttle >= 1.5]
    # df = df[df.episode == 173]
    # df = df[df.reward == 1.5]
    # df = df[df.episode_status == 'off_track']
    #
    plot_dataframe_new(df, ax, waypoints_mid, waypoints_inn, waypoints_out)
    #
    plt.grid(True)
    mng.window.state("zoomed")
    plt.show()
    plt.clf()
    plt.close()
