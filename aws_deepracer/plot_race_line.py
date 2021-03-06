import os
import matplotlib.pyplot as plt
from training_log_viewer import get_waypoints, plot_waypoints, display_setup
from aws_deepracer.functions_2d import *


start_waypoints = [1, 23, 29, 34, 38, 42, 51, 69, 81, 85, 90, 108, 119]
start_waypoints = [1, 23, 29, 34, 38, 42, 73, 81, 85, 90, 108, 119]

start_waypoints = [1, 24, 33, 41, 71, 81, 90, 106, 111, 119]
start_waypoints2 = [1, 24, 33, 41, 73, 81, 90, 106, 111, 119]
car_actions = ['speed_up', 'slow_down', 'speed_up', 'speed_up', 'speed_up', 'slow_down', 'speed_up', 'slow_down', 'speed_up']


def plot_track(track_file):
    waypoints_mid, waypoints_inn, waypoints_out = get_waypoints(track_file)

    fig = plt.figure(figsize=display_setup['figure_size'], dpi=display_setup['dpi'])
    mng = plt.get_current_fig_manager()

    ax = fig.add_subplot()
    legends = []

    plot_waypoints(ax, waypoints_mid, waypoints_inn, waypoints_out)
    waypoints_length = len(waypoints_mid)
    if waypoints_mid[0][0] == waypoints_mid[-1][0] and waypoints_mid[0][1] == waypoints_mid[-1][1]:
        waypoints_length -= 1
    legends.append('waypoints : {}'.format(waypoints_length))

    track_length = 0
    for i in range(waypoints_length):
        p1 = [waypoints_mid[i][0], waypoints_mid[i][1]]
        p2 = [waypoints_mid[(i + 1) % waypoints_length][0], waypoints_mid[(i + 1) % waypoints_length][1]]
        track_length += distance_of_2points(p1, p2)
    legends.append('track length : {:.3f}'.format(track_length))

    racing_points_x = []
    racing_points_y = []
    for i in range(len(start_waypoints) - 1):
        end_waypoint = start_waypoints[i + 1] - 1
        racing_points_x.append((waypoints_inn[end_waypoint][0] + waypoints_mid[end_waypoint][0]) * 0.5)
        racing_points_y.append((waypoints_inn[end_waypoint][1] + waypoints_mid[end_waypoint][1]) * 0.5)
    racing_points_x.append(racing_points_x[0])
    racing_points_y.append(racing_points_y[0])
    ax.plot(racing_points_x, racing_points_y, c='r', linestyle='-.', linewidth=1)

    racing_line_length = 0
    for i in range(len(racing_points_x) - 1):
        racing_line_length += distance_of_2points([racing_points_x[i], racing_points_y[i]], [racing_points_x[i + 1], racing_points_y[i + 1]])
    legends.append('racing line length : {:.3f}'.format(racing_line_length))

    if len(start_waypoints2) > 0:
        racing_points_x = []
        racing_points_y = []
        for i in range(len(start_waypoints2) - 1):
            end_waypoint = start_waypoints2[i + 1] - 1
            racing_points_x.append((waypoints_inn[end_waypoint][0] + waypoints_mid[end_waypoint][0]) * 0.5)
            racing_points_y.append((waypoints_inn[end_waypoint][1] + waypoints_mid[end_waypoint][1]) * 0.5)
        racing_points_x.append(racing_points_x[0])
        racing_points_y.append(racing_points_y[0])
        ax.plot(racing_points_x, racing_points_y, c='darkred', linestyle='-.', linewidth=1)

        racing_line_length = 0
        for i in range(len(racing_points_x) - 1):
            racing_line_length += distance_of_2points([racing_points_x[i], racing_points_y[i]],
                                                      [racing_points_x[i + 1], racing_points_y[i + 1]])

        waypoints_length = len(waypoints_mid)
        if waypoints_mid[0][0] == waypoints_mid[-1][0] and waypoints_mid[0][1] == waypoints_mid[-1][1]:
            waypoints_length -= 1
        legends.append('racing line2 length : {:.3f}'.format(racing_line_length))

    mng.window.state("zoomed")
    plt.title(os.path.basename(track_file).split('.')[0])
    plt.legend(legends)
    plt.show()
    plt.close()


if __name__ == '__main__':
    track_file = r'../npy/reinvent_base.npy'
    plot_track(track_file)
