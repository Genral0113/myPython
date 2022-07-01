import os
import matplotlib.pyplot as plt
from training_log_viewer import get_waypoints, plot_waypoints, display_setup
from aws_deepracer.functions_2d import *


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

    mng.window.state("zoomed")
    plt.title(os.path.basename(track_file).split('.')[0])
    plt.legend(legends)
    plt.show()
    plt.close()


if __name__ == '__main__':
    track_file_dir = os.path.join(os.path.dirname(__file__), r'../npy')
    # track_file = r'ChampionshipCup2019_track.npy'
    track_file = r'reinvent_base.npy'
    # track_file = r'Albert.npy'
    # for track_file in os.listdir(track_file_dir):
    #     track_file_name = os.path.join(track_file_dir, track_file)
    #     if os.path.isfile(track_file_name) and track_file_name[-3:] == 'npy':
    #         plot_track(track_file_name)
    #         # print(os.path.basename(track_file).split('.')[0])
    track_file_name = os.path.join(track_file_dir, track_file)
    plot_track(track_file_name)
