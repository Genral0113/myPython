import os
import matplotlib.pyplot as plt
from training_log_viewer import get_waypoints, plot_waypoints, display_setup
from functions_2d import *


def plot_track(track_file):
    waypoints1, waypoints2, waypoints3 = get_waypoints(track_file)

    fig = plt.figure(figsize=display_setup['figure_size'], dpi=display_setup['dpi'])
    mng = plt.get_current_fig_manager()

    ax = fig.add_subplot()

    plot_waypoints(ax, waypoints1, waypoints2, waypoints3)

    waypoints_length = len(waypoints1) - 1
    track_length = 0
    for i in range(waypoints_length):
        p1 = [waypoints1[i][0], waypoints1[i][1]]
        p2 = [waypoints1[(i + 1) % waypoints_length][0], waypoints1[(i + 1) % waypoints_length][1]]
        track_length += distance_of_2points(p1, p2)
    print('the track length is {}'.format(track_length))

    mng.window.state("zoomed")
    plt.title(os.path.basename(track_file).split('.')[0])
    plt.show()
    plt.close()


if __name__ == '__main__':
    track_file_dir = os.path.join(os.path.dirname(__file__), r'npy')
    track_file = r'ChampionshipCup2019_track.npy'
    # for track_file in os.listdir(track_file_dir):
    #     track_file_name = os.path.join(track_file_dir, track_file)
    #     if os.path.isfile(track_file_name) and track_file_name[-3:] == 'npy':
    #         plot_track(track_file_name)
    #         # print(os.path.basename(track_file).split('.')[0])
    track_file_name = os.path.join(track_file_dir, track_file)
    plot_track(track_file_name)
