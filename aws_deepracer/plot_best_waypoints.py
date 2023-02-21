import os
import matplotlib.pyplot as plt
from training_log_viewer import get_waypoints, plot_waypoints, display_setup
from aws_deepracer.functions_2d import *

racing_track = np.array([[2.89674, 0.70087, 4.0, 0.07644],
                [3.16466, 0.69299, 4.0, 0.06701],
                [3.43292, 0.68822, 4.0, 0.06708],
                [3.7378, 0.68548, 4.0, 0.07622],
                [4.10749, 0.68438, 4.0, 0.09242],
                [4.41121, 0.68403, 3.02716, 0.10033],
                [4.70859, 0.68388, 2.30663, 0.12893],
                [5.32, 0.68405, 2.04575, 0.29887],
                [5.42, 0.68409, 1.93502, 0.05168],
                [5.78, 0.68422, 1.881, 0.19139],
                [6.22029, 0.69528, 1.87714, 0.23463],
                [6.40459, 0.72226, 1.87714, 0.09923],
                [6.55489, 0.76197, 1.87714, 0.08281],
                [6.69764, 0.82037, 1.87103, 0.08243],
                [6.83824, 0.90421, 1.75526, 0.09326],
                [6.97477, 1.02121, 1.75526, 0.10244],
                [7.08704, 1.15896, 1.75526, 0.10124],
                [7.16979, 1.30619, 1.75526, 0.09622],
                [7.26217, 1.69748, 1.75526, 0.22906],
                [7.26012, 1.79605, 1.75526, 0.05617],
                [7.2437, 1.91019, 1.77887, 0.06482],
                [7.10671, 2.23205, 1.81561, 0.19266],
                [6.94118, 2.41942, 1.91849, 0.13032],
                [6.72311, 2.56894, 2.09321, 0.12632],
                [6.47578, 2.66988, 2.37371, 0.11254],
                [6.21929, 2.72778, 2.08259, 0.12626],
                [5.97558, 2.75403, 2.08259, 0.1177],
                [5.76663, 2.76273, 2.08259, 0.10042],
                [5.51021, 2.76516, 2.08259, 0.12313],
                [5.23435, 2.76857, 2.08259, 0.13247],
                [5.0914, 2.78674, 2.08259, 0.06919],
                [4.95258, 2.82112, 2.12744, 0.06723],
                [4.79184, 2.88257, 2.32841, 0.07391],
                [4.59011, 2.99156, 2.58892, 0.08856],
                [4.36941, 3.15147, 2.96609, 0.09189],
                [4.16698, 3.33482, 2.19148, 0.12463],
                [3.97893, 3.53254, 1.85593, 0.14702],
                [3.82569, 3.70718, 1.80469, 0.12874],
                [3.68331, 3.87352, 1.80469, 0.12133],
                [3.54906, 4.03738, 1.80469, 0.11738],
                [3.33596, 4.2533, 1.80469, 0.1681],
                [3.20369, 4.34796, 1.80469, 0.09013],
                [3.08059, 4.41014, 1.80469, 0.07642],
                [2.9599, 4.45075, 1.91555, 0.06648],
                [2.83522, 4.47635, 2.2135, 0.0575],
                [2.69007, 4.49159, 2.17351, 0.06715],
                [2.49518, 4.49747, 2.06484, 0.09443],
                [2.24938, 4.49143, 1.90082, 0.12935],
                [1.98541, 4.47498, 1.90082, 0.13914],
                [1.70213, 4.40925, 1.90082, 0.15299],
                [1.41598, 4.27228, 1.90082, 0.1669],
                [1.16268, 4.06456, 1.90082, 0.17234],
                [0.96753, 3.78363, 1.90082, 0.17995],
                [0.87363, 3.43687, 2.4319, 0.14772],
                [0.85453, 3.09651, 2.87359, 0.11863],
                [0.8766, 2.81168, 3.52509, 0.08104],
                [0.91229, 2.57756, 2.71735, 0.08715],
                [0.96294, 2.31103, 1.5421, 0.17592],
                [1.00528, 2.10016, 1.3, 0.16545],
                [1.04306, 1.91271, 1.3, 0.14709],
                [1.09363, 1.66868, 1.3, 0.1917],
                [1.20742, 1.22666, 1.3, 0.3511],
                [1.24898, 1.12468, 1.3, 0.08471],
                [1.29166, 1.05197, 1.3, 0.06485],
                [1.34858, 0.98568, 1.30451, 0.06698],
                [1.42791, 0.92327, 1.47131, 0.0686],
                [1.54143, 0.8641, 1.82331, 0.07021],
                [1.71765, 0.80674, 2.49241, 0.07435],
                [2.14683, 0.74654, 4.0, 0.10835],
                [2.59129, 0.71477, 4.0, 0.1114]])


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

    ax.plot(racing_track[:, 0:2][:, 0:1], racing_track[:, 0:2][:, 1:2], c='r', linestyle='-.', linewidth=1)

    mng.window.state("zoomed")
    plt.title(os.path.basename(track_file).split('.')[0])
    plt.legend(legends)
    plt.show()
    plt.close()


if __name__ == '__main__':
    track_file = r'../npy/reinvent_base.npy'
    plot_track(track_file)
    # print(racing_track[:, 0:2][:, 1:2])
