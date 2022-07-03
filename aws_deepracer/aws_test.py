import math
from training_log_viewer import get_waypoints
import numpy as np


def distance_of_2points(p1, p2):
    return ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5


def directions_of_2points(p1, p2):
    directions = math.atan2(p2[1] - p1[1], p2[0] - p1[0])
    directions = math.degrees(directions)
    return directions


def get_track_direction(waypoints, closet_waypoints, next_waypoints):
    prev_waypoint = min(closet_waypoints)
    next_waypoint = max(closet_waypoints)
    if prev_waypoint == 0 and next_waypoint == len(waypoints) - 1:
        prev_waypoint = next_waypoint
        next_waypoint = 0
    if next_waypoints > 0:
        next_waypoint = (next_waypoint + next_waypoints) % len(waypoints)
    return directions_of_2points(waypoints[prev_waypoint], waypoints[next_waypoint])


def read_waypoints(track_file):
    waypoints = np.load(track_file)
    waypoints_mid = waypoints[:, 0:2]
    waypoints_inn = waypoints[:, 2:4]
    waypoints_out = waypoints[:, 4:6]
    return waypoints_mid, waypoints_inn, waypoints_out


if __name__ == '__main__':
    track_file = r'../npy/reinvent_base.npy'
    waypoints_mid, waypoints_inn, waypoints_out = get_waypoints(track_file)

    # closet_waypoints = [113, 112]
    # print('the track direction is {}'.format(get_track_direction(waypoints_mid, closet_waypoints, 1)))

    for i in range(len(waypoints_mid)):
        # track_direction = directions_of_2points(waypoints_mid[i], waypoints_mid[(i + 1) % len(waypoints_mid)])
        # distance = distance_of_2points(waypoints_mid[i], waypoints_mid[(i + 1) % len(waypoints_mid)])
        # print('waypoints({}, {}), distance {}, direction {}'.format(i, (i + 1) % len(waypoints_mid), distance, track_direction))
        track_width = distance_of_2points(waypoints_inn[i], waypoints_out[i])
        print('the width of {}th waypoint is {}'.format(i + 1, track_width))

    if waypoints_mid[0][1] == waypoints_mid[118][1]:
        print('the {}th and {}th are the same'.format(0, 118))
