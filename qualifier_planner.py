import math
import matplotlib.pyplot as plt
import numpy as np

TRACK_FILE = "ChampionshipCup2019_track.npy"

# Parameters
FUTURE_STEP = 6
MID_STEP = 4
TURN_THRESHOLD = 8  # degrees
DIST_THRESHOLD = 0.8  # metres

# Colour macros
FAST = 0
SLOW = 1
BONUS_FAST = 2


def identify_corner(waypoints, closest_waypoints, future_step):
    # Identify next waypoint and a further waypoint
    point_prev = waypoints[closest_waypoints[0]]
    point_next = waypoints[closest_waypoints[1]]
    point_future = waypoints[min(len(waypoints) - 1,  closest_waypoints[1] + future_step)]

    # Calculate headings to waypoints
    heading_current = math.degrees(math.atan2(point_prev[1] - point_next[1], point_prev[0] - point_next[0]))
    heading_future = math.degrees(math.atan2(point_prev[1] - point_future[1], point_prev[0] - point_future[0]))

    # Calculate the difference between the headings
    diff_heading = abs(heading_current - heading_future)

    # Check we didn't choose the reflex angle
    if diff_heading > 180:
        diff_heading = 360 - diff_heading

    # Calculate distance to further waypoint
    dist_future = np.linalg.norm([point_next[0] - point_future[0], point_next[1] - point_future[1]])

    return diff_heading, dist_future


# This is a modified version of the actual select_speed function used in
# reward_qualifier.py so that there is a 3rd possible return value to allow
# visualisation of the "bonus fast" points
def select_speed(waypoints, closest_waypoints, future_step, mid_step):
    # Identify if a corner is in the future
    diff_heading, dist_future = identify_corner(waypoints, closest_waypoints, future_step)

    if diff_heading < TURN_THRESHOLD:
        # If there's no corner encourage going faster
        speed_colour = FAST
    else:
        if dist_future < DIST_THRESHOLD:
            # If there is a corner and it's close encourage going slower
            speed_colour = SLOW
        else:
            # If the corner is far away, re-assess closer points
            diff_heading_mid, dist_mid = identify_corner(waypoints, closest_waypoints, mid_step)

            if diff_heading_mid < TURN_THRESHOLD:
                # If there's no corner encourage going faster
                speed_colour = BONUS_FAST
            else:
                # If there is a corner and it's close encourage going slower
                speed_colour = SLOW

    return speed_colour


# Get waypoints from numpy file
waypoints = np.load(TRACK_FILE)

# Extract the x and y columns from the waypoints
waypoints = waypoints[:, 2:4]

print("----- Parameters -----")
print("   FUTURE_STEP: %d" % (FUTURE_STEP))
print("      MID_STEP: %d" % (MID_STEP))
print("TURN_THRESHOLD: %d" % (TURN_THRESHOLD))
print("DIST_THRESHOLD: %.1f" % (DIST_THRESHOLD))
print("----------------------")

color_dict = {0: '#ff7f0e', 1: '#1f77b4', 2: '#ff460e'}
label_dict = {0: 'Fast Incentive', 1: 'Slow Incentive', 2: 'Bonus Fast Incentive'}

colours = []

TURN_THRESHOLD_FUTURE_CALCULATED = 0
TURN_THRESHOLD_MID_CALCULATED = 0
TURN_THRESHOLD_8_CALCULATED = 0
DIST_THRESHOLD_FUTURE_CALCULATED = 0
DIST_THRESHOLD_MID_CALCULATED = 0
DIST_THRESHOLD_8_CALCULATED = 0

for i in range(len(waypoints)):
    # Simulate input parameter
    closest_waypoints = [i - 1, i]

    # Determine what speed will be rewarded
    speed_colour = select_speed(waypoints, closest_waypoints, FUTURE_STEP, MID_STEP)
    colours.append(speed_colour)

    temp_diff, temp_dist = identify_corner(waypoints, closest_waypoints, FUTURE_STEP)
    if temp_diff > TURN_THRESHOLD_FUTURE_CALCULATED:
        TURN_THRESHOLD_FUTURE_CALCULATED = temp_diff
    if temp_dist > DIST_THRESHOLD_FUTURE_CALCULATED:
        DIST_THRESHOLD_FUTURE_CALCULATED = temp_dist

    temp_diff, temp_dist = identify_corner(waypoints, closest_waypoints, MID_STEP)
    if temp_diff > TURN_THRESHOLD_MID_CALCULATED:
        TURN_THRESHOLD_MID_CALCULATED = temp_diff
    if temp_dist > DIST_THRESHOLD_MID_CALCULATED:
        DIST_THRESHOLD_MID_CALCULATED = temp_dist

    temp_diff_8, temp_dist_8 = identify_corner(waypoints, closest_waypoints, 8)
    if temp_diff_8 > TURN_THRESHOLD_8_CALCULATED:
        TURN_THRESHOLD_8_CALCULATED = temp_diff_8
    if temp_dist_8 > DIST_THRESHOLD_8_CALCULATED:
        DIST_THRESHOLD_8_CALCULATED = temp_dist_8

print('The maximum direction difference of next {} points is {}'.format(FUTURE_STEP, TURN_THRESHOLD_FUTURE_CALCULATED))
print('The maximum distance difference of next {} points is {}'.format(FUTURE_STEP, DIST_THRESHOLD_FUTURE_CALCULATED))
print('The maximum direction difference of next {} points is {}'.format(MID_STEP, TURN_THRESHOLD_MID_CALCULATED))
print('The maximum distance difference of next {} points is {}'.format(MID_STEP, DIST_THRESHOLD_MID_CALCULATED))
print('The maximum direction difference of next {} points is {}'.format(8, TURN_THRESHOLD_8_CALCULATED))
print('The maximum distance difference of next {} points is {}'.format(8, DIST_THRESHOLD_8_CALCULATED))

# Plot points
fig, ax = plt.subplots()

for g in np.unique(colours):
    ix = np.where(colours == g)
    ax.scatter(waypoints[ix, 0], waypoints[ix, 1], c=color_dict[g], label=label_dict[g])

ax.legend(fancybox=True, shadow=True)
ax.set_aspect('equal')
plt.axis('off')

plt.show()
plt.close()
