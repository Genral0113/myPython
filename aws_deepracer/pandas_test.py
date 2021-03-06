import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from training_log_viewer import read_log, get_waypoints, plot_waypoints, plot_dataframe, display_setup

waypoints_npy_file = r'../npy/ChampionshipCup2019_track.npy'
csv_file = r'../aws/training-simtrace/2019/dlcf-test-clone/0-iteration.csv'
waypoints_length = 154

df = read_log(csv_file, episode_num=-1)

fig = plt.figure(figsize=display_setup['figure_size'], dpi=display_setup['dpi'])
mng = plt.get_current_fig_manager()

ax = fig.add_subplot(2, 2, 1)

waypoints_mid, waypoints_inn, waypoints_out = get_waypoints(waypoints_npy_file)
if display_setup['display_waypoints']:
    plot_waypoints(ax, waypoints_mid, waypoints_inn, waypoints_out)

plot_dataframe(df, ax)

ax = fig.add_subplot(2, 1, 2)
df_group_by_episode = df.groupby('episode')
for name, group in df_group_by_episode:
    ax.plot(name, group['reward'].sum(), c='b')

mng.window.state("zoomed")
plt.show()
plt.close()
