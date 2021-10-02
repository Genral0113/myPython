import matplotlib.pyplot as plt
from training_log_viewer import get_waypoints, plot_waypoints, display_setup


track_file = r'npy\Spain_track.npy'
track_file = r'npy\China_track.npy'
track_file = r'npy\Albert.npy'
waypoints1, waypoints2, waypoints3 = get_waypoints(track_file)

fig = plt.figure(figsize=display_setup['figure_size'], dpi=display_setup['dpi'])
mng = plt.get_current_fig_manager()

ax = fig.add_subplot()

plot_waypoints(ax, waypoints1, waypoints2, waypoints3)

mng.window.state("zoomed")
plt.show()
plt.close()
