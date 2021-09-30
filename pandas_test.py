import numpy as np

import pandas as pd

csv_file = r'aws\training-simtrace\2019\dlcf-test-clone\0-iteration.csv'

column_names = ['episode', 'steps', 'X', 'Y', 'yaw', 'steer', 'throttle', 'action1', 'action12', 'reward', 'done',
                'all_wheels_on_track', 'progress', 'closest_waypoint', 'track_len', 'tstamp', 'episode_status',
                'pause_duration']

column_dtype = {'episode': int, 'steps': int, 'X': float, 'Y': float, 'yaw': float, 'steer': float,
                'throttle': float, 'action1': str, 'action12': str, 'reward': float, 'done': bool,
                'all_wheels_on_track': bool, 'progress': float, 'closest_waypoint': int, 'track_len': float,
                'tstamp': float, 'episode_status': str, 'pause_duration': float}

df = pd.read_csv(csv_file, engine='python', names=column_names, skiprows=1)

print(df.info())
