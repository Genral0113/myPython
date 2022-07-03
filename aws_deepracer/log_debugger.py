import pandas as pd
from htc_awsdeepracer_2022_v4 import *


def read_csv_file(file_name, episode_num=-1):

    column_names = ['episode', 'steps', 'X', 'Y', 'yam', 'steer', 'throttle', 'action1', 'action2', 'reward', 'done',
                    'all_wheels_on_track', 'progress', 'closest_waypoint', 'track_len', 'tstamp', 'episode_status',
                    'pause_duration']

    column_dtype = {'episode': int, 'steps': int, 'X': float, 'Y': float, 'yaw': float, 'steer': float,
                    'throttle': float, 'action1': str, 'action12': str, 'reward': float, 'done': bool,
                    'all_wheels_on_track': bool, 'progress': float, 'closest_waypoint': int, 'track_len': float,
                    'tstamp': float, 'episode_status': str, 'pause_duration': float}

    df = pd.read_csv(file_name, skiprows=1, names=column_names, dtype=column_dtype)

    if episode_num >= 0:
        df = df[df.episode == episode_num]

    log_parmas = {
        'episode': df['episode'].tolist(),
        'steps': df['steps'].tolist(),
        'x': df['X'].tolist(),
        'y': df['Y'].tolist(),
        'yam': df['yam'].tolist(),
        'steer': df['steer'].tolist(),
        'throttle': df['throttle'].tolist(),
        # 'action': df['action1'] + ',' + df['action2'],
        'reward': df['reward'].tolist(),
        'done': df['done'].tolist(),
        'all_wheels_on_track': df['all_wheels_on_track'].tolist(),
        'progress': df['progress'].tolist(),
        'closest_waypoint': df['closest_waypoint'].tolist(),
        'track_len': df['track_len'].tolist(),
        'tstamp': df['tstamp'].tolist(),
        'episode_status': df['episode_status'].tolist(),
        'pause_duration': df['pause_duration'].tolist()
    }

    return log_parmas


def get_waypoints(npy_file):
    waypoints = np.load(npy_file)
    waypoints_mid = waypoints[:, 0:2]
    waypoints_inn = waypoints[:, 2:4]
    waypoints_out = waypoints[:, 4:6]
    return waypoints_mid, waypoints_inn, waypoints_out


def distance_of_2points(p1, p2):
    dist = np.linalg.norm([p1[0] - p2[0], p1[1] - p2[1]])
    return dist


def distance_to_racing_line(closest_coords, second_closest_coords, car_coords):

        # Calculate the distances between 2 closest racing points
        a = distance_of_2points(closest_coords, second_closest_coords)

        # Distances between car and closest and second closest racing point
        b = distance_of_2points(car_coords, closest_coords)
        c = distance_of_2points(car_coords, second_closest_coords)

        # Calculate distance between car and racing line (goes through 2 closest racing points)
        # try-except in case a=0
        try:
            distance = abs(-(a ** 4) + 2 * (a ** 2) * (b ** 2) + 2 * (a ** 2) * (c ** 2) - (b ** 4) + 2 * (b ** 2) * (c ** 2) - (c ** 4)) ** 0.5 / (2 * a)
        except:
            distance = b

        return distance


def is_left_of_center(waypoints_inn, waypoints_out, car_coords, closest_waypoint):
    waypoints_length = len(waypoints_inn)
    if waypoints_inn[0][0] == waypoints_inn[-1][0] and waypoints_inn[0][1] == waypoints_inn[-1][1]:
        waypoints_length -= 1

    closest_waypoint = closest_waypoint % waypoints_length

    distance_inn = distance_of_2points(car_coords, waypoints_inn[closest_waypoint])
    distance_out = distance_of_2points(car_coords, waypoints_out[closest_waypoint])

    left_side = False
    if distance_inn < distance_out:
        left_side = True
    return left_side


def get_closest_waypoints(waypoints, car_coords):

    waypoints_length = len(waypoints)
    if waypoints[0][0] == waypoints[-1][0] and waypoints[0][1] == waypoints[-1][1]:
        waypoints_length -= 1

    closest_waypoint = -1
    second_closest_waypoint = -1

    distance = 999
    for i in range(waypoints_length):
        if distance_of_2points(car_coords, waypoints[i]) <= distance:
            closest_waypoint = i
            distance = distance_of_2points(car_coords, waypoints[i])

    distance = 999
    for i in range(waypoints_length):
        if distance_of_2points(car_coords, waypoints[i]) <= distance:
            if i != closest_waypoint:
                second_closest_waypoint = i
                distance = distance_of_2points(car_coords, waypoints[i])

    return [closest_waypoint, second_closest_waypoint]


if __name__ == '__main__':
    track_file = r'../npy/reinvent_base.npy'

    log_file = r'C:\Users\asus\Desktop\2022 aws\4b0ffbd9-9b3a-4941-a697-314f82831812\sim-trace\evaluation\20220701002122-O-aHYUMxQWCftUHHNMCq7Q\evaluation-simtrace\0-iteration.csv'
    log_file = r'C:\Users\asus\Desktop\2022 aws\autobus-v3-training_job_G3i0ZBFqRlOX86MhVbgxYA_logs\3dbebb02-802a-4273-909d-3014dd8f3855\sim-trace\training\training-simtrace\0-iteration.csv'
    log_file = r'C:\Users\asus\Desktop\2022 aws\autobus-v3-training_job_G3i0ZBFqRlOX86MhVbgxYA_logs\3dbebb02-802a-4273-909d-3014dd8f3855\sim-trace\training\training-simtrace\10-iteration.csv'
    log_file = r'C:\Users\asus\Desktop\2022 aws\autobus-v6-evaluation_job_J3aa6-cOR0KPAARSeR9Czg_logs\53fb60e8-1f8a-4ec7-b87d-76d5a6b35f60\sim-trace\evaluation\20220702234903-J3aa6-cOR0KPAARSeR9Czg\evaluation-simtrace\0-iteration.csv'
    log_file = r'C:\Users\asus\Desktop\2022 aws\model-v7-training_job_qMARHs5_T0WxP8RmuQ837Q_logs\74dff1a2-b748-4975-aea9-3c9de4f79e35\sim-trace\training\training-simtrace\all-iterations.csv'
    log_file = r'C:\Users\asus\Desktop\2022 aws\model-v7-training_job_ni5gD3LsRRaAUgDr7ZYmVg_logs\2e0f27ff-0f90-4ec1-91eb-ae764b38097f\sim-trace\training\training-simtrace\all-iterations.csv'

    waypoints, waypoints_inn, waypoints_out = get_waypoints(track_file)
    track_width = distance_of_2points(waypoints_inn[0], waypoints_out[0])

    log_parmas = read_csv_file(log_file, episode_num=-1)

    params = {}
    for i in range(len(log_parmas['episode'])):
        params['waypoints'] = waypoints
        params['all_wheels_on_track'] = log_parmas['all_wheels_on_track'][i]
        params['x'] = log_parmas['x'][i]
        params['y'] = log_parmas['y'][i]
        params['steps'] = log_parmas['steps'][i]
        params['heading'] = log_parmas['yam'][i]
        params['steering_angle'] = log_parmas['steer'][i]
        params['speed'] = log_parmas['throttle'][i]
        params['track_width'] = track_width
        params['is_left_of_center'] = is_left_of_center(waypoints_inn, waypoints_out, [log_parmas['x'][i], log_parmas['y'][i]], log_parmas['closest_waypoint'][i])
        params['progress'] = log_parmas['progress'][i]

        closest_waypoints = get_closest_waypoints(waypoints, [log_parmas['x'][i], log_parmas['y'][i]])
        params['closest_waypoints'] = closest_waypoints

        params['distance_from_center'] = distance_to_racing_line(waypoints[closest_waypoints[0]], waypoints[closest_waypoints[1]], [log_parmas['x'][i], log_parmas['y'][i]])
        params['reward'] = log_parmas['reward'][i]
        params['episode'] = log_parmas['episode'][i]

        params['is_offtrack'] = False
        if log_parmas['episode_status'] == 'off_track':
            params['is_offtrack'] = True

        if params['steps']:
            reward = reward_function(params)
            if abs(reward - params['reward']) > 0.5 and params['steps'] != 1:
                print('{}th episode {}th step -> new reward is {} and old reward is {}'.format(params['episode'], params['steps'], reward, params['reward']))
