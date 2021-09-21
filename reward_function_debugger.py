import os
import csv
import math
import matplotlib.pyplot as plt
# from model6 import reward_function
from model8 import reward_function
# from deep_racer import reward_function


def read_csv_file(file_name, episode_num=-1):
    episode = []
    steps = []
    x = []
    y = []
    yam = []
    steer = []
    throttle = []
    action = []
    reward = []
    done = []
    all_wheels_on_track = []
    progress = []
    closest_waypoint = []
    track_len = []
    tstamp = []
    episode_status = []
    pause_duration = []

    with open(file_name) as input_file:
        csv_reader = csv.reader(input_file)
        header_row = next(csv_reader)  # skip header row
        for row_data in csv_reader:
            if row_data[0] != 'episode' and (episode_num == -1 or int(row_data[0]) == episode_num):
                episode.append(int(row_data[0]))
                steps.append(float(row_data[1]))
                x.append(float(row_data[2]))
                y.append(float(row_data[3]))
                yam.append(float(row_data[4]))
                steer.append(float(row_data[5]))
                throttle.append(float(row_data[6]))
                action.append([steer, throttle])
                reward.append(float(row_data[9]))
                tmp = row_data[10]
                if tmp == 'True':
                    done.append(True)
                else:
                    done.append(False)
                tmp = row_data[11]
                if tmp == 'True':
                    all_wheels_on_track.append(True)
                else:
                    all_wheels_on_track.append(False)
                progress.append(float(row_data[12]))
                closest_waypoint.append(int(row_data[13]))
                track_len.append(float(row_data[14]))
                tstamp.append(float(row_data[15]))
                episode_status.append(row_data[16])
                pause_duration.append(float(row_data[17]))

    log_parmas = {
        'episode': episode,
        'steps': steps,
        'x': x,
        'y': y,
        'yam': yam,
        'steer': steer,
        'throttle': throttle,
        'action': action,
        'reward': reward,
        'done': done,
        'all_wheels_on_track': all_wheels_on_track,
        'progress': progress,
        'closest_waypoint': closest_waypoint,
        'track_len': track_len,
        'tstamp': tstamp,
        'episode_status': episode_status,
        'pause_duration': pause_duration
    }

    return log_parmas


def get_params(log_params, index):
    # useful functions
    def dist_2_points(x1, x2, y1, y2):
        return abs(abs(x1 - x2) ** 2 + abs(y1 - y2) ** 2) ** 0.5

    def closest_2_racing_points_index(racing_coords, car_coords):

        # Calculate all distances to racing points
        distances = []
        for i in range(len(racing_coords)):
            distance = dist_2_points(x1=racing_coords[i][0], x2=car_coords[0],
                                     y1=racing_coords[i][1], y2=car_coords[1])
            distances.append(distance)

        # Get index of the closest racing point
        closest_index = distances.index(min(distances))

        # Get index of the second closest racing point
        distances_no_closest = distances.copy()
        distances_no_closest[closest_index] = 999
        second_closest_index = distances_no_closest.index(
            min(distances_no_closest))

        return closest_index, second_closest_index

    def dist_to_racing_line(closest_coords, second_closest_coords, car_coords):

        # Calculate the distances between 2 closest racing points
        a = abs(dist_2_points(x1=closest_coords[0],
                              x2=second_closest_coords[0],
                              y1=closest_coords[1],
                              y2=second_closest_coords[1]))

        # Distances between car and closest and second closest racing point
        b = abs(dist_2_points(x1=car_coords[0],
                              x2=closest_coords[0],
                              y1=car_coords[1],
                              y2=closest_coords[1]))
        c = abs(dist_2_points(x1=car_coords[0],
                              x2=second_closest_coords[0],
                              y1=car_coords[1],
                              y2=second_closest_coords[1]))

        # Calculate distance between car and racing line (goes through 2 closest racing points)
        # try-except in case a=0
        try:
            distance = abs(-(a ** 4) + 2 * (a ** 2) * (b ** 2) + 2 * (a ** 2) * (c ** 2) -
                           (b ** 4) + 2 * (b ** 2) * (c ** 2) - (c ** 4)) ** 0.5 / (2 * a)
        except:
            distance = b

        return distance

    # Calculate which one of the closest racing points is the next one and which one the previous one
    def next_prev_racing_point(closest_coords, second_closest_coords, car_coords, heading):

        # Virtually set the car more into the heading direction
        heading_vector = [math.cos(math.radians(
            heading)), math.sin(math.radians(heading))]
        new_car_coords = [car_coords[0] + heading_vector[0],
                          car_coords[1] + heading_vector[1]]

        # Calculate distance from new car coords to 2 closest racing points
        distance_closest_coords_new = dist_2_points(x1=new_car_coords[0],
                                                    x2=closest_coords[0],
                                                    y1=new_car_coords[1],
                                                    y2=closest_coords[1])
        distance_second_closest_coords_new = dist_2_points(x1=new_car_coords[0],
                                                           x2=second_closest_coords[0],
                                                           y1=new_car_coords[1],
                                                           y2=second_closest_coords[1])

        if distance_closest_coords_new <= distance_second_closest_coords_new:
            next_point_coords = closest_coords
            prev_point_coords = second_closest_coords
        else:
            next_point_coords = second_closest_coords
            prev_point_coords = closest_coords

        return next_point_coords, prev_point_coords

    def direction_2_points(prev_point, next_point):
        return math.atan2(next_point[1] - prev_point[1], next_point[0] - prev_point[0])

    params = {
        'x': 0.0,
        'y': 0.0,
        'closest_waypoints': [],
        'distance_from_center': 0.0,
        'speed': 0.0,
        'steering_angle': 0.0,
        'heading': 0.0,
        'all_wheels_on_track': False,
        'is_offtrack': True,
        'is_left_of_center': True,
        'track_width': 0.762,
        'progress': 0.0,
        'steps': 0.0,
        'waypoints': []
    }

    x = log_params['x'][index]
    y = log_params['y'][index]
    #
    waypoints = [[3.2095088958740234, 0.6831344813108444],
                 [3.359275460243225, 0.6833638250827789],
                 [3.5090349912643433, 0.6834017932415009],
                 [3.6587949991226196, 0.6834610402584076],
                 [3.808555006980896, 0.6835170090198517],
                 [3.9583150148391724, 0.6835691034793854],
                 [4.1080756187438965, 0.6836211383342743],
                 [4.2578349113464355, 0.6836741119623184],
                 [4.407594919204712, 0.683727964758873],
                 [4.557354927062988, 0.6837812215089798],
                 [4.7071144580841064, 0.6838362663984299],
                 [4.856873989105225, 0.6838938742876053],
                 [5.006633043289185, 0.6839521080255508],
                 [5.156393527984619, 0.6840048730373383],
                 [5.306154489517212, 0.6840500980615616],
                 [5.455911874771118, 0.6841173022985458],
                 [5.605645418167114, 0.6843366473913193],
                 [5.75542140007019, 0.6842880994081497],
                 [5.905304670333862, 0.6835954934358597],
                 [6.055286169052124, 0.6823406517505646],
                 [6.204955101013184, 0.6861690580844879],
                 [6.354061603546143, 0.6985173225402832],
                 [6.502514362335205, 0.7188082784414291],
                 [6.643739938735962, 0.7683110386133194],
                 [6.77488899230957, 0.8412670791149139],
                 [6.89846134185791, 0.9262270629405975],
                 [7.0100367069244385, 1.0257667303085327],
                 [7.0997467041015625, 1.1460862159729004],
                 [7.172473669052124, 1.2770325541496277],
                 [7.230445146560669, 1.4172040224075317],
                 [7.272417068481445, 1.565867006778717],
                 [7.283682584762573, 1.7152734994888306],
                 [7.265743970870972, 1.8636599779129024],
                 [7.233960151672363, 2.010729968547821],
                 [7.1842029094696045, 2.154710531234741],
                 [7.114001989364624, 2.2871004343032837],
                 [7.0233659744262695, 2.406221032142639],
                 [6.917426347732544, 2.512663960456848],
                 [6.79807996749878, 2.604923009872436],
                 [6.6672019958496085, 2.6775895357131962],
                 [6.526654481887817, 2.729645013809204],
                 [6.380491495132446, 2.7596704959869385],
                 [6.229795932769775, 2.7700384855270386],
                 [6.079286813735961, 2.7733629941940308],
                 [5.929529666900635, 2.7721140384674072],
                 [5.7797839641571045, 2.7707979679107666],
                 [5.630027532577515, 2.769605040550232],
                 [5.48030161857605, 2.7690484523773193],
                 [5.330573081970215, 2.768457531929016],
                 [5.180745601654053, 2.765363574028015],
                 [5.031071662902832, 2.766121029853821],
                 [4.8823630809783936, 2.7846319675445557],
                 [4.735179901123047, 2.821260929107666],
                 [4.596354961395264, 2.878996968269348],
                 [4.471064329147339, 2.959028959274292],
                 [4.358901500701904, 3.0601580142974854],
                 [4.255730390548706, 3.1701360940933228],
                 [4.16035795211792, 3.2856805324554443],
                 [4.066727519035339, 3.4024704694747925],
                 [3.9719725847244263, 3.518454909324646],
                 [3.8773505687713623, 3.6345274448394775],
                 [3.7827706336975098, 3.7506459951400757],
                 [3.6881529092788696, 3.86673903465271],
                 [3.5935609340667725, 3.9826358556747437],
                 [3.4988315105438232, 4.09949803352356],
                 [3.4035515785217285, 4.217398405075073],
                 [3.294981002807617, 4.319329500198364],
                 [3.1679095029830933, 4.398614168167114],
                 [3.0387414693832397, 4.461370468139648],
                 [2.854969024658203, 4.497744560241699],
                 [2.797850012779234, 4.495018482208252],
                 [2.633301019668579, 4.497664451599121],
                 [2.4294214248657227, 4.4980690479278564],
                 [2.2890069484710693, 4.492910385131836],
                 [2.1444239616394043, 4.488077163696289],
                 [1.99241304397583, 4.483960390090942],
                 [1.842801034450531, 4.479875564575195],
                 [1.6925734877586365, 4.4749414920806885],
                 [1.539882481098175, 4.468656063079834],
                 [1.3862689733505262, 4.457833528518677],
                 [1.2433670163154602, 4.418424367904663],
                 [1.1135604083538055, 4.345951080322266],
                 [0.9965091645717638, 4.250534892082216],
                 [0.8920779228210449, 4.136229991912842],
                 [0.8050850629806519, 4.006568551063538],
                 [0.7456648498773575, 3.8689799308776855],
                 [0.7141403257846834, 3.723703503608705],
                 [0.7072480469942093, 3.572937488555908],
                 [0.714956521987915, 3.4234429597854614],
                 [0.7365620285272598, 3.275694489479065],
                 [0.7720642238855366, 3.129692554473875],
                 [0.8129126578569412, 2.9843615293502808],
                 [0.8494300991296768, 2.838486909866333],
                 [0.8816098272800446, 2.692067503929138],
                 [0.9119606614112854, 2.5454180240631104],
                 [0.942350447177887, 2.3987735509872437],
                 [0.9727316200733185, 2.2521289587020874],
                 [1.0031171143054962, 2.1054846048355103],
                 [1.0335085093975067, 1.958836555480957],
                 [1.063848465681076, 1.8122150301933289],
                 [1.0942798256874084, 1.6655445098876953],
                 [1.125132828950882, 1.518646478652954],
                 [1.1569859981536865, 1.3717305064201355],
                 [1.1986910104751587, 1.2280805110931396],
                 [1.2531161606311798, 1.0885401666164398],
                 [1.3394269943237305, 0.9674179255962372],
                 [1.440102458000183, 0.8561052978038788],
                 [1.5720524787902832, 0.7863914519548416],
                 [1.7143170237541199, 0.7385813295841217],
                 [1.862565040588379, 0.7073544710874557],
                 [2.011545956134796, 0.6859170347452164],
                 [2.1608630418777466, 0.6737564653158188],
                 [2.3105164766311646, 0.6708721071481705],
                 [2.4604655504226685, 0.6761422604322433],
                 [2.610395073890686, 0.6808701455593109],
                 [2.760238528251648, 0.6832202970981598],
                 [2.909994959831238, 0.6831925511360168],
                 [3.059733510017395, 0.6826554089784622]]
    #
    speed = log_params['throttle'][index]
    #
    heading = log_params['yam'][index]
    #
    steering_angle = log_params['steer'][index]
    #
    steps = log_params['steps'][index]
    #
    progress = log_params['progress'][index]
    #
    closest_waypoints = [0, 0]
    closest_index, second_closest_index = closest_2_racing_points_index(waypoints, [x, y])
    closest_coords = waypoints[closest_index]
    second_closest_coords = waypoints[second_closest_index]
    next_point, prev_point = next_prev_racing_point(closest_coords, second_closest_coords, [x, y], heading)
    if next_point == waypoints[closest_index]:
        closest_waypoints = [second_closest_index, closest_index]
    elif next_point == waypoints[second_closest_index]:
        closest_waypoints = [closest_index, second_closest_index]
    #
    all_wheels_on_track = log_params['all_wheels_on_track'][index]
    #
    distance_from_center = dist_to_racing_line(closest_coords, second_closest_coords, [x, y])
    #
    is_offtrack = log_params['episode_status'][index]
    if is_offtrack == 'off_track':
        is_offtrack = True
    else:
        is_offtrack = False
    #
    is_left_of_center = True
    track_direction = direction_2_points(prev_point, next_point)
    agent_direction = direction_2_points(prev_point, [x, y])
    direction_diff = track_direction - agent_direction
    if direction_diff >= 0:
        is_left_of_center = True
    else:
        is_left_of_center = False
    params['is_left_of_center'] = is_left_of_center
    #
    track_width = 0.762
    params['track_width'] = track_width

    params = {
        'x': x,
        'y': y,
        'closest_waypoints': closest_waypoints,
        'distance_from_center': distance_from_center,
        'speed': speed,
        'steering_angle': steering_angle,
        'heading': heading,
        'all_wheels_on_track': all_wheels_on_track,
        'is_offtrack': is_offtrack,
        'is_left_of_center': is_left_of_center,
        'track_width': 0.762,
        'progress': progress,
        'steps': steps,
        'waypoints': waypoints,
        'maximum_speed': 3,
        'multiply_factor': 3,
        'direction_diff_limit': 30
    }

    return params


def plot_reward(training_log):
    for file_name in os.listdir(training_log):
        if file_name.split('.')[1] == 'csv':
            file_name_full_path = os.path.join(training_log, file_name)
            image_file_name_full_path = os.path.join(training_log, file_name.split('.')[0] + '.jpg')

            log_parmas = read_csv_file(file_name_full_path)

            track_len = log_parmas['track_len']
            reward = log_parmas['reward']
            episode = log_parmas['episode']
            steps = log_parmas['steps']

            x = []
            y = []
            y_training = []

            debug_reward = 0
            training_reward = 0

            episode_num = episode[0]
            for i in range(len(track_len)):

                if episode_num != episode[i]:
                    x.append(episode_num)

                    y_training.append(training_reward)
                    y.append(debug_reward)

                    episode_num = episode[i]

                training_reward += reward[i]

                params = get_params(log_parmas, i)

                debug_reward += reward_function(params)

            legent = []

            plt.plot(x, y_training)
            legent.append('Training')

            plt.plot(x, y)
            legent.append('Debug')

            plt.legend(legent)
            plt.xlabel('Episode')
            plt.ylabel('Reward')
            plt.grid(True)
            plt.title(file_name.split('.')[0])
            plt.savefig(image_file_name_full_path)
            plt.show()


if __name__ == '__main__':
    training_log = os.path.dirname(__file__) + r'\aws\training-simtrace\dlcf-htc-2021-model6-clone\all'
    plot_reward(training_log)
    training_log = os.path.dirname(__file__) + r'\aws\training-simtrace\dlcf-htc-2021-model6\all'
    plot_reward(training_log)
    training_log = os.path.dirname(__file__) + r'\aws\training-simtrace\dlcf-htc-2021-model1\all'
    plot_reward(training_log)
    training_log = os.path.dirname(__file__) + r'\aws\training-simtrace\model1\all'
    plot_reward(training_log)

