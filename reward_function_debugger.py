import os
import csv
import math
import matplotlib.pyplot as plt
import numpy as np

from functions_2d import *
# from model6 import reward_function
# from model8 import reward_function
# from deep_racer import reward_function
# from model9 import reward_function
# from model10 import *
# from model11 import *
from track2019v1 import *


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
        second_closest_index = distances_no_closest.index(min(distances_no_closest))

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
    waypoints = [[0.3236568570137024, 2.6803284883499146],
                 [0.3449653461575508, 2.5307400226593018],
                 [0.3729203939437866, 2.3822485208511353],
                 [0.40683891251683235, 2.2350025177001953],
                 [0.4460252486169338, 2.0890684723854065],
                 [0.49285656260326505, 1.9454265236854553],
                 [0.550458125770092, 1.805765986442566],
                 [0.6199908927083015, 1.6716570258140564],
                 [0.7020252197980881, 1.5448175072669983],
                 [0.7991533726453781, 1.4292609691619873],
                 [0.9147003889083862, 1.33219575881958],
                 [1.0465626120567322, 1.258876621723175],
                 [1.1902250051498413, 1.2127535939216614],
                 [1.339946985244751, 1.1937379837036133],
                 [1.4910080432891846, 1.200415313243866],
                 [1.641608476638794, 1.2070926427841187],
                 [1.7913410067558289, 1.2273582220077515],
                 [1.9404860138893127, 1.251646637916565],
                 [2.089533567428589, 1.2765108942985535],
                 [2.238669514656067, 1.3008531033992767],
                 [2.38798451423645, 1.3240716457366943],
                 [2.53842556476593, 1.337261587381363],
                 [2.689424991607666, 1.333412766456604],
                 [2.8389484882354736, 1.3125049769878387],
                 [2.9825469255447388, 1.2660765051841736],
                 [3.1155110597610474, 1.1946403682231903],
                 [3.2352620363235474, 1.1026768684387207],
                 [3.3411115407943726, 0.9949667453765869],
                 [3.436005473136902, 0.8774079084396362],
                 [3.5255489349365234, 0.7556869089603424],
                 [3.622694492340088, 0.6400226801633835],
                 [3.7314956188201904, 0.5352497547864914],
                 [3.8509503602981567, 0.4428102010861039],
                 [3.9796855449676514, 0.3638005927205086],
                 [4.115742087364197, 0.2981748580932617],
                 [4.257688522338867, 0.24650338292121887],
                 [4.404093503952026, 0.2093062549829483],
                 [4.478791236877441, 0.18695326149463654],
                 [4.5534889698028564, 0.18695326149463654],
                 [4.704370498657227, 0.1796717643737793],
                 [4.855273008346558, 0.18669861555099487],
                 [5.005021572113037, 0.20660366117954254],
                 [5.152501821517944, 0.23931320011615753],
                 [5.295422077178955, 0.28815483301877975],
                 [5.432490110397339, 0.3516535572707653],
                 [5.564228057861328, 0.42563064210116863],
                 [5.6922008991241455, 0.5059686079621315],
                 [5.816339492797852, 0.5921122580766678],
                 [5.936516046524048, 0.6837043017148973],
                 [6.052911996841431, 0.7800580710172651],
                 [6.165846824645996, 0.8804492801427841],
                 [6.2757179737091064, 0.9841870367527007],
                 [6.382829904556274, 1.0907731652259827],
                 [6.487284898757935, 1.199965626001358],
                 [6.589090585708618, 1.3116328120231628],
                 [6.688275098800659, 1.4256349802017212],
                 [6.784881353378296, 1.541829526424408],
                 [6.8789684772491455, 1.6600730419158936],
                 [6.97060751914978, 1.7802234292030334],
                 [7.059880495071411, 1.9021430611610413],
                 [7.146878957748413, 2.0256965160369877],
                 [7.231692314147949, 2.1507595181465144],
                 [7.314452171325684, 2.277191996574402],
                 [7.39509916305542, 2.4049805402755737],
                 [7.473531007766724, 2.5341415405273438],
                 [7.54969048500061, 2.664654493331909],
                 [7.623512506484985, 2.796504020690918],
                 [7.6949238777160645, 2.929674983024597],
                 [7.763846158981323, 3.0641499757766724],
                 [7.830197811126709, 3.199911952018738],
                 [7.893891334533691, 3.336941957473755],
                 [7.954820871353149, 3.475222587585449],
                 [8.013047695159912, 3.614661931991577],
                 [8.068377017974854, 3.7552770376205444],
                 [8.120350360870361, 3.8971649408340454],
                 [8.16848349571228, 4.040401577949524],
                 [8.212241411209106, 4.185032606124878],
                 [8.251099824905396, 4.3310558795928955],
                 [8.284547090530396, 4.478411436080932],
                 [8.312141180038452, 4.626972436904907],
                 [8.333237648010254, 4.776589393615723],
                 [8.345539331436157, 4.927172899246216],
                 [8.346457719802856, 5.07824206352234],
                 [8.333818435668945, 5.228774547576904],
                 [8.303643941879272, 5.376732349395752],
                 [8.25075912475586, 5.5181055068969735],
                 [8.17261004447937, 5.647185564041138],
                 [8.070677995681763, 5.758450508117676],
                 [7.951415300369263, 5.851076126098633],
                 [7.8213183879852295, 5.9277729988098145],
                 [7.683374881744385, 5.989306926727295],
                 [7.540656089782715, 6.0388734340667725],
                 [7.394761562347412, 6.078141927719116],
                 [7.2466254234313965, 6.107878684997559],
                 [7.097042560577393, 6.12921404838562],
                 [6.946609020233154, 6.143393516540527],
                 [6.79572868347168, 6.151581525802612],
                 [6.644643068313599, 6.15369176864624],
                 [6.493581295013428, 6.150143146514893],
                 [6.3427414894104, 6.141233444213867],
                 [6.192314624786377, 6.126964569091797],
                 [6.042650461196899, 6.1062493324279785],
                 [5.8946311473846436, 6.075901031494141],
                 [5.748281002044678, 6.038307189941406],
                 [5.603698492050171, 5.9943931102752686],
                 [5.4612555503845215, 5.94398045539856],
                 [5.321802377700806, 5.885821342468262],
                 [5.187443017959595, 5.816747426986694],
                 [5.063535451889038, 5.73041558265686],
                 [4.9501824378967285, 5.630155086517334],
                 [4.846731901168823, 5.5198845863342285],
                 [4.754265069961548, 5.400370836257935],
                 [4.673691511154175, 5.272421836853027],
                 [4.6044135093688965, 5.138104677200317],
                 [4.544577360153198, 4.999350547790527],
                 [4.491533994674683, 4.8578386306762695],
                 [4.441592097282411, 4.715216636657717],
                 [4.389981150627137, 4.573192596435549],
                 [4.330840349197388, 4.434109210968018],
                 [4.257412075996399, 4.301963806152344],
                 [4.163705587387085, 4.1833449602127075],
                 [4.048325896263123, 4.085645437240601],
                 [3.9156869649887085, 4.013772964477539],
                 [3.773463487625122, 3.961835503578186],
                 [3.6279555559158325, 3.92071795463562],
                 [3.4803909063339233, 3.8877369165420532],
                 [3.3314545154571533, 3.8614039421081543],
                 [3.181254506111145, 3.8450316190719604],
                 [3.0300209522247314, 3.841495990753174],
                 [2.8794320821762085, 3.854126453399658],
                 [2.7314239740371704, 3.884672522544861],
                 [2.5882774591445923, 3.9332735538482666],
                 [2.4507784843444824, 3.9959983825683594],
                 [2.316756010055542, 4.065796494483948],
                 [2.182188034057617, 4.134568452835083],
                 [2.0435360074043274, 4.194759368896484],
                 [1.8996134996414185, 4.240862846374512],
                 [1.7514809966087324, 4.270630478858948],
                 [1.6008480191230774, 4.284300923347473],
                 [1.4496694803237915, 4.284500598907471],
                 [1.2989755272865295, 4.271936655044556],
                 [1.149607002735138, 4.246785402297974],
                 [1.0045660138130188, 4.204668045043945],
                 [0.8659217953681946, 4.1436344385147095],
                 [0.7383454740047455, 4.062453508377075],
                 [0.6252143532037735, 3.962201476097107],
                 [0.5294039137661457, 3.845063090324402],
                 [0.45213192608207464, 3.715085983276367],
                 [0.3926612101495266, 3.5761590003967285],
                 [0.3495015874505043, 3.431317448616028],
                 [0.3208933472633362, 3.2829004526138306],
                 [0.30512550473213196, 3.1325994729995728],
                 [0.3009575456380844, 2.981549024581909],
                 [0.3078780025243759, 2.830607533454895]]
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
    direction_diff = agent_direction - track_direction
    if direction_diff >= 0:
        is_left_of_center = True
    else:
        is_left_of_center = False
    #

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
        'waypoints': waypoints
    }

    return params


def plot_reward(training_log_dir, factor=20):
    for file_name in os.listdir(training_log_dir):
        file_name_full_path = os.path.join(training_log_dir, file_name)
        if os.path.isfile(file_name_full_path) and file_name.split('.')[1] == 'csv':
            file_name_full_path = os.path.join(training_log_dir, file_name)
            reward_image = get_image_file_name(file_name_full_path)
            reward_image = get_image_file_name(reward_image, find_string='training-simtrace', insert_string=r'\reward',
                                               create_folder_unexist=True)

            log_parmas = read_csv_file(file_name_full_path)

            track_len = log_parmas['track_len']
            reward = log_parmas['reward']
            episode = log_parmas['episode']
            steps = log_parmas['steps']
            tstamp = log_parmas['tstamp']

            x = []
            y = []
            reward_training = []
            steps_per_second = []
            reward_percentage = []

            debug_reward = 0.0
            training_reward = 0.0
            seconds_of_episode = 0.0
            steps_of_episode = 0.0
            reward_p = 0.0

            episode_num = episode[0]
            start_time = tstamp[0]
            for i in range(len(track_len)):
                if episode_num != episode[i]:
                    # save the episode number
                    x.append(episode_num)

                    # save the accumulated rewards from training
                    reward_training.append(training_reward)

                    # save the debug rewards
                    y.append(debug_reward)

                    # save the steps per second
                    steps_per_second.append(steps_of_episode / seconds_of_episode)

                    # save the reward percentage
                    reward_percentage.append(round(reward_p / steps_of_episode, 2))

                    episode_num = episode[i]
                    start_time = tstamp[i]
                    reward_p = 0.0

                training_reward += reward[i]

                # seconds of each episode
                seconds_of_episode = tstamp[i] - start_time
                steps_of_episode = steps[i]

                params = get_params(log_parmas, i)

                max_reward = reward_function_maximum(params)

                temp_reward = reward_function(params)

                debug_reward += temp_reward

                # calculate reward percentage
                reward_p += temp_reward / max_reward

            legent = []

            plt.plot(x, reward_training)
            legent.append('Training')

            # # plot steps per second
            # plt.plot(x, steps_per_second)
            # legent.append('steps/s')

            plt.plot(x, y)
            legent.append('Debug')

            reward_percentage_pos = []
            for i in range(len(reward_percentage)):
                reward_percentage_pos.append(factor * reward_percentage[i])
            plt.plot(x, reward_percentage_pos)
            legent.append('Reward %')
            for a, b, c in zip(x, reward_percentage_pos, reward_percentage):
                plt.text(a, b, c, fontsize=6)

            plt.legend(legent)
            plt.xlabel('Episode')
            plt.ylabel('Reward')
            plt.grid(True)
            plt.title(file_name.split('.')[0])
            plt.savefig(reward_image)
            plt.show()
            plt.close()


def plot_track(training_log_dir, plot_episode=-1):
    waypoints = np.array([[0.3236568570137024, 2.6803284883499146],
                          [0.3449653461575508, 2.5307400226593018],
                          [0.3729203939437866, 2.3822485208511353],
                          [0.40683891251683235, 2.2350025177001953],
                          [0.4460252486169338, 2.0890684723854065],
                          [0.49285656260326505, 1.9454265236854553],
                          [0.550458125770092, 1.805765986442566],
                          [0.6199908927083015, 1.6716570258140564],
                          [0.7020252197980881, 1.5448175072669983],
                          [0.7991533726453781, 1.4292609691619873],
                          [0.9147003889083862, 1.33219575881958],
                          [1.0465626120567322, 1.258876621723175],
                          [1.1902250051498413, 1.2127535939216614],
                          [1.339946985244751, 1.1937379837036133],
                          [1.4910080432891846, 1.200415313243866],
                          [1.641608476638794, 1.2070926427841187],
                          [1.7913410067558289, 1.2273582220077515],
                          [1.9404860138893127, 1.251646637916565],
                          [2.089533567428589, 1.2765108942985535],
                          [2.238669514656067, 1.3008531033992767],
                          [2.38798451423645, 1.3240716457366943],
                          [2.53842556476593, 1.337261587381363],
                          [2.689424991607666, 1.333412766456604],
                          [2.8389484882354736, 1.3125049769878387],
                          [2.9825469255447388, 1.2660765051841736],
                          [3.1155110597610474, 1.1946403682231903],
                          [3.2352620363235474, 1.1026768684387207],
                          [3.3411115407943726, 0.9949667453765869],
                          [3.436005473136902, 0.8774079084396362],
                          [3.5255489349365234, 0.7556869089603424],
                          [3.622694492340088, 0.6400226801633835],
                          [3.7314956188201904, 0.5352497547864914],
                          [3.8509503602981567, 0.4428102010861039],
                          [3.9796855449676514, 0.3638005927205086],
                          [4.115742087364197, 0.2981748580932617],
                          [4.257688522338867, 0.24650338292121887],
                          [4.404093503952026, 0.2093062549829483],
                          [4.478791236877441, 0.18695326149463654],
                          [4.5534889698028564, 0.18695326149463654],
                          [4.704370498657227, 0.1796717643737793],
                          [4.855273008346558, 0.18669861555099487],
                          [5.005021572113037, 0.20660366117954254],
                          [5.152501821517944, 0.23931320011615753],
                          [5.295422077178955, 0.28815483301877975],
                          [5.432490110397339, 0.3516535572707653],
                          [5.564228057861328, 0.42563064210116863],
                          [5.6922008991241455, 0.5059686079621315],
                          [5.816339492797852, 0.5921122580766678],
                          [5.936516046524048, 0.6837043017148973],
                          [6.052911996841431, 0.7800580710172651],
                          [6.165846824645996, 0.8804492801427841],
                          [6.2757179737091064, 0.9841870367527007],
                          [6.382829904556274, 1.0907731652259827],
                          [6.487284898757935, 1.199965626001358],
                          [6.589090585708618, 1.3116328120231628],
                          [6.688275098800659, 1.4256349802017212],
                          [6.784881353378296, 1.541829526424408],
                          [6.8789684772491455, 1.6600730419158936],
                          [6.97060751914978, 1.7802234292030334],
                          [7.059880495071411, 1.9021430611610413],
                          [7.146878957748413, 2.0256965160369877],
                          [7.231692314147949, 2.1507595181465144],
                          [7.314452171325684, 2.277191996574402],
                          [7.39509916305542, 2.4049805402755737],
                          [7.473531007766724, 2.5341415405273438],
                          [7.54969048500061, 2.664654493331909],
                          [7.623512506484985, 2.796504020690918],
                          [7.6949238777160645, 2.929674983024597],
                          [7.763846158981323, 3.0641499757766724],
                          [7.830197811126709, 3.199911952018738],
                          [7.893891334533691, 3.336941957473755],
                          [7.954820871353149, 3.475222587585449],
                          [8.013047695159912, 3.614661931991577],
                          [8.068377017974854, 3.7552770376205444],
                          [8.120350360870361, 3.8971649408340454],
                          [8.16848349571228, 4.040401577949524],
                          [8.212241411209106, 4.185032606124878],
                          [8.251099824905396, 4.3310558795928955],
                          [8.284547090530396, 4.478411436080932],
                          [8.312141180038452, 4.626972436904907],
                          [8.333237648010254, 4.776589393615723],
                          [8.345539331436157, 4.927172899246216],
                          [8.346457719802856, 5.07824206352234],
                          [8.333818435668945, 5.228774547576904],
                          [8.303643941879272, 5.376732349395752],
                          [8.25075912475586, 5.5181055068969735],
                          [8.17261004447937, 5.647185564041138],
                          [8.070677995681763, 5.758450508117676],
                          [7.951415300369263, 5.851076126098633],
                          [7.8213183879852295, 5.9277729988098145],
                          [7.683374881744385, 5.989306926727295],
                          [7.540656089782715, 6.0388734340667725],
                          [7.394761562347412, 6.078141927719116],
                          [7.2466254234313965, 6.107878684997559],
                          [7.097042560577393, 6.12921404838562],
                          [6.946609020233154, 6.143393516540527],
                          [6.79572868347168, 6.151581525802612],
                          [6.644643068313599, 6.15369176864624],
                          [6.493581295013428, 6.150143146514893],
                          [6.3427414894104, 6.141233444213867],
                          [6.192314624786377, 6.126964569091797],
                          [6.042650461196899, 6.1062493324279785],
                          [5.8946311473846436, 6.075901031494141],
                          [5.748281002044678, 6.038307189941406],
                          [5.603698492050171, 5.9943931102752686],
                          [5.4612555503845215, 5.94398045539856],
                          [5.321802377700806, 5.885821342468262],
                          [5.187443017959595, 5.816747426986694],
                          [5.063535451889038, 5.73041558265686],
                          [4.9501824378967285, 5.630155086517334],
                          [4.846731901168823, 5.5198845863342285],
                          [4.754265069961548, 5.400370836257935],
                          [4.673691511154175, 5.272421836853027],
                          [4.6044135093688965, 5.138104677200317],
                          [4.544577360153198, 4.999350547790527],
                          [4.491533994674683, 4.8578386306762695],
                          [4.441592097282411, 4.715216636657717],
                          [4.389981150627137, 4.573192596435549],
                          [4.330840349197388, 4.434109210968018],
                          [4.257412075996399, 4.301963806152344],
                          [4.163705587387085, 4.1833449602127075],
                          [4.048325896263123, 4.085645437240601],
                          [3.9156869649887085, 4.013772964477539],
                          [3.773463487625122, 3.961835503578186],
                          [3.6279555559158325, 3.92071795463562],
                          [3.4803909063339233, 3.8877369165420532],
                          [3.3314545154571533, 3.8614039421081543],
                          [3.181254506111145, 3.8450316190719604],
                          [3.0300209522247314, 3.841495990753174],
                          [2.8794320821762085, 3.854126453399658],
                          [2.7314239740371704, 3.884672522544861],
                          [2.5882774591445923, 3.9332735538482666],
                          [2.4507784843444824, 3.9959983825683594],
                          [2.316756010055542, 4.065796494483948],
                          [2.182188034057617, 4.134568452835083],
                          [2.0435360074043274, 4.194759368896484],
                          [1.8996134996414185, 4.240862846374512],
                          [1.7514809966087324, 4.270630478858948],
                          [1.6008480191230774, 4.284300923347473],
                          [1.4496694803237915, 4.284500598907471],
                          [1.2989755272865295, 4.271936655044556],
                          [1.149607002735138, 4.246785402297974],
                          [1.0045660138130188, 4.204668045043945],
                          [0.8659217953681946, 4.1436344385147095],
                          [0.7383454740047455, 4.062453508377075],
                          [0.6252143532037735, 3.962201476097107],
                          [0.5294039137661457, 3.845063090324402],
                          [0.45213192608207464, 3.715085983276367],
                          [0.3926612101495266, 3.5761590003967285],
                          [0.3495015874505043, 3.431317448616028],
                          [0.3208933472633362, 3.2829004526138306],
                          [0.30512550473213196, 3.1325994729995728],
                          [0.3009575456380844, 2.981549024581909],
                          [0.3078780025243759, 2.830607533454895]])

    for file_name in os.listdir(training_log_dir):
        file_name_full_path = os.path.join(training_log_dir, file_name)
        if os.path.isfile(file_name_full_path) and file_name.split('.')[1] == 'csv':
            file_name_full_path = os.path.join(training_log_dir, file_name)
            track_image = get_image_file_name(file_name_full_path)
            track_image = get_image_file_name(track_image, find_string='training-simtrace', insert_string=r'\track',
                                              create_folder_unexist=True)

            log_parmas = read_csv_file(file_name_full_path)

            episode = np.array(log_parmas['episode'])
            x = np.array(log_parmas['x'])
            y = np.array(log_parmas['y'])
            heading = np.array(log_parmas['yam'])
            speed = np.array(log_parmas['throttle'])
            steps = np.array(log_parmas['steps'])

            debug_reward = np.zeros((1, len(episode)))

            legends = []
            plot_show_save_figure = False

            fs = (40, 30)
            dpi = 240
            plt.figure(figsize=fs, dpi=dpi)

            plt.scatter(waypoints[:, 0], waypoints[:, 1], s=15, c='k')
            for a, b, c in zip(waypoints[:, 0], waypoints[:, 1], np.arange(1, len(waypoints) + 1, 1)):
                plt.text(a - 0.01, b - 0.02, c, fontsize=4)
            legends.append('waypoints')

            min_reward = 1e-3
            mininum_reward_calculated = False

            episode_num = episode[0]
            for i in range(len(episode)):
                if episode_num != episode[i] and (plot_episode == -1 or episode_num == plot_episode):
                    plot_show_save_figure = True
                    pos = np.where(episode == episode_num)
                    plt.scatter(x[pos], y[pos], s=2)
                episode_num = episode[i]
                params = get_params(log_parmas, i)
                debug_reward[0][i] = reward_function(params)
                if not mininum_reward_calculated:
                    min_reward = minimum_reward(params)
                    mininum_reward_calculated = True

            if plot_episode == -1:
                for a, b, c, d in zip(x, y, debug_reward[0], steps):
                    if c > min_reward:
                        plt.text(a, b, c, fontsize=4)
                        plt.text(a - 0.01, b - 0.01, d, fontsize=4)
            elif plot_episode > 0:
                pos = np.where(episode == plot_episode)
                for a, b, c, d in zip(x[pos], y[pos], debug_reward[0][pos], steps[pos]):
                    if c > min_reward:
                        plt.text(a, b, c, fontsize=4)
                        plt.text(a - 0.01, b - 0.01, d, fontsize=4)

            plt.xlim(0, 9)
            plt.ylim(-0.5, 8)
            plt.grid(True)
            plt.title(file_name.split('.')[0])
            plt.legend(legends)

            if plot_show_save_figure:
                plt.savefig(track_image)
                plt.show()

            plt.close()


def get_image_file_name(data_file, find_string='aws', insert_string=r'\image', create_folder_unexist=False):
    start_pos = data_file.find(find_string)
    image_file = data_file[0:start_pos + len(find_string)] + insert_string + data_file[start_pos + len(find_string):]
    image_file = image_file[:-4] + '.jpg'
    dirs = os.path.dirname(image_file)
    if not os.path.exists(dirs) and create_folder_unexist:
        os.makedirs(dirs)
    return image_file


if __name__ == '__main__':
    # training_log = os.path.dirname(__file__) + r'\aws\training-simtrace\ben-model3-clone'
    # plot_reward(training_log, factor=100)
    # training_log = os.path.dirname(__file__) + r'\aws\training-simtrace\ben-model4'
    # plot_reward(training_log)
    # training_log = os.path.dirname(__file__) + r'\aws\training-simtrace\dlcf-htc-2021-model1'
    # plot_reward(training_log)
    # training_log = os.path.dirname(__file__) + r'\aws\training-simtrace\dlcf-htc-2021-model6'
    # plot_reward(training_log)
    # training_log = os.path.dirname(__file__) + r'\aws\training-simtrace\dlcf-htc-2021-model6-clone'
    # plot_reward(training_log, factor=30)
    # training_log = os.path.dirname(__file__) + r'\aws\training-simtrace\dlcf-htc-2021-model6-clone-clone-clone'
    # plot_reward(training_log, factor=50)
    # training_log = os.path.dirname(__file__) + r'\aws\training-simtrace\model-x'
    # plot_reward(training_log)
    # training_log = os.path.dirname(__file__) + r'\aws\training-simtrace\model-y'
    # plot_reward(training_log)
    # training_log = os.path.dirname(__file__) + r'\aws\training-simtrace\model1'
    # plot_reward(training_log)
    # training_log = os.path.dirname(__file__) + r'\aws\training-simtrace\model8-clone'
    # plot_reward(training_log, factor=40)
    # training_log = os.path.dirname(__file__) + r'\aws\training-simtrace\model8-clone-clone-clone'
    # plot_reward(training_log, factor=50)
    # training_log = os.path.dirname(__file__) + r'\aws\training-simtrace\model8-clone-clone-clone-clone'
    # plot_reward(training_log, factor=100)
    # training_log = os.path.dirname(__file__) + r'\aws\training-simtrace\model9'
    # plot_reward(training_log)
    # training_log = os.path.dirname(__file__) + r'\aws\training-simtrace\speed'
    # plot_reward(training_log)
    # training_log = os.path.dirname(__file__) + r'\aws\training-simtrace\track_width'
    # plot_reward(training_log)
    # training_log = os.path.dirname(__file__) + r'\aws\training-simtrace\dlcf-htc-2021-model10'
    # training_log = os.path.dirname(__file__) + r'\aws\training-simtrace\2019\dlcf-test'
    training_log = os.path.dirname(__file__) + r'\aws\training-simtrace\2019\dlcf-test-clone'
    # plot_reward(training_log)
    plot_track(training_log)

    # training_log = os.path.dirname(__file__) + r'\aws\training-simtrace\track_width'
    # plot_track(training_log)
