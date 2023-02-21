import math


class Reward:
    def __init__(self, verbose=False):
        self.first_racingpoint_index = None
        self.verbose = verbose

    def reward_function(self, params):

        ################## HELPER FUNCTIONS ###################

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

            return [closest_index, second_closest_index]

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
            # try-except in case a=0 (rare bug in DeepRacer)
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

            return [next_point_coords, prev_point_coords]

        def racing_direction_diff(closest_coords, second_closest_coords, car_coords, heading):

            # Calculate the direction of the center line based on the closest waypoints
            next_point, prev_point = next_prev_racing_point(closest_coords,
                                                            second_closest_coords,
                                                            car_coords,
                                                            heading)

            # Calculate the direction in radius, arctan2(dy, dx), the result is (-pi, pi) in radians
            track_direction = math.atan2(
                next_point[1] - prev_point[1], next_point[0] - prev_point[0])

            # Convert to degree
            track_direction = math.degrees(track_direction)

            # Calculate the difference between the track direction and the heading direction of the car
            direction_diff = abs(track_direction - heading)
            if direction_diff > 180:
                direction_diff = 360 - direction_diff

            return direction_diff

        # Gives back indexes that lie between start and end index of a cyclical list
        # (start index is included, end index is not)
        def indexes_cyclical(start, end, array_len):
            if end is None or start is None:
                return []

            if end < start:
                end += array_len

            return [index % array_len for index in range(start, end)]

        # Calculate how long car would take for entire lap, if it continued like it did until now
        def projected_time(first_index, closest_index, step_count, times_list):

            # Calculate how much time has passed since start
            current_actual_time = (step_count - 1) / 15

            # Calculate which indexes were already passed
            indexes_traveled = indexes_cyclical(first_index, closest_index, len(times_list))

            # Calculate how much time should have passed if car would have followed optimals
            current_expected_time = sum([times_list[i] for i in indexes_traveled])

            # Calculate how long one entire lap takes if car follows optimals
            total_expected_time = sum(times_list)

            # Calculate how long car would take for entire lap, if it continued like it did until now
            try:
                projected_time = (current_actual_time / current_expected_time) * total_expected_time
            except:
                projected_time = 9999

            return projected_time

        #################### RACING LINE ######################

        # Optimal racing line for the 2018
        # Each row: [x,y,speed,timeFromPreviousPoint]
        racing_track = [[2.89674, 0.70087, 4.0, 0.07644],
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
                        [2.59129, 0.71477, 4.0, 0.1114]]

        ################## INPUT PARAMETERS ###################

        # Read all input parameters
        all_wheels_on_track = params['all_wheels_on_track']
        x = params['x']
        y = params['y']
        distance_from_center = params['distance_from_center']
        is_left_of_center = params['is_left_of_center']
        heading = params['heading']
        progress = params['progress']
        steps = params['steps']
        speed = params['speed']
        steering_angle = params['steering_angle']
        track_width = params['track_width']
        waypoints = params['waypoints']
        closest_waypoints = params['closest_waypoints']
        is_offtrack = params['is_offtrack']

        ############### OPTIMAL X,Y,SPEED,TIME ################

        # Get closest indexes for racing line (and distances to all points on racing line)
        closest_index, second_closest_index = closest_2_racing_points_index(racing_track, [x, y])

        # Get optimal [x, y, speed, time] for closest and second closest index
        optimals = racing_track[closest_index]
        optimals_second = racing_track[second_closest_index]

        # Save first racingpoint of episode for later
        if self.verbose == True:
            self.first_racingpoint_index = 0  # this is just for testing purposes
        if steps == 1:
            self.first_racingpoint_index = closest_index

        ################ REWARD AND PUNISHMENT ################

        ## Define the default reward ##
        reward = 1

        ## Reward if car goes close to optimal racing line ##
        DISTANCE_MULTIPLE = 1
        dist = dist_to_racing_line(optimals[0:2], optimals_second[0:2], [x, y])
        distance_reward = max(1e-3, 1 - (dist / (track_width * 0.5)))
        reward += distance_reward * DISTANCE_MULTIPLE

        ## Reward if speed is close to optimal speed ##
        SPEED_DIFF_NO_REWARD = 1
        SPEED_MULTIPLE = 2
        speed_diff = abs(optimals[2] - speed)
        if speed_diff <= SPEED_DIFF_NO_REWARD:
            # we use quadratic punishment (not linear) bc we're not as confident with the optimal speed
            # so, we do not punish small deviations from optimal speed
            speed_reward = (1 - (speed_diff / (SPEED_DIFF_NO_REWARD)) ** 2) ** 2
        else:
            speed_reward = 0
        reward += speed_reward * SPEED_MULTIPLE

        # Reward if less steps
        REWARD_PER_STEP_FOR_FASTEST_TIME = 1
        STANDARD_TIME = 37
        FASTEST_TIME = 27
        times_list = [row[3] for row in racing_track]
        projected_time = projected_time(self.first_racingpoint_index, closest_index, steps, times_list)
        try:
            steps_prediction = projected_time * 15 + 1
            reward_prediction = max(1e-3, (-REWARD_PER_STEP_FOR_FASTEST_TIME * (FASTEST_TIME) /
                                           (STANDARD_TIME - FASTEST_TIME)) * (
                                            steps_prediction - (STANDARD_TIME * 15 + 1)))
            steps_reward = min(REWARD_PER_STEP_FOR_FASTEST_TIME, reward_prediction / steps_prediction)
        except:
            steps_reward = 0
        reward += steps_reward

        # Zero reward if obviously wrong direction (e.g. spin)
        direction_diff = racing_direction_diff(
            optimals[0:2], optimals_second[0:2], [x, y], heading)
        if direction_diff > 30:
            reward = 1e-3

        # Zero reward of obviously too slow
        speed_diff_zero = optimals[2] - speed
        if speed_diff_zero > 0.5:
            reward = 1e-3

        ## Incentive for finishing the lap in less steps ##
        REWARD_FOR_FASTEST_TIME = 1500  # should be adapted to track length and other rewards
        STANDARD_TIME = 37  # seconds (time that is easily done by model)
        FASTEST_TIME = 27  # seconds (best time of 1st place on the track)
        if progress == 100:
            finish_reward = max(1e-3, (-REWARD_FOR_FASTEST_TIME /
                                       (15 * (STANDARD_TIME - FASTEST_TIME))) * (steps - STANDARD_TIME * 15))
        else:
            finish_reward = 0
        reward += finish_reward

        ## Zero reward if off track ##
        if all_wheels_on_track == False:
            reward = 1e-3

        ####################### VERBOSE #######################

        if self.verbose == True:
            print("Closest index: %i" % closest_index)
            print("Distance to racing line: %f" % dist)
            print("=== Distance reward (w/out multiple): %f ===" % (distance_reward))
            print("Optimal speed: %f" % optimals[2])
            print("Speed difference: %f" % speed_diff)
            print("=== Speed reward (w/out multiple): %f ===" % speed_reward)
            print("Direction difference: %f" % direction_diff)
            print("Predicted time: %f" % projected_time)
            print("=== Steps reward: %f ===" % steps_reward)
            print("=== Finish reward: %f ===" % finish_reward)

        #################### RETURN REWARD ####################

        # Always return a float value
        return float(reward)


reward_object = Reward()  # add parameter verbose=True to get noisy output for testing


def reward_function(params):
    return reward_object.reward_function(params)
