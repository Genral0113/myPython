import math
import numpy as np


def reward_function(params):
    # set default values
    car_width = 0.2
    speed_slow = 1.5
    speed_fast = 2.5
    steering_angle_max = 30
    speed_max = 4.0

    #
    # waypoints for re:Invent 2018
    waypoints_all = np.array([[3.05973351, 0.68265541, 3.05937004, 1.06365502, 3.06009698, 0.3016558],
                     [3.2095089, 0.68313448, 3.2084589, 1.06413305, 3.21055889, 0.30213591],
                     [3.35927546, 0.68336383, 3.35915899, 1.06436396, 3.35939193, 0.30236369],
                     [3.50903499, 0.68340179, 3.50888991, 1.06440198, 3.50918007, 0.3024016],
                     [3.658795, 0.68346104, 3.65864801, 1.06446099, 3.65894198, 0.30246109],
                     [3.80855501, 0.68351701, 3.80841804, 1.06451702, 3.80869198, 0.302517],
                     [3.95831501, 0.6835691, 3.958184, 1.064569, 3.95844603, 0.30256921],
                     [4.10807562, 0.68362114, 4.1079421, 1.06462097, 4.10820913, 0.3026213],
                     [4.25783491, 0.68367411, 4.25769901, 1.06467402, 4.25797081, 0.3026742],
                     [4.40759492, 0.68372796, 4.40745783, 1.06472802, 4.40773201, 0.30272791],
                     [4.55735493, 0.68378122, 4.55721807, 1.06478095, 4.55749178, 0.30278149],
                     [4.70711446, 0.68383627, 4.70697212, 1.06483603, 4.70725679, 0.30283651],
                     [4.85687399, 0.68389387, 4.85672522, 1.06489396, 4.85702276, 0.30289379],
                     [5.00663304, 0.68395211, 5.00649023, 1.06495202, 5.00677586, 0.3029522],
                     [5.15639353, 0.68400487, 5.15626907, 1.06500494, 5.15651798, 0.3030048],
                     [5.30615449, 0.6840501, 5.30604792, 1.06505001, 5.30626106, 0.30305019],
                     [5.45591187, 0.6841173, 5.45556688, 1.065117, 5.45625687, 0.3031176],
                     [5.60564542, 0.68433665, 5.60494995, 1.06533599, 5.60634089, 0.30333731],
                     [5.7554214, 0.6842881, 5.75636387, 1.06528699, 5.75447893, 0.3032892],
                     [5.90530467, 0.68359549, 5.90788317, 1.064587, 5.90272617, 0.30260399],
                     [6.05528617, 0.68234065, 6.05638313, 1.06333899, 6.05418921, 0.30134231],
                     [6.2049551, 0.68616906, 6.18435717, 1.06661201, 6.22555304, 0.30572611],
                     [6.3540616, 0.69851732, 6.3118062, 1.07716703, 6.39631701, 0.31986761],
                     [6.50251436, 0.71880828, 6.43282795, 1.09338105, 6.57220078, 0.34423551],
                     [6.64373994, 0.76831104, 6.47744083, 1.11110198, 6.81003904, 0.42552009],
                     [6.77488899, 0.84126708, 6.57148504, 1.16342795, 6.97829294, 0.51910621],
                     [6.89846134, 0.92622706, 6.67055082, 1.23154294, 7.12637186, 0.62091118],
                     [7.01003671, 1.02576673, 6.72773123, 1.28162706, 7.29234219, 0.7699064],
                     [7.0997467, 1.14608622, 6.77856922, 1.35104001, 7.42092419, 0.94113243],
                     [7.17247367, 1.27703255, 6.82915115, 1.44223106, 7.51579618, 1.11183405],
                     [7.23044515, 1.41720402, 6.87087011, 1.543167, 7.59002018, 1.29124105],
                     [7.27241707, 1.56586701, 6.89809895, 1.63690901, 7.64673519, 1.49482501],
                     [7.28368258, 1.7152735, 6.90298605, 1.70007706, 7.66437912, 1.73046994],
                     [7.26574397, 1.86365998, 6.88924599, 1.80526495, 7.64224195, 1.92205501],
                     [7.23396015, 2.01072997, 6.86687517, 1.90870297, 7.60104513, 2.11275697],
                     [7.18420291, 2.15471053, 6.83412886, 2.00434709, 7.53427696, 2.30507398],
                     [7.11400199, 2.28710043, 6.79334211, 2.08133888, 7.43466187, 2.49286199],
                     [7.02336597, 2.40622103, 6.73705101, 2.15485501, 7.30968094, 2.65758705],
                     [6.91742635, 2.51266396, 6.66458893, 2.22764802, 7.17026377, 2.7976799],
                     [6.79807997, 2.60492301, 6.58812094, 2.28699493, 7.008039, 2.92285109],
                     [6.667202, 2.67758954, 6.50831699, 2.33130002, 6.826087, 3.02387905],
                     [6.52665448, 2.72964501, 6.42178011, 2.36336303, 6.63152885, 3.095927],
                     [6.3804915, 2.7596705, 6.33231211, 2.38172889, 6.42867088, 3.1376121],
                     [6.22979593, 2.77003849, 6.21479893, 2.38933396, 6.24479294, 3.15074301],
                     [6.07928681, 2.77336299, 6.07750082, 2.39236689, 6.08107281, 3.1543591],
                     [5.92952967, 2.77211404, 5.93383217, 2.39113808, 5.92522717, 3.15309],
                     [5.77978396, 2.77079797, 5.78272915, 2.38980889, 5.77683878, 3.15178704],
                     [5.63002753, 2.76960504, 5.6324749, 2.38861299, 5.62758017, 3.1505971],
                     [5.48030162, 2.76904845, 5.48068619, 2.38804889, 5.47991705, 3.15004802],
                     [5.33057308, 2.76845753, 5.33505821, 2.38748407, 5.32608795, 3.14943099],
                     [5.1807456, 2.76536357, 5.19198799, 2.38452911, 5.16950321, 3.14619803],
                     [5.03107166, 2.76612103, 4.9962821, 2.38671303, 5.06586123, 3.14552903],
                     [4.88236308, 2.78463197, 4.81436396, 2.40974903, 4.95036221, 3.1595149],
                     [4.7351799, 2.82126093, 4.61906385, 2.45838594, 4.85129595, 3.18413591],
                     [4.59635496, 2.87899697, 4.42014885, 2.54119205, 4.77256107, 3.21680188],
                     [4.47106433, 2.95902896, 4.23875189, 2.65704894, 4.70337677, 3.26100898],
                     [4.3589015, 3.06015801, 4.08945704, 2.79078698, 4.62834597, 3.32952905],
                     [4.25573039, 3.17013609, 3.96964693, 2.9185071, 4.54181385, 3.42176509],
                     [4.16035795, 3.28568053, 3.86142206, 3.04946399, 4.45929384, 3.52189708],
                     [4.06672752, 3.40247047, 3.77114797, 3.16206694, 4.36230707, 3.642874],
                     [3.97197258, 3.51845491, 3.67683005, 3.27751493, 4.26711512, 3.75939488],
                     [3.87735057, 3.63452744, 3.58191299, 3.39394999, 4.17278814, 3.8751049],
                     [3.78277063, 3.750646, 3.48739004, 3.50999808, 4.07815123, 3.99129391],
                     [3.68815291, 3.86673903, 3.3929069, 3.62592697, 3.98339891, 4.1075511],
                     [3.59356093, 3.98263586, 3.2982049, 3.7419579, 3.88891697, 4.22331381],
                     [3.49883151, 4.09949803, 3.2022481, 3.86033392, 3.79541492, 4.33866215],
                     [3.40355158, 4.21739841, 3.11521006, 3.96835995, 3.6918931, 4.46643686],
                     [3.294981, 4.3193295, 3.06513095, 4.01547098, 3.52483106, 4.62318802],
                     [3.1679095, 4.39861417, 2.98513603, 4.06431723, 3.35068297, 4.73291111],
                     [3.03874147, 4.46137047, 2.89281201, 4.10942507, 3.18467093, 4.81331587],
                     [2.85496902, 4.49774456, 2.83030009, 4.11807823, 2.87963796, 4.87741089],
                     [2.79785001, 4.49501848, 2.80717206, 4.11694479, 2.78852797, 4.87309217],
                     [2.63330102, 4.49766445, 2.62734699, 4.11671114, 2.63925505, 4.87861776],
                     [2.42942142, 4.49806905, 2.4363029, 4.11713123, 2.42253995, 4.87900686],
                     [2.28900695, 4.49291039, 2.30386496, 4.11219978, 2.27414894, 4.87362099],
                     [2.14442396, 4.48807716, 2.15521097, 4.10723019, 2.13363695, 4.86892414],
                     [1.99241304, 4.48396039, 2.00273108, 4.10309982, 1.982095, 4.86482096],
                     [1.84280103, 4.47987556, 1.85368502, 4.09903097, 1.83191705, 4.86072016],
                     [1.69257349, 4.47494149, 1.70667601, 4.094203, 1.67847097, 4.85567999],
                     [1.53988248, 4.46865606, 1.55709696, 4.08804512, 1.522668, 4.84926701],
                     [1.38626897, 4.45783353, 1.44331896, 4.08112907, 1.32921898, 4.83453798],
                     [1.24336702, 4.41842437, 1.38852, 4.06615782, 1.09821403, 4.77069092],
                     [1.11356041, 4.34595108, 1.33140004, 4.03337002, 0.89572078, 4.65853214],
                     [0.99650916, 4.25053489, 1.25847101, 3.97388196, 0.73454732, 4.52718782],
                     [0.89207792, 4.13622999, 1.19061995, 3.89951611, 0.5935359, 4.37294388],
                     [0.80508506, 4.00656855, 1.13912404, 3.82332301, 0.47104609, 4.18981409],
                     [0.74566485, 3.86897993, 1.10837901, 3.75236297, 0.38295069, 3.9855969],
                     [0.71414033, 3.7237035, 1.09252095, 3.679106, 0.3357597, 3.76830101],
                     [0.70724805, 3.57293749, 1.08824301, 3.57486296, 0.32625309, 3.57101202],
                     [0.71495652, 3.42344296, 1.09411705, 3.46083498, 0.335796, 3.38605094],
                     [0.73656203, 3.27569449, 1.11056006, 3.34840298, 0.362564, 3.202986],
                     [0.77206422, 3.12969255, 1.13774204, 3.23665404, 0.40638641, 3.02273107],
                     [0.81291266, 2.98436153, 1.18113601, 3.082201, 0.4446893, 2.88652205],
                     [0.8494301, 2.83848691, 1.22032499, 2.92565393, 0.47853521, 2.75131989],
                     [0.88160983, 2.6920675, 1.25479496, 2.76883793, 0.5084247, 2.61529708],
                     [0.91196066, 2.54541802, 1.28501701, 2.62281203, 0.53890431, 2.46802402],
                     [0.94235045, 2.39877355, 1.31542897, 2.47606111, 0.56927192, 2.321486],
                     [0.97273162, 2.25212896, 1.34580803, 2.329427, 0.59965521, 2.17483091],
                     [1.00311711, 2.1054846, 1.37619102, 2.18279409, 0.63004321, 2.02817512],
                     [1.03350851, 1.95883656, 1.406582, 2.03614807, 0.66043502, 1.88152504],
                     [1.06384847, 1.81221503, 1.43696702, 1.88931, 0.69072992, 1.73512006],
                     [1.09427983, 1.66554451, 1.46724105, 1.74339604, 0.7213186, 1.58769298],
                     [1.12513283, 1.51864648, 1.49790096, 1.59741795, 0.7523647, 1.43987501],
                     [1.156986, 1.37173051, 1.52731001, 1.46129096, 0.78666198, 1.28217006],
                     [1.19869101, 1.22808051, 1.55937803, 1.350824, 0.83800399, 1.10533702],
                     [1.25311616, 1.08854017, 1.60245001, 1.24061501, 0.90378231, 0.93646532],
                     [1.33942699, 0.96741793, 1.624964, 1.21966696, 1.05388999, 0.71516889],
                     [1.44010246, 0.8561053, 1.68809497, 1.145347, 1.19210994, 0.5668636],
                     [1.57205248, 0.78639145, 1.71744001, 1.13856101, 1.42666495, 0.43422189],
                     [1.71431702, 0.73858133, 1.81132805, 1.10702395, 1.61730599, 0.3701387],
                     [1.86256504, 0.70735447, 1.92832303, 1.08263695, 1.79680705, 0.33207199],
                     [2.01154596, 0.68591703, 2.05418897, 1.06452298, 1.96890295, 0.30731109],
                     [2.16086304, 0.67375647, 2.18001199, 1.05427504, 2.1417141, 0.29323789],
                     [2.31051648, 0.67087211, 2.30605602, 1.05184603, 2.31497693, 0.28989819],
                     [2.46046555, 0.67614226, 2.44544601, 1.05684602, 2.47548509, 0.2954385],
                     [2.61039507, 0.68087015, 2.60140204, 1.061764, 2.6193881, 0.29997629],
                     [2.76023853, 0.6832203, 2.75728512, 1.06420898, 2.76319194, 0.30223161],
                     [2.90999496, 0.68319255, 2.91309094, 1.06418002, 2.90689898, 0.30220509],
                     [3.05973351, 0.68265541, 3.05937004, 1.06365502, 3.06009698, 0.3016558]])

    # Read input parameters
    x = params['x']
    y = params['y']
    is_left_of_center = params['is_left_of_center']
    all_wheels_on_track = params['all_wheels_on_track']
    closest_waypoints = params['closest_waypoints']
    distance_from_center = params['distance_from_center']
    is_offtrack = params['is_offtrack']
    progress = params['progress']
    speed = params['speed']
    heading = params['heading']
    steering_angle = params['steering_angle']
    steps = params['steps']
    track_width = params['track_width']
    waypoints = params['waypoints']

    speed_ratio = speed / speed_max
    steering_ratio = steering_angle / steering_angle_max

    reward = 1e-1
    if is_offtrack:
        return reward

    closest_waypoint = closest_waypoints[0]

    #
    start_waypoint = 1
    end_waypoints = 21
    if start_waypoint <= closest_waypoint <= end_waypoints:
        car_coords = [x, y]
        moving_direction = heading + steering_angle
        target_point = end_waypoints

        if is_left_of_center:
            reward = speed_up_inn(waypoints_all, car_coords, moving_direction, speed_ratio, target_point)
        else:
            reward = speed_up_out(waypoints_all, car_coords, moving_direction, speed_ratio, target_point)

    #
    start_waypoint = 23
    end_waypoints = 33
    if start_waypoint <= closest_waypoint <= end_waypoints:
        car_coords = [x, y]
        moving_direction = heading + steering_angle
        target_point = end_waypoints

        if is_left_of_center:
            reward = slow_down_inn(waypoints_all, car_coords, moving_direction, speed_ratio, target_point)
        else:
            reward = slow_down_out(waypoints_all, car_coords, moving_direction, speed_ratio, target_point)
    #
    start_waypoint = 34
    end_waypoints = 42
    if start_waypoint <= closest_waypoint <= end_waypoints:
        car_coords = [x, y]
        moving_direction = heading + steering_angle
        target_point = end_waypoints

        if is_left_of_center:
            reward = speed_up_inn(waypoints_all, car_coords, moving_direction, speed_ratio, target_point)
        else:
            reward = speed_up_out(waypoints_all, car_coords, moving_direction, speed_ratio, target_point)

    #
    start_waypoint = 43
    end_waypoints = 50
    if start_waypoint <= closest_waypoint <= end_waypoints:
        car_coords = [x, y]
        moving_direction = heading + steering_angle
        target_point = end_waypoints

        if is_left_of_center:
            reward = speed_up_inn(waypoints_all, car_coords, moving_direction, speed_ratio, target_point)
        else:
            reward = speed_up_out(waypoints_all, car_coords, moving_direction, speed_ratio, target_point)

    #
    start_waypoint = 51
    end_waypoints = 55
    if start_waypoint <= closest_waypoint <= end_waypoints:
        car_coords = [x, y]
        moving_direction = heading + steering_angle
        target_point = end_waypoints

        if is_left_of_center:
            reward = slow_down_inn(waypoints_all, car_coords, moving_direction, speed_ratio, target_point)
        else:
            reward = slow_down_out(waypoints_all, car_coords, moving_direction, speed_ratio, target_point)

    #
    start_waypoint = 56
    end_waypoints = 66
    if start_waypoint <= closest_waypoint <= end_waypoints:
        car_coords = [x, y]
        moving_direction = heading + steering_angle
        target_point = end_waypoints

        if is_left_of_center:
            reward = speed_up_inn(waypoints_all, car_coords, moving_direction, speed_ratio, target_point)
        else:
            reward = speed_up_out(waypoints_all, car_coords, moving_direction, speed_ratio, target_point)

    #
    start_waypoint = 67
    end_waypoints = 70
    if start_waypoint <= closest_waypoint <= end_waypoints:
        car_coords = [x, y]
        moving_direction = heading + steering_angle
        target_point = end_waypoints

        if is_left_of_center:
            reward = slow_down_inn(waypoints_all, car_coords, moving_direction, speed_ratio, target_point)
        else:
            reward = slow_down_out(waypoints_all, car_coords, moving_direction, speed_ratio, target_point)

    #
    start_waypoint = 71
    end_waypoints = 80
    if start_waypoint <= closest_waypoint <= end_waypoints:
        car_coords = [x, y]
        moving_direction = heading + steering_angle
        target_point = end_waypoints

        if is_left_of_center:
            reward = speed_up_inn(waypoints_all, car_coords, moving_direction, speed_ratio, target_point)
        else:
            reward = speed_up_out(waypoints_all, car_coords, moving_direction, speed_ratio, target_point)

    #
    start_waypoint = 81
    end_waypoints = 89
    if start_waypoint <= closest_waypoint <= end_waypoints:
        car_coords = [x, y]
        moving_direction = heading + steering_angle
        target_point = end_waypoints

        if is_left_of_center:
            reward = slow_down_inn(waypoints_all, car_coords, moving_direction, speed_ratio, target_point)
        else:
            reward = slow_down_out(waypoints_all, car_coords, moving_direction, speed_ratio, target_point)

    #
    start_waypoint = 90
    end_waypoints = 104
    if start_waypoint <= closest_waypoint <= end_waypoints:
        car_coords = [x, y]
        moving_direction = heading + steering_angle
        target_point = end_waypoints

        if is_left_of_center:
            reward = speed_up_inn(waypoints_all, car_coords, moving_direction, speed_ratio, target_point)
        else:
            reward = speed_up_out(waypoints_all, car_coords, moving_direction, speed_ratio, target_point)

    #
    start_waypoint = 105
    end_waypoints = 111
    if start_waypoint <= closest_waypoint <= end_waypoints:
        car_coords = [x, y]
        moving_direction = heading + steering_angle
        target_point = end_waypoints

        if is_left_of_center:
            reward = slow_down_inn(waypoints_all, car_coords, moving_direction, speed_ratio, target_point)
        else:
            reward = slow_down_out(waypoints_all, car_coords, moving_direction, speed_ratio, target_point)

    #
    start_waypoint = 112
    end_waypoints = 118
    if start_waypoint <= closest_waypoint <= end_waypoints:
        car_coords = [x, y]
        moving_direction = heading + steering_angle
        target_point = end_waypoints

        if is_left_of_center:
            reward = speed_up_inn(waypoints_all, car_coords, moving_direction, speed_ratio, target_point)
        else:
            reward = speed_up_out(waypoints_all, car_coords, moving_direction, speed_ratio, target_point)

    return reward


def speed_up_inn(waypoints_all, car_coords, moving_direction, speed_ratio, target_point):
    # get waypoints
    waypoints_mid = waypoints_all[:, 0:2]
    waypoints_inn = waypoints_all[:, 2:4]
    waypoints_out = waypoints_all[:, 4:6]

    # calculate the directions difference
    directions_inn = directions_of_2points(car_coords, waypoints_inn[target_point])
    directions_mid = directions_of_2points(car_coords, waypoints_mid[target_point])

    if speed_ratio > 0.5 and directions_mid < moving_direction < directions_inn:
        reward = 1.0
    elif 0.25 < speed_ratio <= 0.5 and directions_mid < moving_direction < directions_inn:
        reward = 0.5
    elif speed_ratio <= 0.25 and directions_mid < moving_direction < directions_inn:
        reward = 0.2
    else:
        reward = 1e-3

    return  reward


def speed_up_out(waypoints_all, car_coords, moving_direction, speed_ratio, target_point):
    # get waypoints
    waypoints_mid = waypoints_all[:, 0:2]
    waypoints_inn = waypoints_all[:, 2:4]
    waypoints_out = waypoints_all[:, 4:6]

    # calculate the directions difference
    directions_mid = directions_of_2points(car_coords, waypoints_mid[target_point])
    directions_out = directions_of_2points(car_coords, waypoints_inn[target_point])

    if speed_ratio > 0.5 and directions_out < moving_direction < directions_mid:
        reward = 1.0
    elif 0.25 < speed_ratio <= 0.5 and directions_out < moving_direction < directions_mid:
        reward = 0.5
    elif speed_ratio <= 0.25 and directions_out < moving_direction < directions_mid:
        reward = 0.2
    else:
        reward = 1e-3

    return reward


def slow_down_inn(waypoints_all, car_coords, moving_direction, speed_ratio, target_point):
    # get waypoints
    waypoints_mid = waypoints_all[:, 0:2]
    waypoints_inn = waypoints_all[:, 2:4]
    waypoints_out = waypoints_all[:, 4:6]

    # calculate the directions difference
    directions_inn = directions_of_2points(car_coords, waypoints_inn[target_point])
    directions_mid = directions_of_2points(car_coords, waypoints_mid[target_point])

    if 0.25 < speed_ratio <= 0.5 and directions_mid < moving_direction < directions_inn:
        reward = 1.0
    elif speed_ratio > 0.5 and directions_mid < moving_direction < directions_inn:
        reward = 0.5
    elif speed_ratio <= 0.25 and directions_mid < moving_direction < directions_inn:
        reward = 0.2
    else:
        reward = 1e-3

    return reward


def slow_down_out(waypoints_all, car_coords, moving_direction, speed_ratio, target_point):
    # get waypoints
    waypoints_mid = waypoints_all[:, 0:2]
    waypoints_inn = waypoints_all[:, 2:4]
    waypoints_out = waypoints_all[:, 4:6]

    # calculate the directions difference
    directions_mid = directions_of_2points(car_coords, waypoints_mid[target_point])
    directions_out = directions_of_2points(car_coords, waypoints_inn[target_point])

    if 0.25 < speed_ratio <= 0.5 and directions_out < moving_direction < directions_mid:
        reward = 1.0
    elif speed_ratio > 0.5 and directions_out < moving_direction < directions_mid:
        reward = 0.5
    elif speed_ratio <= 0.25 and directions_out < moving_direction < directions_mid:
        reward = 0.2
    else:
        reward = 1e-3

    return reward


def directions_of_2points(p1, p2):
    directions = math.atan2(p2[1] - p1[1], p2[0] - p1[0])
    directions = math.degrees(directions)
    return directions