import math


def reward_function(params):
    '''
    In @params object:
    {
        "all_wheels_on_track": Boolean,    # flag to indicate if the vehicle is on the track
        "x": float,                        # vehicle's x-coordinate in meters
        "y": float,                        # vehicle's y-coordinate in meters
        "distance_from_center": float,     # distance in meters from the track center
        "is_left_of_center": Boolean,      # Flag to indicate if the vehicle is on the left side to the track center or not.
        "heading": float,                  # vehicle's yaw in degrees
        "progress": float,                 # percentage of track completed
        "steps": int,                      # number steps completed
        "speed": float,                    # vehicle's speed in meters per second (m/s)
        "streering_angle": float,          # vehicle's steering angle in degrees
        "track_width": float,              # width of the track
        "waypoints": [[float, float], … ], # list of [x,y] as milestones along the track center
        "closest_waypoints": [int, int]    # indices of the two nearest waypoints.
    }
    '''

    closest_waypoints = params['closest_waypoints']

    waypoints2019 = [[0.3236568570137024, 2.6803284883499146],
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

    reward = 1e-3

    # daniel's reward part
    if 135 <= closest_waypoints[1] or closest_waypoints[1] <= 17:
        reward = reward_function_daniel(reward, params)

    # ben's reward part
    if 18 <= closest_waypoints[1] or closest_waypoints[1] <= 56:
        reward = reward_function_ben(reward, params)

    # samuel's reward part
    if 57 <= closest_waypoints[1] or closest_waypoints[1] <= 95:
        reward = reward_function_samuel(reward, params)

    # june's reward part
    if 96 <= closest_waypoints[1] or closest_waypoints[1] <= 134:
        reward = reward_function_june(reward, params)

    return float(reward)


def reward_function_minimum(params):
    closest_waypoints = params['closest_waypoints']

    minimum_reward = 1e-3

    # daniel's reward part
    if 135 <= closest_waypoints[1] or closest_waypoints[1] <= 17:
        minimum_reward = reward_function_minimum_daniel(minimum_reward, params)

    # ben's reward part
    if 18 <= closest_waypoints[1] or closest_waypoints[1] <= 56:
        minimum_reward = reward_function_minimum_ben(minimum_reward, params)

    # samuel's reward part
    if 57 <= closest_waypoints[1] or closest_waypoints[1] <= 95:
        minimum_reward = reward_function_minimum_samuel(minimum_reward, params)

    # june's reward part
    if 96 <= closest_waypoints[1] or closest_waypoints[1] <= 134:
        minimum_reward = reward_function_minimum_june(minimum_reward, params)

    return float(minimum_reward)


def reward_function_maximum(params):
    closest_waypoints = params['closest_waypoints']

    maximum_reward = 1e-3

    # daniel's reward part
    if 135 <= closest_waypoints[1] or closest_waypoints[1] <= 17:
        maximum_reward = reward_function_maximum_daniel(maximum_reward, params)

    # ben's reward part
    if 18 <= closest_waypoints[1] or closest_waypoints[1] <= 56:
        maximum_reward = reward_function_maximum_ben(maximum_reward, params)

    # samuel's reward part
    if 57 <= closest_waypoints[1] or closest_waypoints[1] <= 95:
        maximum_reward = reward_function_maximum_samuel(maximum_reward, params)

    # june's reward part
    if 96 <= closest_waypoints[1] or closest_waypoints[1] <= 134:
        maximum_reward = reward_function_maximum_june(maximum_reward, params)

    return float(maximum_reward)


def reward_function_daniel(current_reward, params):
    reward = 1e-3
    '''
    your code here
    '''
    current_reward += reward
    return float(reward)


def reward_function_minimum_daniel(current_reward, params):
    reward = 1e-3
    '''
    your code here
    '''
    current_reward += reward
    return float(reward)


def reward_function_maximum_daniel(current_reward, params):
    track_width = params['track_width']
    waypoints = params['waypoints']

    reward = 1e-3

    x = 1.0
    y = 1.2
    distance_from_center = 0.0
    is_left_of_center = True
    heading = 30
    progress = 15
    steps = 22
    speed = 4.0
    streering_angle = 25
    closest_waypoints = [13, 14]

    maximum_params = {
        'x': x,
        'y': y,
        'distance_from_center': distance_from_center,
        'is_left_of_center': is_left_of_center,
        'heading': heading,
        'progress': progress,
        'steps': steps,
        'speed': speed,
        'streering_angle': streering_angle,
        'closest_waypoints': closest_waypoints,
        'track_width': track_width,
        'waypoints': waypoints
    }

    reward = reward_function_daniel(current_reward, maximum_params)

    current_reward += reward
    return float(reward)


def reward_function_ben(current_reward, params):
    reward = 1e-3
    '''
    your code here
    '''
    current_reward += reward
    return float(reward)


def reward_function_minimum_ben(current_reward, params):
    reward = 1e-3
    '''
    your code here
    '''
    current_reward += reward
    return float(reward)


def reward_function_maximum_ben(current_reward, params):
    reward = 1e-3
    '''
    your code here
    '''
    current_reward += reward
    return float(reward)


def reward_function_samuel(current_reward, params):
    reward = 1e-3
    '''
    your code here
    '''
    current_reward += reward
    return float(reward)


def reward_function_minimum_samuel(current_reward, params):
    reward = 1e-3
    '''
    your code here
    '''
    current_reward += reward
    return float(reward)


def reward_function_maximum_samuel(current_reward, params):
    reward = 1e-3
    '''
    your code here
    '''
    current_reward += reward
    return float(reward)


def reward_function_june(current_reward, params):
    reward = 1e-3
    '''
    your code here
    '''
    current_reward += reward
    return float(reward)


def reward_function_minimum_june(current_reward, params):
    reward = 1e-3
    '''
    your code here
    '''
    current_reward += reward
    return float(reward)


def reward_function_maximum_june(current_reward, params):
    reward = 1e-3
    '''
    your code here
    '''
    current_reward += reward
    return float(reward)


#
# 2d-functions for reference
def directions_of_2points(p1, p2):
    directions = math.atan2(p2[1] - p1[1], p2[0] - p1[0])
    directions = math.degrees(directions)
    return directions


def line_2p(p1, p2):
    # return line: ax + by + c = 0
    a = 0
    b = 0
    c = 0

    if abs(p1[0] - p2[0]) < 1e-5:  # symmetrical to x ras
        a = 1
        b = 0
        c = -1 * p1[0]
    else:
        a = (p1[1] - p2[1]) / (p1[0] - p2[0])
        b = -1
        c = p1[1] - a * p1[0]

    return a, b, c


def distance_of_2points(p1, p2):
    return ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5


def vertical_point_of_point_to_line(p, a, b, c):
    # return the vertical point of a line: ax + by + c = 0

    vp_x = (b ** 2 * p[0] - a * b * p[1] - a * c) / (a ** 2 + b ** 2)
    vp_y = (a ** 2 * p[1] - a * b * p[0] - b * c) / (a ** 2 + b ** 2)

    return [vp_x, vp_y]


def vertical_line_of_point_to_line(p, a, b, c):
    # line: ax + by +c = 0
    vertical_point = vertical_point_of_point_to_line(p, a, b, c)
    va, vb, vc = line_2p(vertical_point, p)
    return va, vb, vc


def symmetrical_point_of_point_to_line(p, a, b, c):
    # line: ax + by + c = 0

    spx = 0
    spy = 0

    if a == 0 and b == 0:
        spx = p[0]
        spy = p[1]
    else:
        spx = p[0] - 2 * a * (a * p[0] + b * p[1] + c) / (a ** 2 + b ** 2)
        spy = p[1] - 2 * b * (a * p[0] + b * p[1] + c) / (a ** 2 + b ** 2)

    return [spx, spy]


def direction_in_degrees_of_line(a, b, c=0):
    # line: ax + by + c = 0
    direction = 0
    if b == 0:
        direction = math.degrees(0.5 * math.pi)
    else:
        direction = math.degrees(math.atan2(-1 * a / b, 1))

    return direction


def distance_of_point_to_2points_line(p1, p2, p3):
    la, lb, lc = line_2p(p2, p3)
    vp = vertical_point_of_point_to_line(p1, la, lb, lc)
    distance = distance_of_2points(p1, vp)
    return distance


def middle_point_of_2points(p1, p2):
    mx = 0.5 * (p1[0] + p2[0])
    my = 0.5 * (p1[1] + p2[1])
    return mx, my


def vertical_line_of_2points_through_middle_point(p1, p2):
    la, lb, lc = line_2p(p1, p2)
    mx, my = middle_point_of_2points(p1, p2)
    lv_a = -1 / la
    lv_b = -1
    lv_c = my - lv_a * mx
    return lv_a, lv_b, lv_c


def interaction_point_of_2lines(l1, l2):
    x_int = 0
    y_int = 0

    if l1[0] * l2[1] - l1[1] * l2[0] < 1e-5:
        x_int = 1e-5
        y_int = 1e-5
    else:
        x_int = (l1[1] * l2[2] - l1[2] * l2[1]) / (l1[0] * l2[1] - l1[1] * l2[0])
        y_int = (l1[2] * l2[0] - l1[0] * l2[2]) / (l1[0] * l2[1] - l1[1] * l2[0])

    return x_int, y_int