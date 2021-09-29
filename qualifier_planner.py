import math
import matplotlib.pyplot as plt
import numpy as np

TRACK_FILE = "ChampionshipCup2019_track.npy"

# Parameters
FUTURE_STEP = 6
MID_STEP = 4
TURN_THRESHOLD = 10  # degrees
DIST_THRESHOLD = 0.6  # metres

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
# waypoints = np.load(TRACK_FILE)
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

print("----- Parameters -----")
print("   FUTURE_STEP: %d" % (FUTURE_STEP))
print("      MID_STEP: %d" % (MID_STEP))
print("TURN_THRESHOLD: %d" % (TURN_THRESHOLD))
print("DIST_THRESHOLD: %.1f" % (DIST_THRESHOLD))
print("----------------------")

# Extract the x and y columns from the waypoints
# waypoints = waypoints[:, 2:4]

color_dict = {0: '#ff7f0e', 1: '#1f77b4', 2: '#ff460e'}
label_dict = {0: 'Fast Incentive', 1: 'Slow Incentive', 2: 'Bonus Fast Incentive'}

colours = []

TURN_THRESHOLD_FUTURE_CALCULATED = 0
DIST_THRESHOLD_FUTURE_CALCULATED = 0
TURN_THRESHOLD_MID_CALCULATED = 0
DIST_THRESHOLD_MID_CALCULATED = 0

for i in range(len(waypoints)):
    # Simulate input parameter
    closest_waypoints = [i - 1, i]

    # Determine what speed will be rewarded
    speed_colour = select_speed(waypoints, closest_waypoints, FUTURE_STEP, MID_STEP)
    colours.append(speed_colour)

    temp_diff, temp_dist = identify_corner(waypoints, closest_waypoints, FUTURE_STEP)
    if temp_diff > TURN_THRESHOLD_FUTURE_CALCULATED:
        TURN_THRESHOLD_FUTURE_CALCULATED = temp_dist
    if temp_dist > DIST_THRESHOLD_FUTURE_CALCULATED:
        DIST_THRESHOLD_FUTURE_CALCULATED = temp_dist
    temp_diff, temp_dist = identify_corner(waypoints, closest_waypoints, MID_STEP)
    if temp_diff > TURN_THRESHOLD_MID_CALCULATED:
        TURN_THRESHOLD_MID_CALCULATED = temp_dist
    if temp_dist > DIST_THRESHOLD_MID_CALCULATED:
        DIST_THRESHOLD_MID_CALCULATED = temp_dist

print('The maximum direction difference of next {} points is {}'.format(FUTURE_STEP, TURN_THRESHOLD_FUTURE_CALCULATED))
print('The maximum distance difference of next {} points is {}'.format(FUTURE_STEP, DIST_THRESHOLD_FUTURE_CALCULATED))
print('The maximum direction difference of next {} points is {}'.format(MID_STEP, TURN_THRESHOLD_MID_CALCULATED))
print('The maximum distance difference of next {} points is {}'.format(MID_STEP, DIST_THRESHOLD_MID_CALCULATED))

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
