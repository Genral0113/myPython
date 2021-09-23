import math


def line_2p(p1, p2):
    # return line: ax + by + c = 0
    a = 0
    b = 0
    c = 0

    if abs(p1[0] - p2[0]) < 1e-5:   # symmetrical to x ras
        a = 1
        b = 0
        c = -1 * p1[0]
    else:
        a = (p1[1] - p2[1])/(p1[0] - p2[0])
        b = -1
        c = p1[1] - a * p1[0]

    return a, b, c


def distance_of_2points(p1, p2):
    return ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5


def vertical_point_of_point_to_line(p, a, b, c):
    # return the vertical point of a line: ax + by + c = 0

    vp_x = (b ** 2 * p[0] - a * b * p[1] - a * c)/(a ** 2 + b ** 2)
    vp_y = (a ** 2 * p[1] - a * b * p[0] - b * c)/(a ** 2 + b ** 2)

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


def test_2d_functions():

    p1 = [1, 1]
    p2 = [2, 2]

    print('p1:[{}, {}], p2:[{}, {}]'.format(p1[0], p1[1], p2[0], p2[1]))

    a, b, c = line_2p(p1, p2)
    print('the line function is {}*x + {}*y + {} = 0'.format(a, b, c))

    line_direction = direction_in_degrees_of_line(a, b, c)
    print('the direction of line: {}*x + {}*y + {} is {} degrees'.format(a, b, c, line_direction))

    test_point = [3, 9]

    # line: ax + by +c = 0
    vertical_point = vertical_point_of_point_to_line(test_point, a, b, c)
    print('the vertical point of [{}, {}] in line: {}*x + {}*y + {} is [{}, {}]'.format(test_point[0], test_point[1], a, b, c, vertical_point[0], vertical_point[1]))

    # line: ax + by +c = 0
    symmetrical_point = symmetrical_point_of_point_to_line(test_point, a, b, c)
    print('the symmetrical point of [{}, {}] to line: {}*x + {}*y + {} is [{}, {}]'.format(test_point[0], test_point[1], a, b, c, symmetrical_point[0], symmetrical_point[1]))

    p1 = [4, 0]
    p2 = [1, 2]
    p3 = [2, 4]
    distance = distance_of_point_to_2points_line(p1, p2, p3)
    print('the distance is {}'.format(distance))
    print('square root of 8 is {}'.format(math.sqrt(8)))


if __name__ == '__main__':
    test_2d_functions()
