#####################################################
# Homework #1 Numerical Optimization, 2021          #
# Author Hyunjun Jung                               #
# vaild option for method argument:                 #
#  'bisection', 'newton', 'secant', 'regula_falsi'  #
# built-in functions:                               #
#  option 1 -> y = (x-4)^2-10                       #
#  option 2 -> y = |(x-4)^2-10|                     #
#  option 3 -> y = (x+4)^2-10, -1, (x-4)^2-10       #
#  option 5 -> y = 10x*exp(-x^2)-1                  #
#  option 6 -> y = -5x*exp(-x)+1                    #
#####################################################

import math

global TOL, ITR, GRAPH
TOL = 1e-4
LIM = 1e+10
ITR = 1000
GRAPH = []

def functions(opt, x):
    # it returns tuple of (f(x), f'(x))
    assert opt <= 5, 'Not implemented'

    if opt == 1:
        return (x-4)**2-10, 2*(x-4)
    elif opt == 2: # bracketing method 사용 불가
        if x < 1 or x >= 7:
            return abs((x-4)**2-9), 2*(x-4)
        elif x >= 1 and x < 7:
            return abs((x-4)**2-9), -2*(x-4)
    elif opt == 3:
        if x < -1:
            return (x+4)**2-10, 2*(x+4)
        elif x >= -1 and x < 1:
            return -1, 0
        elif x >= 1:
            return (x-4)**2-10, 2*(x-4)
    elif opt == 4: # newton 그냥 diverge함
        return 10*x*math.exp(-x**2)-1, \
               10*math.exp(-x**2)-20*(x**2)*math.exp(-x**2)
    elif opt == 5:
        return -5*x*math.exp(-x)+1, 5*(x-1)*math.exp(-x)


def bisection(function_opt, interval):
    global GRAPH
    a, b = interval
    if b < a:
        print('Wrong interval...')
        return -1, None

    f_a, _ = functions(function_opt, a)
    f_b, _ = functions(function_opt, b)
    if f_a * f_b >= 0:
        print('Guess another initial interval...')
        return -1, None

    mid = (a+b)/2
    f_mid, _ = functions(function_opt, mid)
    GRAPH.append((mid, f_mid))

    if f_mid == 0 or abs(b-a) < TOL:
        print('Found the answer.')
        return 1, mid

    if f_mid * f_a < 0:
        b = mid
    else:
        a = mid

    return 0, (a, b)

def newton(function_opt, initial_value):
    global GRAPH
    f_init, d_f_init = functions(function_opt, initial_value)

    next_value = initial_value - f_init/(d_f_init+1e-8)
    f_next, _ = functions(function_opt, next_value)
    GRAPH.append((next_value, f_next))

    if f_next == 0 or abs(f_next) < TOL:
        print('Found the answer.')
        return 1, next_value

    if abs(f_next) > LIM:
        print('Diverged.')
        return -2, next_value

    return 0, next_value

def secant(function_opt, initial_values):
    global GRAPH
    x1, x2 = initial_values
    f_x1, _ = functions(function_opt, x1)
    f_x2, _ = functions(function_opt, x2)

    next_value = x2 - f_x2*(x2-x1)/(f_x2-f_x1+1e-8)
    f_next, _ = functions(function_opt, next_value)
    GRAPH.append((next_value, f_next))

    if f_next == 0 or abs(f_next) < TOL:
        print('Found the answer.')
        return 1, next_value

    if abs(f_next) > LIM:
        print('Diverged.')
        return -2, next_value

    if abs(x1-next_value) <= abs(x2-next_value):
        interpolation_pair = x1
    else:
        interpolation_pair = x2

    return 0, [next_value, interpolation_pair]

def regula_falsi(function_opt, interval):
    global GRAPH
    a, b = interval
    if b < a:
        print('Wrong interval...')
        return -1, None

    f_a, _ = functions(function_opt, a)
    f_b, _ = functions(function_opt, b)
    if f_a * f_b >= 0:
        print('Guess another initial interval...')
        return -1, None

    next = b - f_b*(b-a)/(f_b-f_a+1e-8)
    f_next, _ = functions(function_opt, next)
    GRAPH.append((next, f_next))

    if f_next == 0 or abs(b-a) < TOL:
        print('Found the answer.')
        return 1, next

    if f_next * f_a < 0:
        b = next
    else:
        a = next

    return 0, (a, b)

def main(function_opt, method, initial_value):
    print(method, 'method -', ITR, 'iteration,', TOL, 'threshold')
    print( 'Initial value -', initial_value)
    if method == 'bisection':
        assert len(initial_value) == 2, 'Wrong shape of input value'
        interval = initial_value
        for i in range(ITR):
            status, value = bisection(function_opt, interval)
            if status == 0:
                interval = value
            elif status == 1:
                print('Iteration terminated.')
                return value, i
            elif status == -1:
                return None, i

    elif method == 'newton':
        assert len(initial_value) == 1, 'Wrong shape of input value'
        point = initial_value[0]
        for i in range(ITR):
            status, value = newton(function_opt, point)
            if status == 0:
                point = value
            elif status == 1:
                print('Iteration terminated.')
                return value, i
            elif status == -2:
                return None, i

    elif method == 'secant':
        assert len(initial_value) == 2, 'Wrong shape of input value'
        points = initial_value
        for i in range(ITR):
            status, value = secant(function_opt, points)
            if status == 0:
                points = value
            elif status == 1:
                print('Iteration terminated.')
                return value, i
            elif status == -2:
                return None, i

    elif method == 'regula_falsi':
        assert len(initial_value) == 2, 'Wrong shape of input value'
        interval = initial_value
        for i in range(ITR):
            status, value = regula_falsi(function_opt, interval)
            if status == 0:
                interval = value
            elif status == 1:
                print('Iteration terminated.')
                return value, i
            elif status == -1:
                return None, i

    else:
        print('Unspecified method name...')

    print('Doesn\'t converge, current value is', value)
    return None, i


if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument(
        '--function-opt', type=int, default='1',
        help='choosing built-in functions')
    parser.add_argument(
        '--method', type=str, default='bisection',
        help='choosing root finding methods')
    parser.add_argument(
        '--initial-value', nargs='+', type=float, default=[1.0, 10.0],
        help='initial value of root finding methods, number of required \
        initial value can be differ due to method')
    parser.add_argument(
        '--plot-range', nargs='+', type=float, default=[-10.0, 10.0],
        help='ploting range in x-axis')
    args = parser.parse_args()

    #Root finding
    result, i = main(args.function_opt, args.method, args.initial_value)
    if result != None:
        print('Answer is', result, 'by iteration', i)

    #Visualization
    import matplotlib.pyplot as plt
    import numpy as np
    x = np.arange(args.plot_range[0], args.plot_range[1], 0.1)
    y = [(functions(args.function_opt, x_)[0],0) for x_ in x]
    plt.plot(x, y)

    init_x = []
    init_y = []
    for elem in args.initial_value:
        init_x.append(elem)
        init_y.append(functions(args.function_opt, elem)[0])
        plt.annotate(str('initial'),
          xy=(elem,functions(args.function_opt, elem)[0]))
    plt.plot(init_x, init_y, 'bo')

    op_x = []
    op_y = []
    for i in range(len(GRAPH)):
        op_x.append(GRAPH[i][0])
        op_y.append(GRAPH[i][1])
        plt.annotate(str(i), xy=(GRAPH[i][0],GRAPH[i][1]))
    plt.plot(op_x, op_y, 'ro')

    plt.savefig(args.method+'_opt'+str(args.function_opt)+'.png')
