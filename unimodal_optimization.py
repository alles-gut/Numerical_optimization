#####################################################
# Homework #2 Numerical Optimization, 2021          #
# Author Hyunjun Jung                               #
# vaild option for method argument:                 #
#  'fibonacci', 'golden'                            #
#####################################################

import math
import random

global TOL, GRAPH, SEEK
TOL = 1e-5
GRAPH = []
SEEK = []

def functions(opt, x):
    # it returns function evaluation f(x)
    assert opt <= 5, 'Not implemented'
    if opt == 1:
        return (x-4)**2-10
    elif opt == 2:
        if x < 4:
            return -4*x+6
        elif x >= 4:
            return 4*x-26
    elif opt == 3:
        if x < 4:
            return (x-4)**2-10
        elif x >= 4:
            return (x-4)**2-14
    elif opt == 4:
        if x < -3 or x >= 11:
            return 39
        else:
            return (x-4)**2-10

def get_fibonacci(N):
    # N - length of fibonacci sequence
    if N == 1:
        return [1]
    elif N == 2:
        return [1, 1]
    else:
        sequence = [1, 1]
        k = 2
        while k < N:
            sequence.append(sequence[-1]+sequence[-2])
            k += 1
        return sequence

def fibonacci(opt, N, a, b):
    # N - maximum number of iteration
    # a, b - initial points
    # opt - function opt
    global GRAPH
    F = get_fibonacci(N)

    # record
    f_a = functions(opt, a)
    f_b = functions(opt, b)
    GRAPH.append((a, f_a))
    GRAPH.append((b, f_b))

    rep = None
    for i in range(N):
        ind = N-(i+1)

        # interior points
        x1 = a * F[ind-1]/F[ind] + b * F[ind-2]/F[ind]
        x2 = a * F[ind-2]/F[ind] + b * F[ind-1]/F[ind]

        # function evaluation
        f_a = functions(opt, a)
        f_b = functions(opt, b)
        if rep == 'x1':
            f_x1 = mem
        else:
            f_x1 = functions(opt, x1)
        if rep == 'x2':
            f_x2 = mem
        else:
            f_x2 = functions(opt, x2)

        # elimination step
        if abs(x1 - x2) < TOL:
            return (x2+x1)/2, i+1, abs(x2-x1)
        elif f_x1 > f_x2:
            a = x1
            rep = 'x1'
            mem = f_x2
            GRAPH.append((a, f_x1))
        elif f_x1 < f_x2:
            b = x2
            rep = 'x2'
            mem = f_x1
            GRAPH.append((b, f_x2))

    return (x2+x1)/2, i+1, abs(x2-x1)

def golden_section(opt, N, a, b):
    # N - maximum number of iteration
    # a, b - initial points
    # opt - function opt
    global GRAPH
    golden = (-1 + math.sqrt(5)) / 2

    # record
    f_a = functions(opt, a)
    f_b = functions(opt, b)
    GRAPH.append((a, f_a))
    GRAPH.append((b, f_b))

    rep = ''
    for i in range(N):

        # interior points
        x1 = a * golden + b * (1-golden)
        x2 = a * (1-golden) + b * golden

        # function evaluation
        f_a = functions(opt, a)
        f_b = functions(opt, b)
        if rep == 'x1':
            f_x1 = mem
        else:
            f_x1 = functions(opt, x1)
        if rep == 'x2':
            f_x2 = mem
        else:
            f_x2 = functions(opt, x2)

        # elimination step
        if abs(x1 - x2) < TOL:
            return (x2+x1)/2, i+1, abs(x2-x1)
        elif f_x1 > f_x2:
            a = x1
            rep = 'x1'
            mem = f_x2
            GRAPH.append((a, f_x1))
        elif f_x1 < f_x2:
            b = x2
            rep = 'x2'
            mem = f_x1
            GRAPH.append((b, f_x2))

    return (x2+x1)/2, i+1, abs(x2-x1)

def seek_bound(opt, d0, x0, K):
    # d0 - step size
    # x0 - initial point
    # K - maximum iteration
    global SEEK

    fl = functions(opt, x0-d0)
    f0 = functions(opt, x0)
    fr = functions(opt, x0+d0)
    SEEK.append((x0, f0))
    SEEK.append((x0-d0, fl))
    SEEK.append((x0+d0, fr))

    if fl >= f0 and f0 >= fr:
        d = d0
        x1_ = x0-d0
        x1 = x0+d0
    elif fl <= f0 and f0 <= fr:
        d = -d0
        x1_ = x0+d0
        x1 = x0-d0
    elif fl >= f0 and f0 <= fr:
        return x0-d0, x0+d0
    else:
        print('?')

    xold = x1
    xold_ = x1_
    for i in range(K):
        xnew = x1+2**(i+1)*d
        fnew = functions(opt, xnew)
        fold = functions(opt, xold)
        fold_ = functions(opt, xold_)
        SEEK.append((xnew, fnew))

        if fnew >= fold and fold <= fold_:
            return max(xnew, xold_), min(xnew, xold_)
        elif fnew >= fold and d > 0:
            return xold, xnew
        elif fnew >= fold and d < 0:
            return xnew, xold
        else:
            xold_ = xold
            xold = xnew

def main(args):
    x0 = random.randint(args.plot_range[0], args.plot_range[1])
    print('Initial point for boundary seeking :', x0)

    a, b = seek_bound(args.function_opt, args.d, x0, args.k)
    print('Initial bound : [',a,',',b,']')

    if args.method == 'fibonacci':
        point, itr, fitv = fibonacci(args.function_opt, args.n, a, b)
    elif args.method == 'golden':
        point, itr, fitv = golden_section(args.function_opt, args.n, a, b)

    print('Output -> point :', point, 'itr :', itr, 'final_interval :', fitv)
    if itr == args.n:
        print("Dosen't converged")

if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument(
        '--function-opt', type=int, default='1',
        help='choosing built-in functions')
    parser.add_argument(
        '--n', type=int, default='30',
        help='maximum iteration number of elimination step')
    parser.add_argument(
        '--k', type=int, default='30',
        help='maximum iteration number of bound seeking step')
    parser.add_argument(
        '--d', type=int, default='1',
        help='step size of bound seeking')
    parser.add_argument(
        '--method', type=str, default='fibonacci',
        help='choosing methods')
    parser.add_argument(
        '--plot-range', nargs='+', type=float, default=[-10.0, 18.0],
        help='ploting range in x-axis')
    args = parser.parse_args()

    # main function
    main(args)

    #Visualization
    import matplotlib.pyplot as plt
    import numpy as np
    x = np.arange(args.plot_range[0], args.plot_range[1], 0.1)
    y = [(functions(args.function_opt, x_),0) for x_ in x]
    plt.plot(x, y)

    init_x = [GRAPH[0][0], GRAPH[1][0]]
    init_y = [GRAPH[0][1], GRAPH[1][1]]
    plt.annotate('initial', xy=(GRAPH[0][0],GRAPH[0][1]))
    plt.annotate('initial', xy=(GRAPH[1][0],GRAPH[1][1]))
    plt.plot(init_x, init_y, 'bo')

    op_x = []
    op_y = []
    for i in range(len(GRAPH)-2):
        op_x.append(GRAPH[i+2][0])
        op_y.append(GRAPH[i+2][1])
        plt.annotate(str(i+1), xy=(GRAPH[i+2][0],GRAPH[i+2][1]))
    plt.plot(op_x, op_y, 'ro')

    plt.savefig(args.method+'_opt'+str(args.function_opt)+'.png')

    plt.clf()

    x = np.arange(args.plot_range[0], args.plot_range[1], 0.1)
    y = [(functions(args.function_opt, x_),0) for x_ in x]
    plt.plot(x, y)

    init_x = [SEEK[0][0], SEEK[1][0]]
    init_y = [SEEK[0][1], SEEK[1][1]]
    plt.annotate('initial', xy=(SEEK[0][0],SEEK[0][1]))
    plt.plot(init_x, init_y, 'go')

    op_x = []
    op_y = []
    for i in range(len(SEEK)-1):
        op_x.append(SEEK[i+1][0])
        op_y.append(SEEK[i+1][1])
        plt.annotate(str(i), xy=(SEEK[i+1][0],SEEK[i+1][1]))
    plt.plot(op_x, op_y, 'bo')

    plt.savefig(args.method+'_opt'+str(args.function_opt)+'_seek'+'.png')
