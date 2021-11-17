#####################################################
# Homework #5 Numerical Optimization, 2021          #
# Author Hyunjun Jung                               #
# vaild option for method argument:                 #
#  'CG', 'nonlinearCG'                              #
#####################################################

import math
import random
import numpy as np

global ITR, E
global GRAPH
global Init_Alpha
ITR = 100
E = 1e-6
GRAPH = []


def functions(opt, x, y):
    # it returns function evaluation f(x,y), grad{f(x,y)}, hessian{f(x,y)}
    # f(x,y) = single scalar value
    # grad{f(x,y)} = [df/dx, df/dy]
    # hessian{f(x,y)} = [[df/dxdx, df/dxdy],[df/dydx, df/dydy]]
    x = float(x)
    y = float(y)
    assert opt <= 3, 'Not implemented'
    if opt == 0:
        f = (x+2*y-7)**2+(2*x+y-5)**2
        grad = None
    if opt == 1:
        f = 40*(y-x**2)**2+(1-x)**2
        grad = np.array([40*2*(-2*x)*(y-x**2)-2*(1-x), 40*2*(y-x**2)])
    elif opt == 2:
        f = (1.5-x+x*y)**2+(2.25-x+x*(y**2))**2+(2.625-x+x*(y**3))**2
        grad = np.array([2*(1.5-x+x*y)*(y-1)+2*(2.25-x+x*(y**2))*(y**2-1)+\
                                               2*(2.625-x+x*(y**3))*(y**3-1),\
                         2*(1.5-x+x*y)*x+2*(2.25-x+x*(y**2))*2*x*y+\
                                              2*(2.625-x+x*(y**3))*3*x*(y**2)])
    return f, grad


def find_step_length(fopt, X, P):
    # Returns optimal step lenth (using backtracking line search)
    # X = current point, P = current direction
    # user defined variable, rho, c within (0,1)
    global Init_Alpha
    max_itr = 300
    rho = 0.8
    c = 0.5
    alpha = Init_Alpha
    next_func_eval,_ = functions(fopt, X[0]+alpha*P[0], X[1]+alpha*P[1])
    current_func_eval,current_grad = functions(fopt, X[0], X[1])
    P = np.array(P)
    itr = 0
    while(next_func_eval>alpha*c*np.dot(current_grad,P)+current_func_eval):
        alpha = alpha*rho
        next_func_eval,_ = functions(fopt, X[0]+alpha*P[0], X[1]+alpha*P[1])
        itr += 1
        if itr > max_itr:
            #print('Unoptimal step length')
            break
    return alpha

def norm(X):
    return (abs(X[0])+abs(X[1]))/2

def CG_method(A, b, x_init=1.2, y_init=1.2):
    global ITR, E, GRAPH
    # A, b for objective function, 0=Ax-b
    print('Operating CG method')

    # initial values
    X = np.array([x_init, y_init], dtype=float)
    print('With initial point :',X)
    GRAPH.append(X)

    r0 = np.matmul(A, X) - b
    p0 = -r0
    k = 0

    rk = r0
    pk = p0
    Xk = X

    for itr in range(ITR):
        if norm(rk) < E:
            print('Converged')
            print('Result :',Xk, ', Itr : ', itr+1)
            GRAPH.append(Xk)
            return X, itr+1

        rTr = np.matmul(rk.T, rk)
        pAp = np.matmul(np.matmul(pk.T, A), pk)
        ak = rTr / pAp

        # update
        Xk = Xk + ak * pk
        GRAPH.append(Xk)

        rk_ = rk + ak*np.matmul(A, pk)
        beta = np.matmul(rk_.T, rk_) / np.matmul(rk.T, rk)
        rk = rk_
        pk = -rk + beta * pk

    print('Terminated before criterion converge')
    print('Result :',Xk, 'Itr : ', itr+1)
    GRAPH.append(Xk)
    return X, ITR

def nonlinear_CG_method(fopt, x_init=1.2, y_init=1.2):
    global ITR, E, GRAPH
    # fopt - option for function choosing
    print('Operating nonlinear CG method')

    # initial values
    X = np.array([x_init, y_init], dtype=float)
    print('With initial point :',X)
    GRAPH.append(X)

    f0, grad_f0 = functions(fopt, X[0], X[1])
    p0 = -grad_f0

    Xk = X
    pk = p0
    grad_fk = grad_f0

    for itr in range(ITR):

        if norm(grad_fk) < E:
            print('Converged')
            print('Result :',Xk, ', Itr : ', itr+1)
            GRAPH.append(Xk)
            return X, itr+1

        ak = find_step_length(fopt, Xk, pk)

        # update
        Xk = Xk + ak * pk
        GRAPH.append(Xk)

        _, grad_fk_ = functions(fopt, X[0], X[1])
        beta = np.matmul(grad_fk_.T, grad_fk_) / np.matmul(grad_fk.T, grad_fk)
        pk = - grad_fk_ + beta * pk

        grad_fk = grad_fk_

    print('Terminated before criterion converge')
    print('Result :',Xk, 'Itr : ', itr+1)
    GRAPH.append(Xk)
    return X, ITR

def main(args):

    print('fopt :', args.function_opt,', method :', args.method)

    global Init_Alpha
    Init_Alpha = 3

    x_init = 15
    y_init = 15
    if args.method == 'CG':
        A = np.array([[10, 8], [8, 10]], dtype=float)
        b = np.array([34, 38], dtype=float)
        X, itr = CG_method(A, b, x_init, y_init)
    elif args.method == 'nonlinearCG':
        X, itr = nonlinear_CG_method(args.function_opt, x_init, y_init)
    else:
        X = 0
        itr = 0

    print('Output -> point :', X, 'itr :', itr)

    return X

if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument(
        '--function-opt', type=int, default=1,
        help='choosing built-in functions')
    parser.add_argument(
        '--method', type=str, default='CG',
        help='choosing methods')
    parser.add_argument(
        '--plot-range', nargs='+', type=float, default=[-17, 17],
        help='ploting range in x-axis')
    args = parser.parse_args()
    # possible method : 'CG', 'nonlinearCG'

    # main function
    result_X = main(args)

    #Visualization
    import matplotlib.pyplot as plt
    import numpy as np

    x = np.arange(args.plot_range[0], args.plot_range[1], 0.01)
    y = np.arange(args.plot_range[0], args.plot_range[1], 0.01)
    X, Y = np.meshgrid(x, y)
    X = np.reshape(X, (-1))
    Y = np.reshape(Y, (-1))
    Z = []
    for i in range(len(X)):
        f_val,_ = functions(args.function_opt, X[i], Y[i])
        Z.append(f_val)
    X = np.reshape(X, (len(x),len(y)))
    Y = np.reshape(Y, (len(x),len(y)))
    Z = np.reshape(Z, (len(x),len(y)))

    plt.contour(X,Y,Z, levels=30,linewidths=0.5)

    plt.plot(GRAPH[0][0], GRAPH[0][1],'go')
    plt.annotate('initial', xy=(GRAPH[0][0], GRAPH[0][1]))
    for i in range(len(GRAPH)-2):
        plt.plot(GRAPH[i+1][0], GRAPH[i+1][1],'bo')
        plt.annotate(str(i+1), xy=(GRAPH[i+1][0], GRAPH[i+1][1]))
    plt.plot(GRAPH[-1][0], GRAPH[-1][1],'ro')
    plt.annotate('result', xy=(GRAPH[-1][0], GRAPH[-1][1]))

    plt.savefig(args.method+'_opt'+str(args.function_opt)+'.png')
