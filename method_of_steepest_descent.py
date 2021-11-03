#####################################################
# Homework #4 Numerical Optimization, 2021          #
# Author Hyunjun Jung                               #
# vaild option for method argument:                 #
#  'steepest', 'newton' , 'quasi_SR1', 'quasi_BFGS' #
#####################################################

import math
import random
import numpy as np

global ITR, E1, E2, E3, E4
global GRAPH
global Init_Alpha
ITR = 100
E1 = 10e-6
E2 = 10e-6
E3 = 10e-6
E4 = 10e-6
GRAPH = []


def functions(opt, x, y):
    # it returns function evaluation f(x,y), grad{f(x,y)}, hessian{f(x,y)}
    # f(x,y) = single scalar value
    # grad{f(x,y)} = [df/dx, df/dy]
    # hessian{f(x,y)} = [[df/dxdx, df/dxdy],[df/dydx, df/dydy]]
    x = float(x)
    y = float(y)
    assert opt <= 3, 'Not implemented'
    if opt == 1:
        f = (x+2*y-6)**2+(2*x+y-6)**2
        grad = np.array([2*(x+2*y-6)+2*2*(2*x+y-6), 2*2*(x+2*y-6)+2*(2*x+y-6)])
        hessian = np.array([[2+2*2*2, 2*2+2*2],
                            [2*2+2*2, 2*2*2+2]])
    elif opt == 2:
        f = 50*(y-x**2)**2+(1-x)**2
        grad = np.array([4*(x**3)-4*x*y+2*x-2, 100*y-2*(x**2)])
        hessian = np.array([[4*3*(x**2)-4*y+2, -4*x],
                            [-2*2*x, 100]])
    elif opt == 3:
        f = (1.5-x+x*y)**2+(2.25-x+x*(y**2))**2+(2.625-x+x*(y**3))**2
        grad = np.array([2*(1.5-x+x*y)*(y-1)+2*(2.25-x+x*(y**2))*(y**2-1)+\
                                               2*(2.625-x+x*(y**3))*(y**3-1),\
                         2*(1.5-x+x*y)*x+2*(2.25-x+x*(y**2))*2*x*y+\
                                              2*(2.625-x+x*(y**3))*3*x*(y**2)])
        hessian = np.array([[2*(y-1)**2+2*(y**2-1)**2+2*(y**3-1)**2,\
                            2*((1.5-x+x*y)+x*(y-1))+2*((2.25-x+x*(y**2))*2*y+\
                            (2*x*y)*(y**2-1))+2*((2.625-x+x*(y**3))*3*(y**2)+\
                            (3*x*(y**2))*(y**3-1))],
                           [2*((1.5-x+x*y) + (y-1)*x)+2*((2.25-x+x*(y**2))*2*y+\
                            ((y**2)-1)*2*x*y)+2*((2.625-x+x*(y**3))*3*(y**2)+\
                            ((y**3)-1)*3*x*(y**2)),\
                            2*(x**2)+2*((2.25-x+x*(y**2))*2*x + (2*x*y)*2*x*y)+\
                            2*((2.625-x+x*(y**3))*3*2*x*y+\
                            (2*x*(y**2))*3*x*(y**2))]])
    return f, grad, hessian

def termination_test(
        opt, X_k=None, X_k_=None,
        grad_fX_k=None, fX_k=None,
        fX_k_=None, Pk=None):
    # X_k = current X, X_k_ = former X
    if opt == 1:
        if X_k.any() == None or X_k_.any() == None:
            return False
        elif np.linalg.norm(X_k - X_k_, 1) < E1:
            print('Terminated from criterion option 1')
            return True;
        else:
            return False
    elif opt == 2:
        if X_k == None.any() or X_k_ == None.any():
            return False
        elif np.linalg.norm(X_k - X_k_, 1) / np.linalg.norm(X_k, 1) < E2:
            print('Terminated from criterion option 2')
            return True;
        else:
            return False
    elif opt == 3:
        if grad_fX_k == None.any():
            return False
        elif np.linalg.norm(grad_fX_k, 1) < E3:
            print('Terminated from criterion option 3')
            return True;
        else:
            return False
    elif opt == 4:
        if fX_k == None.any() or fX_k_ == None.any():
            return False
        elif np.linalg.norm(fX_k-fX_k_, 1) / np.linalg.norm(fX_k, 1) < E4:
            print('Terminated from criterion option 4')
            return True;
        else:
            return False
    elif opt == 5:
        if Pk == None.any() or grad_fX_k == None.any():
            return False
        elif np.dot(Pk,grad_fX_k) >= 0:
            print('Terminated from criterion option 5')
            return True;
        else:
            return False
    else:
        if k > ITR:
            print('Terminated from criterion option 6')
            return True;
        else:
            return False

def find_step_length(fopt, X, P):
    # Returns optimal step lenth (using backtracking line search)
    # X = current point, P = current direction
    # user defined variable, rho, c within (0,1)
    global Init_Alpha
    max_itr = 10
    rho = 0.8
    c = 0.5
    alpha = Init_Alpha
    next_func_eval,_,_ = functions(fopt, X[0]+alpha*P[0], X[1]+alpha*P[1])
    current_func_eval,current_grad,_ = functions(fopt, X[0], X[1])
    P = np.array(P)
    itr = 0
    while(next_func_eval>alpha*c*np.dot(current_grad,P)+current_func_eval):
        alpha = alpha*rho
        next_func_eval,_,_ = functions(fopt, X[0]+alpha*P[0], X[1]+alpha*P[1])
        itr += 1
        if itr > max_itr:
            print('Unoptimal step length')
            break
    return alpha

def steepest_descent(fopt, copt, x_init=1.2, y_init=1.2):
    # fopt - option for function choosing
    # copt - option for criteria choosing
    print('Operating method of steepest descent')

    # initial values
    X = np.array([x_init, y_init])
    print('With initial point :',X)
    GRAPH.append(X)

    for itr in range(ITR):

        _,grad_X,_ = functions(fopt, X[0], X[1])
        alpha = find_step_length(fopt, X, -grad_X)
        X_ = X.copy()
        X = X-alpha*grad_X
        GRAPH.append(X)

        # criterion check
        f_X_,grad_X,_ = functions(fopt, X_[0], X_[1])
        f_X,_,_ = functions(fopt, X[0], X[1])
        if termination_test(copt, X_k=X, X_k_=X_,
                            grad_fX_k=grad_X, fX_k=f_X,
                            fX_k_=f_X_, Pk=-grad_X):
            print('Converged')
            print('Result :',X, ', Itr : ', itr+1)
            GRAPH.append(X)
            return X, itr+1

    print('Terminated before criterion converge')
    print('Result :',X, 'Itr : ', itr+1)
    GRAPH.append(X)
    return X, ITR

def newton(fopt, copt, x_init=1.2, y_init=1.2):
    # fopt - option for function choosing
    # copt - option for criteria choosing
    print('Operating Newton\'s method')

    # initial values
    X = np.array([x_init, y_init])
    print('With initial point :',X)
    GRAPH.append(X)

    for itr in range(ITR):

        _,grad_X,hessian_X = functions(fopt, X[0], X[1])
        hessian_inv = np.linalg.inv(hessian_X)
        X_ = X.copy()
        P = -np.matmul(hessian_inv,np.array([grad_X]).T).T[0]
        X = X+P
        GRAPH.append(X)

        # criterion check
        f_X_,grad_X,_ = functions(fopt, X_[0], X_[1])
        f_X,_,_ = functions(fopt, X[0], X[1])
        if termination_test(copt, X_k=X, X_k_=X_,
                            grad_fX_k=grad_X, fX_k=f_X,
                            fX_k_=f_X_, Pk=P):
            print('Converged')
            print('Result :',X, ', Itr : ', itr+1)
            GRAPH.append(X)
            return X, itr+1

    print('Terminated before criterion converge')
    print('Result :',X, 'Itr : ', itr+1)
    GRAPH.append(X)
    return X, ITR

def quasi_newton_SR1(fopt, copt, x_init=1.2, y_init=1.2):
    # fopt - option for function choosing
    # copt - option for criteria choosing
    print('Operating Quasi-Newton\'s method : SR1')

    # initial values
    r = 0.5
    tau = 0.1
    X = np.array([x_init, y_init])
    H = np.array([[1.0,0.0],[0.0,1.0]])*tau
    print('With initial point :',X, 'inital inverse B :', H)
    GRAPH.append(X)

    for itr in range(ITR):

        _,grad_X,_ = functions(fopt, X[0], X[1])
        P = -np.matmul(H,np.array([grad_X]).T).T[0]
        P = P/abs(P)
        alpha = find_step_length(fopt, X, P)
        _,grad_X_1,_ = functions(fopt, X[0]+alpha*P[0], X[1]+alpha*P[1])

        y = grad_X_1 - grad_X
        s = alpha*P
        if np.linalg.norm(s,1) < E1:
            print('Converged')
            print('Result :',X, ', Itr : ', itr+1)
            GRAPH.append(X)
            return X, itr+1

        Hy = np.matmul(H,np.array([y]).T).T[0]
        # shape of s_Hy : (2,)
        s_Hy = s-Hy
        if np.dot(s_Hy,y) >= r*np.linalg.norm(s_Hy,1)*np.linalg.norm(y,1) and\
                             r*np.linalg.norm(s_Hy,1)*np.linalg.norm(y,1) > 0:
            H = H + np.matmul(np.array([s_Hy]).T,np.array([s_Hy]))/np.dot(s_Hy,y)
        else:
            H = H

        X_ = X.copy()
        X = X-np.matmul(H,np.array([grad_X]).T).T[0]
        GRAPH.append(X)

        # criterion check
        f_X_,grad_X,_ = functions(fopt, X_[0], X_[1])
        f_X,_,_ = functions(fopt, X[0], X[1])
        if termination_test(copt, X_k=X, X_k_=X_,
                            grad_fX_k=grad_X, fX_k=f_X,
                            fX_k_=f_X_, Pk=P):
            print('Converged')
            print('Result :',X, ', Itr : ', itr+1)
            GRAPH.append(X)
            return X, itr+1

    print('Terminated before criterion converge')
    print('Result :',X, 'Itr : ', itr+1)
    GRAPH.append(X)
    return X, ITR

def quasi_newton_BFGS(fopt, copt, x_init=1.2, y_init=1.2):
    # fopt - option for function choosing
    # copt - option for criteria choosing
    print('Operating Quasi-Newton\'s method : BFGS')

    # initial values
    tau = 0.1
    X = np.array([x_init, y_init])
    H = np.array([[1.0,0.0],[0.0,1.0]])*tau
    I = np.array([[1.0,0.0],[0.0,1.0]])
    print('With initial point :',X, 'inital inverse B :', H)
    GRAPH.append(X)

    for itr in range(ITR):

        _,grad_X,_ = functions(fopt, X[0], X[1])
        P = -np.matmul(H,np.array([grad_X]).T).T[0]
        P = P/abs(P)
        alpha = find_step_length(fopt, X, P)
        _,grad_X_1,_ = functions(fopt, X[0]+alpha*P[0], X[1]+alpha*P[1])

        y = grad_X_1 - grad_X
        s = alpha*P
        rho = 1/np.dot(y,s)
        if np.linalg.norm(s,1) < E1:
            print('Converged')
            print('Result :',X, ', Itr : ', itr+1)
            GRAPH.append(X)
            return X, itr+1

        syT = np.matmul(np.array([s]).T,np.array([y]))
        ysT = np.matmul(np.array([y]).T,np.array([s]))
        I_rhosyT = I - rho*syT
        I_rhoysT = I - rho*ysT
        rhossT = rho*np.matmul(np.array([s]).T,np.array([s]))

        X_ = X.copy()
        H = np.matmul(np.matmul(I_rhosyT,H),I_rhoysT) + rhossT
        X = X-np.matmul(H,np.array([grad_X]).T).T[0]
        GRAPH.append(X)

        # criterion check
        f_X_,grad_X,_ = functions(fopt, X_[0], X_[1])
        f_X,_,_ = functions(fopt, X[0], X[1])
        if termination_test(copt, X_k=X, X_k_=X_,
                            grad_fX_k=grad_X, fX_k=f_X,
                            fX_k_=f_X_, Pk=P):
            print('Converged')
            print('Result :',X, ', Itr : ', itr+1)
            GRAPH.append(X)
            return X, itr+1

    print('Terminated before criterion converge')
    print('Result :',X, 'Itr : ', itr+1)
    GRAPH.append(X)
    return X, ITR


def main(args):

    print('fopt :', args.function_opt,', method :', args.method)

    global Init_Alpha
    if args.function_opt == 1:
        Init_Alpha = 3
    elif args.function_opt == 2:
        Init_Alpha = 5
    elif args.function_opt == 3:
        Init_Alpha = 0.01

    x_init = 15
    y_init = 15
    if args.method == 'steepest':
        X, itr = steepest_descent(args.function_opt, args.criteria_opt, x_init, y_init)
    elif args.method == 'newton':
        X, itr = newton(args.function_opt, args.criteria_opt, x_init, y_init)
    elif args.method == 'quasi_SR1':
        X, itr = quasi_newton_SR1(args.function_opt, args.criteria_opt, x_init, y_init)
    elif args.method == 'quasi_BFGS':
        X, itr = quasi_newton_BFGS(args.function_opt, args.criteria_opt, x_init, y_init)
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
            '--criteria-opt', type=int, default=1,
            help='choosing criteria')
    parser.add_argument(
        '--method', type=str, default='steepest',
        help='choosing methods')
    parser.add_argument(
        '--plot-range', nargs='+', type=float, default=[-20.0, 20.0],
        help='ploting range in x-axis')
    args = parser.parse_args()
    # possible method : 'steepest', 'newton' , 'quasi_SR1', 'quasi_BFGS'

    # main function
    result_X = main(args)

    #Visualization
    import matplotlib.pyplot as plt
    import numpy as np

    x = np.arange(args.plot_range[0], args.plot_range[1], 0.1)
    y = np.arange(args.plot_range[0], args.plot_range[1], 0.1)
    X, Y = np.meshgrid(x, y)
    X = np.reshape(X, (-1))
    Y = np.reshape(Y, (-1))
    Z = []
    for i in range(len(X)):
        f_val,_,_ = functions(args.function_opt, X[i], Y[i])
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

    plt.savefig(args.method+'_opt'+str(args.function_opt)+'_crt'+str(args.criteria_opt)+'.png')
