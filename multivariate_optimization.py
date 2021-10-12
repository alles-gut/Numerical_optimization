#####################################################
# Homework #3 Numerical Optimization, 2021          #
# Author Hyunjun Jung                               #
# vaild option for method argument:                 #
#  'nelder_mead', 'powell'                          #
#####################################################

import math
import random

global ITR, E1, E2, E3, E4
global X_L, X_R, Y_L, Y_R
global GRAPH_X, GRAPH_Y
ITR = 1000
E1 = 10e-6
E2 = 10e-6
E3 = 10e-6
E4 = 10e-6
X_L = -10
X_R = 10
Y_L = -10
Y_R = 10
GRAPH = []


def functions(opt, x, y):
    # it returns function evaluation f(x,y), df(x,y)/dx, df(x,y)/dy
    assert opt <= 3, 'Not implemented'
    if opt == 1:
        return (x+2*y-6)**2 + (2*x+y-6)**2,\
               2*(x+2*y-6) + 2*2*(2*x+y-6),\
               2*2*(x+2*y-6) + 2*(2*x+y-6)
    elif opt == 2:
        return 50*(y-x**2)**2 + (1-x)**2,\
               -50*2*2*x*(y-x**2) - 2*(1-x),\
               50*2*(y-x**2)
    elif opt == 3:
        return (1.5-x+x*y)**2 + (2.25-x+x*(y**2))**2 + (2.625-x+x*(y**3))**2,\
               2*(y-1)*(1.5-x+x*y) + 2*(y**2-1)*(2.25-x+x*(y**2)) + \
               2*(y**3-1)*(2.625-x+x*(y**3)),\
               2*x*(1.5-x+x*y) + 2*(2*x*y)*(2.25-x+x*(y**2)) + \
               2*(3*(y**2)*x)*(2.625-x+x*(y**3))

def l1_norm(X):
    ans = 0
    for i in range(len(X)):
        ans += X[i]
    return ans

def termination_test(
        opt, X_k=None, X_k_=None,
        grad_fX_k=None, fX_k=None,
        fX_k_=None, Pk=None, k=None
    ):

    if opt == 1:
        if X_k == None or X_k_ == None:
            return True
        elif l1_norm([X_k[0]-X_k_[0], X_k[1]-X_k_[1]]) < E1:
            print('Terminated from criterion option 1')
            return False;
        else:
            return True
    elif opt == 2:
        if X_k == None or X_k_ == None:
            return True
        elif l1_norm([X_k[0]-X_k_[0], X_k[1]-X_k_[1]]) / l1_norm(X_k) < E2:
            print('Terminated from criterion option 2')
            return False;
        else:
            return True
    elif opt == 3:
        if grad_fX_k == None:
            return True
        elif l1_norm(grad_fX_k) < E3:
            print('Terminated from criterion option 3')
            return False;
        else:
            return True
    elif opt == 4:
        if fX_k == None or fX_k_ == None:
            return True
        elif l1_norm([fX_k-fX_k_]) < E4:
            print('Terminated from criterion option 4')
            return False;
        else:
            return True
    elif opt == 5:
        if Pk == None or grad_fX_k == None:
            return True
        elif Pk[0] * grad_fX_k[0] + Pk[1] * grad_fX_k[1] >= 0:
            print('Terminated from criterion option 5')
            return False;
        else:
            return True
    else:
        if k > ITR:
            print('Terminated from criterion option 6')
            return False;
        else:
            return True

def get_sorted_index(input_list):
    # Returns sorted index correspond to input list
    # e.g. input : [4.1, 9.2, 5.6] -> return : [1,3,2]
    value = []
    index = []
    for i in range(len(input_list)):
        temp_val = min(input_list)
        temp_ind = input_list.index(temp_val)
        index.append(temp_ind)
        input_list[temp_ind] = float("inf")
    return index

def Nelder_mead(fopt, copt, alpha, beta, gamma):
    # fopt - option for function choosing
    # copt - option for criteria choosing
    assert alpha > 0, 'Invalid alpha for Nelder-mead'
    assert beta > 1, 'Invalid beta for Nelder-mead'
    assert gamma > 0 and gamma < 1, 'Invalid beta for Nelder-mead'

    # Get initial values
    X = random.sample(list(range(X_L,X_R)),3)
    Y = random.sample(list(range(Y_L,Y_R)),3)
    print('Initial points :','[',X[0],Y[0],']','[',X[1],Y[1],']','[',X[2],Y[2],']')

    GRAPH.append([X,Y])

    F = []
    for i in range(3):
        f_val,_,_ = functions(fopt, X[i], Y[i])
        F.append(f_val)

    si  = get_sorted_index(F)

    def contraction(F, X, Y, si, x_r, y_r, f_r, c_x, c_y, gamma, fopt):
        f_n_1 = F[si[-1]]
        if f_r < f_n_1:
            x_c = c_x + gamma(x_r - c_x)
            y_c = c_y + gamma(y_r - c_y)
        elif f_r >= f_n_1:
            x_c = c_x + gamma(X[si[-1]] - c_x)
            y_c = c_y + gamma(Y[si[-1]] - c_y)

        f_c,_,_ = functions(fopt, x_c, y_c)

        if f_c < min(f_r, f_n_1):
            return [X[si[0]], X[si[1]], x_c], [Y[si[0]], Y[si[1]], y_c]
        elif f_c >= min(f_r, f_n_1):
            return [X[si[0]], X[si[1]]+X[si[0]]/2, X[si[2]]+X[0]/2],\
                   [Y[si[0]], Y[si[1]]+Y[si[0]]/2, Y[si[2]]+Y[0]/2]

    def expansion(X, Y, si, x_r, y_r, c_x, c_y, beta, fopt):
        x_e = c_x + beta * (x_r - c_x)
        y_e = c_y + beta * (y_r - c_y)

        f_e,_,_ = functions(fopt, x_e, y_e)
        f_r,_,_ = functions(fopt, x_r, y_r)

        if f_e <= f_r:
            return [X[si[0]], X[si[1]], x_e], [Y[si[0]], Y[si[1]], y_e]
        else:
            return [X[si[0]], X[si[1]], x_r], [Y[si[0]], Y[si[1]], y_r]

    #reflection and main loof
    k = 1
    X_k = None
    X_k_ = None
    grad_fX_k = None
    fX_k = None
    fX_k_ = None
    Pk = None
    while(termination_test(copt, X_k=X_k, X_k_=X_k_,
                           grad_fX_k=grad_fX_k, fX_k=fX_k,
                           fX_k_=fX_k_, Pk=Pk, k=k)):

        x_k_1 = X[-1] #X_(k-1)
        y_k_1 = Y[-1] #Y_(k-1)
        f_k_1,_,_ = functions(fopt, X[-1], Y[-1])

        c_x = X[si[0]]+X[si[1]] / 2
        c_y = Y[si[0]]+Y[si[1]] / 2
        x_r = c_x + alpha * (c_x - X[si[-1]])
        y_r = c_y + alpha * (c_y - Y[si[-1]])

        f_r,_,_ = functions(fopt, x_r, y_r)

        if F[si[0]] <= f_r and f_r <= F[si[1]]:
            X[-1] = x_r, Y[-1] = y_r
        elif f_r >= F[si[1]]:
            X, Y = contraction(F, X, Y, si, x_r, y_r, f_r, c_x, c_y, gamma, fopt)
        elif f_r <= F[si[0]]:
            X, Y = expansion(X, Y, si, x_r, y_r, c_x, c_y, beta, fopt)

        F = []
        for i in range(3):
            f_val,_,_ = functions(fopt, X[i], Y[i])
            F.append(f_val)
        si  = get_sorted_index(F)

        GRAPH.append([X,Y])
        X_k = [X[si[-1]], Y[si[-1]]]
        X_k_ = [x_k_1, y_k_1]
        f_k, g_x, g_y = functions(fopt, X[si[-1]], Y[si[-1]])
        grad_fX_k = [g_x, g_y]
        fX_k = f_k
        fX_k_ = f_k_1
        Pk = [X[si[-1]] - x_k_1, Y[si[-1]] - y_k_1]
        k+=1
        if k >= ITR:
            print('Terminated before criterion converge')
            break

    print('Last points :','[',X[0],Y[0],']','[',X[1],Y[1],']','[',X[2],Y[2],']')
    print('Result : ', '[',(X[0]+X[1]+X[2])/3, (Y[0]+Y[1]+Y[2])/3,']')
    return [(X[0]+X[1]+X[2])/3, (Y[0]+Y[1]+Y[2])/3], k


def Powell(fopt, copt):

    u1 = [1,0]
    u2 = [0,1]
    U = [u1, u2]
    N = 2

    # Get initial values
    x_0 = random.choice(list(range(X_L,X_R)))
    y_0 = random.choice(list(range(Y_L,Y_R)))
    p_0 = [x_0, y_0]
    GRAPH.append(p_0)
    print('Initial point :',p_0)

    def find_gamma(fopt, p, u, step_size = 0.1, max_itr = 100):
        # p, u : [x value,y value] vector
        p_x_ = p[0]
        p_y_ = p[1]
        u_x_ = u[0]
        u_y_ = u[1]

        fval,_,_ = functions(fopt, p_x_, p_y_)
        for i in range(max_itr):
            gamma = step_size * i
            p_x = p_x_ + gamma * u_x_
            p_y = p_y_ + gamma * u_y_
            fval_new,_,_ = functions(fopt, p_x, p_y)
            if fval < fval_new:
                return gamma-step_size
            fval = fval_new
        return gamma

    gamma = find_gamma(fopt, p_0, u1)
    p_1 = [p_0[0]+gamma*u1[0], p_0[1]+gamma*u1[1]]
    gamma = find_gamma(fopt, p_1, u2)
    p_n = [p_1[0]+gamma*u2[0], p_1[1]+gamma*u2[1]]
    GRAPH.append(p_1)
    GRAPH.append(p_n)

    k = 1
    X_k = None
    X_k_ = None
    grad_fX_k = None
    fX_k = None
    fX_k_ = None
    Pk = None
    while(termination_test(copt, X_k=X_k, X_k_=X_k_,
                           grad_fX_k=grad_fX_k, fX_k=fX_k,
                           fX_k_=fX_k_, Pk=Pk, k=k)):

        u_new = [p_n[0]-p_0[0], p_n[1]-p_0[1]]
        U = [U[1], u_new]
        gamma = find_gamma(fopt, p_0, u_new)

        p_0 = [p_0[0]+gamma*u_new[0], p_0[1]+gamma*u_new[1]]

        u1 = U[0]
        u2 = U[1]
        gamma = find_gamma(fopt, p_0, u1)
        p_1 = [p_0[0]+gamma*u1[0], p_0[1]+gamma*u1[1]]
        gamma = find_gamma(fopt, p_1, u2)
        p_n = [p_1[0]+gamma*u2[0], p_1[1]+gamma*u2[1]]
        GRAPH.append(p_1)
        GRAPH.append(p_n)

        f_k, g_x, g_y = functions(fopt, p_n[0], p_n[1])
        f_k_1,_,_ = functions(fopt, p_1[0], p_1[1])

        X_k = p_n
        X_k_ = p_0
        grad_fX_k = [g_x, g_y]
        fX_k = f_k
        fX_k_ = f_k_1
        Pk = u2
        k+=1
        if k >= ITR:
            print('Terminated before criterion converge')
            break

    print('Result : ', p_n)
    return p_n, k


def main(args):

    if args.method == 'nelder_mead':
        X, itr = Nelder_mead(args.function_opt, args.criteria_opt,
                                         args.alpha, args.beta, args.gamma)
    elif args.method == 'powell':
        X, itr = Powell(args.function_opt, args.criteria_opt)

    print('Output -> point :', X, 'itr :', itr)
    if itr >= ITR:
        print("Dosen't converged")

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
        '--alpha', type=float, default=0.5)
    parser.add_argument(
        '--beta', type=float, default=1.5)
    parser.add_argument(
        '--gamma', type=float, default=0.5)
    parser.add_argument(
        '--method', type=str, default='nelder_mead',
        help='choosing methods')
    parser.add_argument(
        '--plot-range', nargs='+', type=float, default=[-20.0, 20.0],
        help='ploting range in x-axis')
    args = parser.parse_args()

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

    if args.method == 'nelder_mead':
        plt.plot(GRAPH[0][0], GRAPH[0][1],'go')
        plt.annotate('initial', xy=(GRAPH[0][0][0], GRAPH[0][1][0]))
        plt.annotate('initial', xy=(GRAPH[0][0][1], GRAPH[0][1][1]))
        plt.annotate('initial', xy=(GRAPH[0][0][2], GRAPH[0][1][2]))
        for i in range(len(GRAPH)-2):
            plt.plot(GRAPH[i+1][0][2], GRAPH[i+1][1][2],'bo')
            plt.annotate(str(i+1), xy=(GRAPH[i+1][0][2], GRAPH[i+1][1][2]))

        plt.plot(result_X[0], result_X[1],'ro')
        plt.annotate('result', xy=(result_X[0], result_X[1]))

        plt.savefig(args.method+'_opt'+str(args.function_opt)+'_crt'+str(args.criteria_opt)+'.png')

    elif args.method == 'powell':
        plt.plot(GRAPH[0][0], GRAPH[0][1],'go')
        plt.annotate('initial', xy=(GRAPH[0][0], GRAPH[0][1]))
        for i in range(len(GRAPH)-2):
            plt.plot(GRAPH[i+1][0], GRAPH[i+1][1],'bo')
            plt.annotate(str(i+1), xy=(GRAPH[i+1][0], GRAPH[i+1][1]))
        plt.plot(result_X[0], result_X[1],'ro')
        plt.annotate('result', xy=(result_X[0], result_X[1]))
        plt.savefig(args.method+'_opt'+str(args.function_opt)+'_crt'+str(args.criteria_opt)+'.png')

    '''
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
    '''
