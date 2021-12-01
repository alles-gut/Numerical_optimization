#####################################################
# Homework #6 Numerical Optimization, 2021          #
# Author Hyunjun Jung                               #
# vaild option for method argument:                 #
#  'GN' for 'Gauss-Newton’s method' and             #
#  'LM' for 'Levenberg-Marquardt method'            #
#####################################################

import csv
import random
import numpy as np

global ITR, E
global GRAPH
ITR = 100
E = 1e-6
GRAPH = []

def models(opt, coef, obs):
    # it returns function evaluation of the model
    # Input
    # opt : number of model (1 or 2)
    # coef : coefficient      [4,]
    # obs : observation data  [50, 3]
    # Ouput
    # f : function out        [50, 1]
    # J : jacobian            [50, 4]

    assert opt <= 1, 'Not implemented'
    if opt == 0: # ax + by + cz + d
        len_obs = obs.shape[0]
        f = coef[0]*obs[:,0] + coef[1]*obs[:,1] + coef[2]*obs[:,2] + coef[3]
        J = np.array([obs[:,0], obs[:,1], obs[:,2], np.array([1]*len_obs)],
                                                              dtype=np.float32)
    if opt == 1: # exp(-[(x-a)^2 + (y-b)^2 + (z-c)^2]/d^2)
        u = -((obs[:,0]-coef[0])**2 + (obs[:,1]-coef[1])**2 + \
                                                         (obs[:,2]-coef[2])**2)
        d = coef[3]**2 + 1e-6
        f = np.exp(u/d)
        J1 = 2*(obs[:,0]-coef[0])/d * f
        J2 = 2*(obs[:,1]-coef[1])/d * f
        J3 = 2*(obs[:,2]-coef[2])/d * f
        J4 = -u*2 / (coef[3]**3 + 1e-6) * f
        J = np.array([J1, J2, J3, J4], dtype=np.float32)

    return np.expand_dims(f, 1), J.T

def norm(X):
    return np.abs(X).sum()/len(X)

def GN(mopt, data, init_coef):
    # Gauss-Newton's method
    # Input
    # mopt : option for model choice
    # data : observation data, x_i, y_i, z_i, f_i   [50, 4]
    # init_coef : initial coefficient               [4,1]

    global ITR, E, GRAPH
    print('Operating Gauss-Newton\'s method')

    # initial values
    print('With initial point :',init_coef)

    fk,Jk = models(mopt, init_coef[:,0], data[:,:3])
    rk = fk-data[:,3:4]
    GRAPH.append(norm(rk))

    JTJ = np.matmul(Jk.T, Jk)
    JTr = np.matmul(Jk.T, rk)
    pk = -np.matmul(np.linalg.inv(JTJ),JTr)

    coef = init_coef + pk

    for itr in range(ITR):
        if (rk**2).sum() < E or abs(pk).sum() < E:
            print('Converged')
            print('Result :',coef, ', residual : ', (rk**2).sum(), ', Itr : ', itr+1)
            return coef, itr+1

        fk,Jk = models(mopt, coef[:,0], data[:,:3])
        rk = fk-data[:,3:4]
        GRAPH.append(norm(rk))

        JTJ = np.matmul(Jk.T, Jk)
        JTr = np.matmul(Jk.T, rk)
        pk = -np.matmul(np.linalg.inv(JTJ),JTr)
        coef = coef + pk

    print('Terminated before criterion converge')
    print('Result :',coef, ', residual : ', (rk**2).sum(), 'Itr : ', itr+1)
    print(GRAPH)
    return coef, ITR

def LM(mopt, data, init_coef, la):
    # Levenberg-Marquardt method
    # Input
    # mopt : option for model choice
    # data : observation data, x_i, y_i, z_i, f_i   [50, 4]
    # init_coef : initial coefficient               [4]
    # la : lambda value

    global ITR, E, GRAPH
    print('Operating Levenberg-Marquardt method')

    # initial values
    print('With initial point :',init_coef)

    fk,Jk = models(mopt, init_coef[:,0], data[:,:3])
    rk = fk-data[:,3:4]
    GRAPH.append(norm(rk))

    JTr = np.matmul(Jk.T, rk)
    JTJ = np.matmul(Jk.T, Jk) + la * np.eye(4)
    pk = -np.matmul(np.linalg.inv(JTJ),JTr)

    while np.dot(pk[0], JTr[0]) <= 0:
        la = la/10
        JTJ = np.matmul(Jk.T, Jk) + la * np.eye(4)
        pk = -np.matmul(np.linalg.inv(JTJ),JTr)
        if la < 1e-3:
            break

    while np.dot(pk[0], JTr[0]) > 0:
        la = la*10
        JTJ = np.matmul(Jk.T, Jk) + la * np.eye(4)
        pk = -np.matmul(np.linalg.inv(JTJ),JTr)
        if la > 1e+3:
            break

    coef = init_coef + 0.001*pk

    for itr in range(ITR):
        if (rk**2).sum() < E or abs(pk).sum() < E:
            print('Converged')
            print('Result :',coef, ', residual : ', (rk**2).sum(), ', Itr : ', itr+1)
            return coef, itr+1

        fk,Jk = models(mopt, coef[:,0], data[:,:3])
        rk = fk-data[:,3:4]
        GRAPH.append(norm(rk))

        JTr = np.matmul(Jk.T, rk)
        JTJ = np.matmul(Jk.T, Jk) + la * np.eye(4)
        pk = -np.matmul(np.linalg.inv(JTJ),JTr)

        while np.dot(pk[0], JTr[0]) <= 0:
            la = la/10
            JTJ = np.matmul(Jk.T, Jk) + la * np.eye(4)
            pk = -np.matmul(np.linalg.inv(JTJ),JTr)
            if la < 1e-3:
                break

        while np.dot(pk[0], JTr[0]) > 0:
            la = la*10
            JTJ = np.matmul(Jk.T, Jk) + la * np.eye(4)
            pk = -np.matmul(np.linalg.inv(JTJ),JTr)
            if la > 1e+3:
                break

        coef = coef + 0.001*pk

    print('Terminated before criterion converge')
    print('Result :',coef, ', residual : ', (rk**2).sum(), 'Itr : ', itr+1)
    return coef, ITR

def main(args):

    print('mopt :', args.model_opt,', method :', args.method)

    # initial values
    init_coef = np.ones((4,1))*0
    la = 100

    # loading data
    data = []
    filename = 'observation_data_for_LSM_LM(2021Fall).csv'
    f = open(filename, 'r', encoding='utf-8')
    rdr = csv.reader(f)
    for ind, line in enumerate(rdr):
        if ind == 0:
            pass
        else:
            data.append([float(line[0]), float(line[1]),float(line[2]), float(line[3])])
    f.close()
    data = np.array(data, dtype=np.float32)
    print('Data shape :', data.shape)

    if args.method == 'GN':
        coef, itr = GN(args.model_opt, data, init_coef)
    elif args.method == 'LM':
        coef, itr = LM(args.model_opt, data, init_coef, la)
    else:
        coef = 0
        itr = 0

    print('Output -> coef :', coef, 'itr :', itr)
    return coef

if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument(
        '--model-opt', type=int, default=0,
        help='choosing built-in functions')
    parser.add_argument(
        '--method', type=str, default='GN',
        help='choosing methods')
    args = parser.parse_args()
    # possible method : 'GN', 'LM'

    # main function
    result_coef = main(args)
