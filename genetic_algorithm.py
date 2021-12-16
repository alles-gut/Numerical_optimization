#####################################################
# Homework #7 Numerical Optimization, 2021          #
# Author Hyunjun Jung                               #
# vaild option for method argument:                 #
#  'GA' for 'Genetic Algorithm'                     #
#####################################################

import csv
import random
import numpy as np
import math

global BINLEN
global ITR, E
global local_GRAPH, global_GRAPH

BINLEN  = 16  # length of binary representation
ITR = 100
E = 1e-10
local_GRAPH = []
global_GRAPH = []


def bin2dec(binary):
    dec = 0
    for i in range(len(binary)):
        dec += (2**i)*int(binary[-(i+1)])
    return dec / (2**BINLEN-1)

def dec2bin(dec):
    return bin(int(math.floor(dec*(2**BINLEN-1))))[2:].zfill(BINLEN)

def list2string(input_list):
    return ''.join(str(e) for e in input_list)

def string2list(input_string):
    return [int(e) for e in input_string]

def functions(opt, x):
    # it returns function evaluation of the model

    assert opt <= 1, 'Not implemented'
    if opt == 0: # 2(x-0.5)^2 + 1
        f = 2*(x-0.5)**2 + 1
    elif opt == 1: # |x-0.5|(cos(12pi[x-0.5])+1.2)
        f = abs(x-0.5)*(math.cos(12*math.pi*(x-0.5))+1.2)

    return f

def GA(fopt, params):
    # Genetic Algorithm
    # Input
    # fopt : option for function choice
    # params : parameters
    #         [population_size,
    #          crossover_probability,
    #          mutation_probability]

    global ITR, E, local_GRAPH
    print('Operating Genetic Algorithm with function', fopt)

    # initial values
    print('Parameters :',params)

    init_population = [list(np.random.choice([0, 1], size=(BINLEN,)))
                                                     for i in range(params[0])]

    evaluation = []
    for individual in init_population:
        evaluation.append(functions(fopt, bin2dec(list2string(individual))))

    sorted_index = sorted(range(len(evaluation)), key=lambda k: evaluation[k])

    population = init_population
    last_best_value = 0
    curr_best_value = evaluation[sorted_index[0]]

    for itr in range(ITR):

        if abs(curr_best_value-last_best_value) < E:
            print('Terminated by improvement')
            print('Result :',bin2dec(list2string(population[sorted_index[0]])),
                                                              ', Itr :', itr+1)
            return bin2dec(list2string(population[sorted_index[0]])), itr+1


        last_best_value = curr_best_value
        local_GRAPH.append(bin2dec(list2string(population[sorted_index[0]])))

        # selection
        parents = [population[sorted_index[0]], population[sorted_index[1]]]

        # crossover (uniform crossover)
        crossover = []
        for i in range((params[0]-2)//2//2):
            prob = np.random.choice([0, 1], size=(BINLEN,),
                                             p=[1-params[1]/100, params[1]/100])
            child1 = []
            child2 = []
            for i in range(BINLEN):
                if prob[i]:
                    child2.append(parents[0][i])
                    child1.append(parents[1][i])
                else:
                    child1.append(parents[0][i])
                    child2.append(parents[1][i])

            crossover.append(child1)
            crossover.append(child2)

        # mutation
        mutation = []
        for i in range((params[0]-2)//2//2):
            prob = np.random.choice([0, 1], size=(BINLEN,),
                                             p=[1-params[2]/100, params[2]/100])
            child1 = []
            child2 = []
            for i in range(BINLEN):
                if prob[i]:
                    child1.append(abs(parents[0][i]-1))
                    child2.append(abs(parents[1][i]-1))
                else:
                    child1.append(parents[0][i])
                    child2.append(parents[1][i])

            crossover.append(child1)
            crossover.append(child2)

        # fill gab
        new_population = [list(np.random.choice([0, 1], size=(BINLEN,)))
                      for i in range(params[0]-len(parents+crossover+mutation))]

        # replace
        population = parents + crossover + mutation + new_population

        # fitness
        evaluation = []
        for individual in population:
            evaluation.append(functions(fopt, bin2dec(list2string(individual))))

        sorted_index = sorted(range(len(evaluation)),
                                               key=lambda k: evaluation[k])

        curr_best_value = evaluation[sorted_index[0]]

    print('Terminated by maximum iteration')
    print('Result :',bin2dec(list2string(population[sorted_index[0]])),
                                                              ', Itr :', itr+1)
    return bin2dec(list2string(population[sorted_index[0]])), ITR

def main(args):

    print('fopt :', args.function_opt)

    params = [args.population_size,
              args.crossover_probability,
              args.mutation_probability]

    output, itr = GA(args.function_opt, params)

    print('Output :', output, 'itr :', itr)
    return output, itr

if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument(
        '--function-opt', type=int, default=1,
        help='choosing built-in functions')
    parser.add_argument(
        '--population-size', type=int, default=20,
        help='population size of chromosomes')
    parser.add_argument(
        '--crossover-probability', type=int, default=90,
        help='probability of crossover in percent')
    parser.add_argument(
        '--mutation-probability', type=int, default=0.5,
        help='probability of mutation in percent')
    parser.add_argument(
        '--plot-range', nargs='+', type=float, default=[0, 1],
        help='ploting range in x-axis')
    args = parser.parse_args()

    # main function
    output, itr = main(args)

    # visualize the single evaluation
    import matplotlib.pyplot as plt
    import numpy as np
    x = np.arange(args.plot_range[0], args.plot_range[1], 0.01)
    y = [(functions(args.function_opt, x_),0) for x_ in x]
    plt.plot(x, y)

    op_x = []
    op_y = []
    for i in range(len(local_GRAPH)):
        x_val = local_GRAPH[i]
        y_val = functions(args.function_opt, local_GRAPH[i])
        op_x.append(x_val)
        op_y.append(y_val)
        plt.annotate(str(i), xy=(x_val,y_val))
    plt.plot(op_x, op_y, 'ro')

    plt.savefig('fopt'+str(args.function_opt)+'.png')

    plt.cla()

    # multiple prediction
    pred_iter = 100
    temp = []
    for i in range(pred_iter):
        output,itr = main(args)
        global_GRAPH.append(output)
        temp.append(itr)
    print('Result of multiple prediction :', np.array(global_GRAPH).mean(),
                                         ', Iteration :', np.array(temp).mean())

    plt.plot(x, y)

    op_x = []
    op_y = []
    for i in range(len(global_GRAPH)):
        x_val = global_GRAPH[i]
        y_val = functions(args.function_opt, global_GRAPH[i])
        op_x.append(x_val)
        op_y.append(y_val)
        plt.annotate(str(i), xy=(x_val,y_val))
    plt.plot(op_x, op_y, 'bo')

    plt.savefig('multiple_fopt'+str(args.function_opt)+'.png')
