import os
import numpy as np
import matplotlib.pyplot as plt


def get_rounded(number):
    if number < 0:
        sign = -1
        number *= -1
    else:
        sign = 1

    n_decimals = 0
    temp = number
    while True:
        temp /= 10

        if temp < 1:
            temp *= 10
            temp = int(temp)
            break

        temp = int(temp)
        n_decimals += 1

    return temp*10**n_decimals*sign

def parse_files_construct_table(instances, logs_path_name='tests/solver/colgen'):

    root_name = os.path.dirname(os.getcwd())
    while True:
        base = os.path.basename(root_name)
        if base == 'decogo':
            break
        else:
            root_name = os.path.dirname(root_name)

    logs_path = os.path.join(root_name, logs_path_name)

    total_time_cg_bound = {}

    for i, problem_name in enumerate(instances):
        total_time_cg_bound[i] = []

        with open(os.path.join(logs_path, problem_name + '.txt')) as log_file:
            lines = log_file.readlines()

            for k, line in enumerate(lines):

                if line.startswith('IA obj. val:') or \
                        line.startswith('CG relaxation obj. value in iter'):
                    if line.startswith('IA obj. val:'):
                        val = float(line.split(':')[1])
                        time = float(lines[k + 1].split(':')[1])
                    if line.startswith('CG relaxation obj. value in iter'):
                        val = float(line.split(':')[1])
                        j = k + 1
                        while not lines[j].startswith('Used time at CG iter'):
                            j += 1
                        time = float(lines[j].split(':')[1].split('--')[1])

                    total_time_cg_bound[i].append((time, val))

    fig = plt.figure(figsize=(6, 3.2))
    ax = plt.gca()
    ax.set_xlabel('Time, s')
    ax.set_ylabel('IA objective value')

    for i, problem_name in enumerate(instances):
        ax.plot(*zip(*total_time_cg_bound[i]), label=problem_name)

    ax.legend()
    plt.tight_layout()
    fig.savefig('cg_relaxation_with_without_fast_cg.eps')


if __name__ == '__main__':

    instances = ['Frank-Wolfe=False',
                 'Frank-Wolfe=True']
    instances.sort()
    logs_path_name = 'tests/stats/tu_paper/acceleration_comparison_with_fast_cg'
    parse_files_construct_table(instances, logs_path_name)
