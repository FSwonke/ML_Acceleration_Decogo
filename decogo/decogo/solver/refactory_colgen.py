"""Class which implements Column Generation with user-defined input model"""

import copy
import logging
import math
import time

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pylab as pl
from matplotlib.patches import Rectangle
import seaborn as sns
import pandas as pd
from sklearn.decomposition import PCA
from scipy import stats
from sklearn.preprocessing import StandardScaler
from decogo.solver.settings import Settings
from decogo.util.block_vector import BlockVector

logger = logging.getLogger('decogo')


class RefactoryColGen:
    """Class which implements Column Generation algorithm with user-defined \
    input model

    :param problem: Decomposed problem class, which stores all input data
    :type problem: DecomposedProblem
    :param settings: Settings class
    :type settings: Settings
    :param result: Class which stores the results
    :type result: Results
    """

    def __init__(self, problem, settings, result):
        """Constructor method"""
        self.problem = problem
        self.result = result
        self.settings = settings

        self.alpha = None
        self.n_subproblems_main = {}
        self.phase_list = {}
        self.newpoints = {}
        self.predictions = {}
        self.corrections = {}
        self.y_main = {}
        self.X_main = {}
        self.anomaly = {}
        self.y_score = {}
        self.a_threshold = {}
        self.spe_theshold = {}
        self.spe_threshold = {}
        self.x_ia = {}
        self.X_main_plot = {}
        self.PC_tr = {}
        self.PC_main = {}
        self.t_score = {}
        self.spe_score = {}
        self.spe_main = {}
        self.t2 = {}
        # initiation of lists for phase list
        for k in range(self.problem.block_model.num_blocks):
            self.phase_list[k] = []
            self.newpoints[k] = []
            self.predictions[k] = None
            self.corrections[k] = []
            self.y_main[k] = []
            self.X_main[k] = []
            self.anomaly[k] = []
            self.n_subproblems_main[k] = 1
            self.y_score[k] = None
            self.a_threshold[k] = None
            self.spe_threshold[k] = None
            self.x_ia[k] = []
            self.X_main_plot[k] = []
            self.PC_main[k] = None
            self.PC_tr[k] = None
            self.t_score[k] = None
            self.spe_score[k] = None
            self.spe_main[k] = None
            self.t2[k] = []

        self.ndata = {}

    def solve(self):
        """
        Inner approximation algorithm
        """

        self.ia_init()
        print('=======================')
        print('IA_init done')
        print('=======================')
        print(self.ndata)
        for k_phase in range(self.problem.block_model.num_blocks):

            self.phase_list[k_phase].append(self.ndata[k_phase])
        k_phase = 0

        # initial column generation
        tic_init_cg = time.time()
        i_find_sol = 0
        while True:  # find feasible solution and eliminate slacks
            # in IA master problem
            i_find_sol += 1
            tic = time.time()

            # option to generate sub-problem data with exact solver
            if self.settings.exact_solve_data is False:
                self.column_generation(approx_solver=True)  # solve sub-problems
                # approximately
            else:
                self.column_generation()  # solve sub-problems exactly
            # approximately

            print('=======================')
            print('Approx Colgen done')
            print('=======================')
            for k_phase in range(self.problem.block_model.num_blocks):
                self.phase_list[k_phase].append(self.ndata[k_phase])
            k_phase = 0
            time_cg = round(time.time() - tic, 2)
            logger.info('Time used for init CG '
                        'in iter {0}: --{1}-- seconds'
                        .format(self.result.main_iterations, time_cg))
            self.result.current_used_time += time_cg
            logger.info('-----------------------------------------------------')
            logger.info('Elapsed time: --{0}-- seconds'
                        .format(round(self.result.current_used_time, 2)))
            logger.info('-----------------------------------------------------')
            # find solution performed always in the beginning
            tic = time.time()
            self.find_solution_init(self.result.main_iterations)
            time_find_solution = round(time.time() - tic, 2)
            logger.info('Time used for init FindSol '
                        'in iter {0}: --{1}-- seconds'
                        .format(self.result.main_iterations, time_find_solution))
            self.result.current_used_time += time_find_solution
            logger.info('-----------------------------------------------------')
            logger.info('Elapsed time: --{0}-- seconds'
                        .format(round(self.result.current_used_time, 2)))
            logger.info('-----------------------------------------------------')

            if self.result.primal_bound < float('inf'):
                logger.info('Found the first feasible solution')

                logger.info('IA obj. val: {0}'.format(
                    self.result.cg_relaxation * self.result.sense))
                logger.info('Elapsed time: {0}'.format(
                    self.result.current_used_time))

                break

        j = 0
        while True:  # apply cg_fast_fw or approx_cg
            j += 1
            tic = time.time()

            if self.settings.cg_fast_fw:
                self.column_generation_fast_fw()
                time_init_cg = round(time.time() - tic, 2)
                logger.info('Time used for init cg fast fw '
                            'in iter {0}: --{1}-- seconds'
                            .format(j, time_init_cg))
            else:
                self.column_generation(approx_solver=True)
                time_init_cg = round(time.time() - tic, 2)
                logger.info('Time used for init approx cg '
                            'in iter {0}: --{1}-- seconds'
                            .format(j, time_init_cg))

            self.result.current_used_time += time_init_cg
            logger.info(
                '-----------------------------------------------------')
            logger.info('Elapsed time: --{0}-- seconds'
                        .format(round(self.result.current_used_time, 2)))
            logger.info(
                '-----------------------------------------------------')
            if j == 5:
                break
        for k_phase in range(self.problem.block_model.num_blocks):
            self.phase_list[k_phase].append(self.ndata[k_phase])
        k_phase = 0

        time_init_cg = round(time.time() - tic_init_cg, 2)
        logger.info('\nCG relaxation obj. value '
                    'in iter {0}: {1}'.format(self.result.main_iterations,
                                              self.result.sense *
                                              self.result.cg_relaxation))
        logger.info('Time used for total init CG '
                    'in iter {0}: --{1}-- seconds'.
                    format(self.result.main_iterations, time_init_cg))

        logger.info('-----------------------------------------------------')
        logger.info('Elapsed time at CG iter {0}: --{1}-- seconds'
                    .format(self.result.main_iterations,
                            round(self.result.current_used_time, 2)))
        logger.info('-----------------------------------------------------')

        # flag which says if reduced cost is positive and we can stop
        stop_by_cg_converg = False

        # init hat K
        hat_k_set = {k for k in range(self.problem.block_model.num_blocks)
                     if self.problem.block_model.sub_models[k].linear
                     is False}

        time_i_loop_set = []
        print('====================================================')
        print('phase_list')
        print(self.phase_list)
        print('====================================================')

        #for k in range(self.problem.block_model.num_blocks):
         #   self.init_ML(k)
        k=0

        while True:
            print('=====================================================')
            print('                 main iteration                      ')
            print('=====================================================')

            self.result.main_iterations += 1
            print('iteration', self.result.main_iterations)


            tic_i_start = time.time()

            num_subproblems_solved = self.result.cg_num_minlp_problems

            tic = time.time()
            print('Column Generation')
            self.column_generation(hat_k_set)

            time_column_generation = round(time.time() - tic, 2)
            logger.info('CG relaxation obj. value in iter {0}: {1}'
                        .format(self.result.main_iterations, self.result.sense *
                                self.result.cg_relaxation))
            logger.info('\nTime used for CG: --{0}-- seconds'
                        .format(time_column_generation))

            self.result.current_used_time += time_column_generation
            logger.info('-------------------------------------------------')
            logger.info('Elapsed time at CG iter {0}: --{1}-- seconds'
                        .format(self.result.main_iterations,
                                round(self.result.current_used_time, 2)))
            logger.info('-------------------------------------------------')
            logger.info('\nNum of MINLP subproblems solved '
                        'in iter loop *{0}*   {1}'
                        .format(self.result.main_iterations,
                                self.result.cg_num_minlp_problems -
                                num_subproblems_solved))
            logger.info('Total number of minlp subproblems '
                        'solved in iter {0}: {1}'
                        .format(self.result.main_iterations,
                                self.result.cg_num_minlp_problems))

            column_blocks = [self.problem.get_inner_points_size(k) for k in
                             range(self.problem.block_model.num_blocks)]
            column_sum = sum(self.problem.get_inner_points_size(k) for k in
                             range(self.problem.block_model.num_blocks))

            logger.info('Total number of columns '
                        'in iter {0}: {1}'
                        .format(self.result.main_iterations,
                                column_sum))
            logger.info('Columns in blocks '
                        'in iter {0}: {1}'
                        .format(self.result.main_iterations, column_blocks))

            logger.info('Time used for CG in iter '
                        '{0}: --{1}-- seconds'.format(
                            self.result.main_iterations,
                            time_column_generation))

            logger.info('------------------------------------')
            logger.info('CG regarding all blocks')
            # col/cut generation for all blocks; calculate reduced cost
            tic = time.time()
            print('SolverInnerLP')
            z, x_ia, w_ia, slacks, duals, obj_value_ia = \
                self.problem.master_problems.solve_ia(self.settings.lp_solver)

            reduced_cost_direction = np.concatenate(([1], duals))

            # reduce block set based on the reduced cost
            hat_k_set = []

            for k in range(self.problem.block_model.num_blocks):

                if self.problem.block_model.sub_models[k].linear is False:

                    #fpoint, _, delta_k, new_point, _ = \
                        #self.generate_column(k, reduced_cost_direction)
                    '''
                    feasible_point, primal_bound, delta_k, \
                    new_point, _ 
                    '''
                    fpoint, _, delta_k, _, _ = self.ML_ColGen(k, reduced_cost_direction, x_ia=x_ia)

                    if delta_k <= -1e-3:
                        hat_k_set.append(k)

            time_column_generation = round(time.time() - tic, 2)
            logger.info('Time used for CG for all blocks: --{0}-- seconds'
                        .format(round(time_column_generation, 2)))
            self.result.current_used_time += time_column_generation
            logger.info('-----------------------------------------------------')
            logger.info('Elapsed time: --{0}-- seconds'
                        .format(round(self.result.current_used_time, 2)))
            logger.info('-----------------------------------------------------')

            # check if reduced block set is empty.
            # if it is empty then we stop, since for all blocks reduced cost
            # was positive

            if len(hat_k_set) == 0:
                if self.settings.cg_check_convergence is True:
                    logger.info('CG converges, checking the convergence by '
                                'exact subproblem solving')
                    # solve again the subproblems regarding the same direction
                    # until optimality
                    # if here reduced cost is positive, then we can stop,
                    # otherwise we should continue
                    hat_k_set = []
                    for k in range(self.problem.block_model.num_blocks):
                        if self.problem.block_model.sub_models[k].linear \
                                is False:
                            # heuristic=False forces SCIP to use default
                            # settings, i.e it tries to solve the subproblem
                            # until optimality not clear whether it will slow
                            # down everything too much maybe here is better
                            # to use just stricter settings for SCIP

                            #fpoint, _, delta_k, new_point, _ = \
                                #self.generate_column(k,
                                                     #reduced_cost_direction,
                                                     #heuristic=False)
                            '''
                            feasible_point, primal_bound, delta_k,\
                                new_point, _ \
                            '''
                            fpoint, _, delta_k, _, _ = self.ML_ColGen(k, reduced_cost_direction, x_ia=x_ia)

                            #'''

                            if delta_k <= -1e-3:
                                hat_k_set.append(k)

                    if len(hat_k_set) == 0:
                        stop_by_cg_converg = True
                else:
                    stop_by_cg_converg = True

            # primal heuristics
            if self.settings.cg_find_solution:
                tic = time.time()
                self.find_solution(
                    self.result.main_iterations)
                time_find_solution = round(time.time() - tic, 2)
                logger.info('Time used for FindSol in iter {0}'
                            ': --{1}-- seconds'.
                            format(self.result.main_iterations,
                                   time_find_solution))
                self.result.current_used_time += time_find_solution
                logger.info('-------------------------------------------------')
                logger.info('Elapsed time at FindSol iter {0}: --{1}-- seconds'
                            .format(self.result.main_iterations,
                                    round(self.result.current_used_time, 2)))
                logger.info('-------------------------------------------------')

            # main iteration time counting
            time_i_loop = round(time.time() - tic_i_start, 2)
            time_i_loop_set.append(time_i_loop)
            logger.info('Total time used in iter {0}'
                        ': --{1}-- seconds'. format(self.result.main_iterations,
                                                    time_i_loop))

            if self.result.main_iterations == \
                    self.settings.cg_max_main_iter:
                logger.info('\nIteration limit')
                break

            if stop_by_cg_converg is True:
                logger.info('CG converged')
                break

        self.result.num_of_columns_after_cg = sum(
            self.problem.get_inner_points_size(k)
            for k in range(self.problem.block_model.num_blocks))

        self.result.total_number_columns = sum(
            self.problem.get_inner_points_size(k)
            for k in range(self.problem.block_model.num_blocks))

        self.result.total_sub_problem_number \
            = self.result.cg_num_minlp_problems + \
            self.result.cg_num_unfixed_nlp_problems + \
            self.result.cg_num_fixed_nlp_problems

        self.result.sub_problem_number_after_cg = \
            self.result.total_sub_problem_number

        #self.plot_train_data()
        print('times main iteration: ', time_i_loop_set)
        print('subproblems in main iteration')
        print(self.n_subproblems_main)
        k=0
        #for k in range(self.problem.block_model.num_blocks):
            #pred, y_test, score = self.test_ML(k)
            #print('Block ', k)
            #self.write_text(k)
            #print('new points: ', self.newpoints[k])
            #print('predictions: ', self.predictions[k])
            #print('corrections: ', self.corrections[k])
            #print('val set', self.y_main[k])
            #self.plot_main(k)
        #self.plot_train_data()
        self.testing()

    def ia_init(self, duals=None):
        """Initialization of inner outer approximation

        :param duals: Initial dual vector for initialization
        :type duals: ndarray
        :return: Solution of MIP projection master problem
        :rtype: BlockVector
        """

        logger.info('\nInitialization')

        tic = time.time()

        # initial dual vector
        if duals is None:
            duals = np.zeros(shape=self.problem
                             .block_model.cuts.num_of_global_cuts)

        self.sub_gradient(duals)

        time_sub_gradient = round(time.time() - tic, 2)
        logger.info('\nTime used for SubGradient: --{0}-- seconds'
                    .format(time_sub_gradient))
        self.result.current_used_time += time_sub_gradient

        logger.info('-----------------------------------------------------')
        logger.info('Elapsed time: {0}'.format(self.result.current_used_time))
        logger.info('-----------------------------------------------------')

    def column_generation(self, subset_of_blocks=None, approx_solver=False):
        """Performs column generation steps (see paper)

        :param subset_of_blocks: apply column generation for reduced block set
        :type subset_of_blocks: list or None
        :param approx_solver: enables approximate solving of subproblems in \
        column generation
        :type approx_solver: bool
        :return: Dual solution from IA master problem regarding global \
        constraints
        :rtype: ndarray
        """
        logger.info('\n=======================================================')
        if approx_solver:
            logger.info('Column generation: approximated subproblem solving')
        else:
            logger.info('Column generation')
        new_columns_generated = [0] * self.problem.block_model.num_blocks
        num_minlp_problems_solved = self.result.cg_num_minlp_problems

        if subset_of_blocks is not None:
            blocks = subset_of_blocks
        else:
            blocks = {k for k in range(self.problem.block_model.num_blocks)
                      if self.problem.block_model.sub_models[k].linear is False}

        i = 0

        while True:

            z, x_ia, w_ia, slacks, duals, obj_value_ia = \
                self.problem.master_problems.solve_ia(self.settings.lp_solver)
            print('=======x_ia========')
            #print(x_ia)
            #print(type(x_ia[0, :]))
            #print(x_ia[0, :])
            self.result.cg_relaxation = obj_value_ia
            if i == 0:
                initial_obj_value_ia = obj_value_ia
                logger.info('\nInitial CG objective value: {0}'.format(
                    self.result.sense * initial_obj_value_ia))

            max_slack_value = max(max(item) for item in slacks)
            sum_slack_values = sum(item[0] + item[1] for item in slacks)

            i += 1

            logger.info('{0: <15}{1: <30}{2: <30}'
                        '{3: <30}'.format('CG iter', 'IA obj. value',
                                          'max slack value IA',
                                          'sum slack values IA'))

            logger.info('{0: <15}{1: <30}{2: <30}''{3: <30}'
                        .format(i, self.result.sense *
                                self.result.cg_relaxation,
                                max_slack_value, sum_slack_values))

            reduced_cost_direction = np.concatenate(([1], duals))
            #print('=============INFO==================================')
            #print('dict training data self.tdata:')
            #print(self.tdata)
            #if i == 4:

            # generate new columns
            generate_column_time_list = {}
            reduced_cost_list = {}

            # generate columns according to dual values
            for k in blocks:
                tic = time.time()
                '''
                _, _, reduced_cost_list[k], new_point, _ = \
                    self.generate_column(k, reduced_cost_direction,
                                         approx_solver=approx_solver,
                                         x_k=x_ia.get_block(k))
                '''
                fpoint, _, reduced_cost_list[k], \
                    new_point, _ = self.ML_ColGen(k, reduced_cost_direction, x_ia=x_ia,
                                                  approx_solver=approx_solver)


                #'''
                generate_column_time_list[k] = round(time.time() - tic, 2)
                if new_point is True:
                    new_columns_generated[k] += 1

            # generate columns according to non-zero slacks
            generate_column_slack_time_list = {}
            if max_slack_value > 1e-1:

                z, x_ia, w_ia, slacks, duals, obj_value_ia = \
                    self.problem.master_problems.solve_ia(
                        self.settings.lp_solver)
                self.result.cg_relaxation = obj_value_ia
                max_slack_value = max(max(item) for item in slacks)

                if max_slack_value > 1e-1:

                    slack_direction = self.get_slack_directions(slacks)
                    for k in blocks:
                        tic = time.time()
                        '''
                        _, _, _, new_point, _ = \
                            self.generate_column(k, slack_direction,
                                                 approx_solver=approx_solver,
                                                 x_k=x_ia.get_block(k))
                        '''
                        fpoint, _, _, \
                            new_point, _ = self.ML_ColGen(k, reduced_cost_direction, x_ia=x_ia,
                                                      approx_solver=approx_solver)


                        generate_column_slack_time_list[k] = \
                            round(time.time() - tic, 2)
                        if new_point is True:
                            new_columns_generated[k] += 1


            if i >= self.settings.cg_max_iter:
                logger.info('Iteration limit')
                # logger.info('Reduced cost: {0}'
                #             .format(str(reduced_cost_list)))
                logger.info('New columns added: {0}'
                            .format(str(new_columns_generated)))
                break

            if all([item >= -1e-6 for item in
                    reduced_cost_list.values()]) is True:
                logger.info('Reduced costs greater than zero')
                logger.info('New columns added: {0}'
                            .format(str(new_columns_generated)))
                break

        self.result.num_cg_iterations += i

        # number of MINLP subproblems during CG
        logger.info(
            'number of minlp subproblems '
            'solved during CG: {0}'.format(self.result.cg_num_minlp_problems -
                                           num_minlp_problems_solved))
        logger.info('\n=======================================================')
        return duals

    def column_generation_fast_fw(self):
        """ Performs fast FW column generation steps (see paper) """
        logger.info('---------------------------------------------------------')
        logger.info('Fast column generation')

        start_time = time.time()

        # solve ia master problem
        z, x_ia, w_ia, slacks, duals, obj_value_ia = \
            self.problem.master_problems.solve_ia(self.settings.lp_solver)
        self.result.cg_relaxation = obj_value_ia

        direction = np.concatenate(([1], duals))

        # it is always good idea to have a look at the existing code and
        # make the code as simple as possible to read
        tilde_w = BlockVector()
        for k in range(self.problem.block_model.num_blocks):
            if self.problem.block_model.sub_models[k].linear is False:
                column, _ = self.problem.get_min_column(k, direction)
                tilde_w.set_block(k, column)
            else:
                tilde_w.set_block(k, w_ia.get_block(k))

        v_plus = copy.copy(tilde_w)

        # todo: refactor this, make use of BlockVector,
        # if neccessary extend BlockVector class
        for j, (index, vector) in enumerate(tilde_w.vectors.items()):
            if j == 0:
                tilde_w_array = np.array(vector.reshape(1, len(vector)))
            else:
                tilde_w_array = \
                    np.concatenate((tilde_w_array,
                                    vector.reshape(1, len(vector))))

        for j, (index, vector) in enumerate(v_plus.vectors.items()):
            if j == 0:
                v_plus_array = np.array(vector.reshape(1, len(vector)))
            else:
                v_plus_array = \
                    np.concatenate((v_plus_array,
                                    vector.reshape(1, len(vector))))
        sigma_cg = abs(duals)
        global_cuts_rhs = []
        for cut in self.problem.block_model.cuts.global_cuts:
            global_cuts_rhs.append(cut.rhs)
        global_cuts_rhs = np.array(global_cuts_rhs)

        tic_fast_cg = time.time()
        num_unfixed_nlp_problems_solved = \
            self.result.cg_num_unfixed_nlp_problems

        gamma_cg_plus = 1

        i = 0
        logger.info('{0: <10}{1: <30}'
                    '{2: <30}'.format('iter', 'IA obj. value', 'slacks'))
        logger.info('{0: <10}{1: <30}'
                    '{2: <30}'.format(i, obj_value_ia * self.result.sense,
                                      sum(map(sum, slacks))))

        if sum(map(sum, slacks)) < 1e-2:
            logger.info('IA obj. val: {0}'.format(
                obj_value_ia * self.result.sense))
            logger.info('Elapsed time: {0}'.format(
                self.result.current_used_time + (time.time() - start_time)))

        new_columns_generated_cumulative = \
            {k: 0 for k in range(self.problem.block_model.num_blocks)
             if self.problem.block_model.sub_models[k].linear is False}

        while True:
            i += 1

            generate_column_time_list = {}
            reduced_cost_list = {}
            r_cg = BlockVector()
            new_columns_generated = []
            for k in range(self.problem.block_model.num_blocks):
                tic = time.time()
                if self.settings.cg_fast_approx:
                    _, _, reduced_cost_list[k], new_point, r_k = \
                        self.local_solve_subproblem(
                            k, direction, x_k=x_ia.get_block(k))
                else:
                    #_, _, reduced_cost_list[k], new_point, r_k = \
                     #   self.generate_column(k, direction)

                    _, _, reduced_cost_list[k], new_point, r_k = \
                        self.ML_ColGen(k,direction, heuristic=True,
                            approx_solver=False, x_ia=x_ia)

                    generate_column_time_list[k] = round(time.time() - tic,
                                                     2)
                if r_k is not None:
                    r_cg.set_block(k, r_k)
                else:
                    r_cg.set_block(k, w_ia.get_block(k))
                # logger.info(
                #     'added r_{0} in iter {1}'.format(k, i))
                if new_point is True:
                    # count in current iteration
                    new_columns_generated.append(1)

                    # cumulative count
                    new_columns_generated_cumulative[k] += 1
                else:
                    new_columns_generated.append(0)

            if all(item == 0 for item in new_columns_generated):
                logger.info('No new columns generated in the current iteration')
                break

            z, x_ia, w_ia, slacks, duals, obj_value_ia = \
                self.problem.master_problems.solve_ia(self.settings.lp_solver)
            self.result.cg_relaxation = obj_value_ia

            logger.info('\n{0: <10}{1: <30}'
                        '{2: <30}'.format('iter', 'IA obj. value', 'slacks'))
            logger.info('{0: <10}{1: <30}'
                        '{2: <30}'.format(i,
                                          obj_value_ia * self.result.sense,
                                          sum(map(sum, slacks))))

            if sum(map(sum, slacks)) < 1e-2:
                logger.info('IA obj. val: {0}'.format(
                    obj_value_ia * self.result.sense))
                logger.info('Elapsed time: {0}'.format(self.result.current_used_time +
                                               (time.time() - start_time)))

            # use column
            for j, (index, vector) in enumerate(r_cg.vectors.items()):
                if j == 0:
                    r_cg_array = np.array(vector.reshape(1, len(vector)))
                else:
                    r_cg_array = \
                        np.concatenate((r_cg_array,
                                        vector.reshape(1, len(vector))))

            # the operations add, substract and multiplication by scalar is
            # implemented in BlockVector class. Maybe it is easier to use them
            # The implementation of summing the components of BlockVector
            # can implemented and added there

            # determining the step size
            coeff_w_r_array = r_cg_array - tilde_w_array
            coeff_a = np.multiply((np.sum(coeff_w_r_array[:, 1:], axis=0)
                                   - global_cuts_rhs), sigma_cg)
            coeff_a = \
                2 * np.dot(coeff_a, np.sum(coeff_w_r_array[:, 1:], axis=0))
            coeff_b = np.multiply(sigma_cg,
                                  np.sum(tilde_w_array[:, 1:], axis=0))
            coeff_b = 2 * np.dot(coeff_b,
                                 np.sum(coeff_w_r_array[:, 1:], axis=0))
            coeff_b += np.sum(coeff_w_r_array[:, 0], axis=0)

            if coeff_a == 0:  # tilda_w equals nu_plus
                # logger.info('tilda_w equals nu_plus')
                break
            else:
                theta_cg = - coeff_b / coeff_a
            # logger.info('theta_cg: {0}'.format(theta_cg))
            if theta_cg == 0:
                break

            # update step
            add_term = theta_cg * (r_cg_array - tilde_w_array)
            v_array = copy.copy(v_plus_array)
            v_plus_array = tilde_w_array + add_term
            # fast FW step
            gamma_cg = gamma_cg_plus
            gamma_cg_plus = 0.5 * (1 + math.sqrt(4 * gamma_cg ** 2
                                                 + 1))
            tilde_w_array = v_plus_array + \
                            (gamma_cg - 1) / gamma_cg_plus * \
                            (v_plus_array - v_array)

            gap = {}
            for k in range(self.problem.block_model.num_blocks):
                if self.problem.block_model.sub_models[k].linear is False:
                    gap[k] = np.dot(direction, (tilde_w_array[k, :] -
                                                r_cg_array[k, :]))

            # logger.info('Value of gap (g_k):')
            # logger.info(gap)

            logger.info('Number of new columns in the current iteration:')
            logger.info(new_columns_generated)

            # update direction
            direction = 2 * np.multiply(sigma_cg, np.sum(tilde_w_array[:, 1:],
                                                         axis=0) -
                                        global_cuts_rhs)
            direction = np.concatenate(([1], direction))

            if i == self.settings.cg_fast_fw_max_iter:
                break

        logger.info('\nNew columns in FastCG:')
        logger.info(list(new_columns_generated_cumulative.values()))

        # number of unfixed nlp subproblems during CG
        logger.info('number of unfixed nlp subproblems '
                    'solved during CG: {0}'
                    .format(self.result.cg_num_unfixed_nlp_problems -
                            num_unfixed_nlp_problems_solved))
        time_fast_cg = round(time.time() - tic_fast_cg, 2)
        logger.info('Time used for solving subproblem'
                    ': --{0}-- seconds'.
                    format(time_fast_cg))
        logger.info('---------------------------------------------------------')

    def sub_gradient(self, direction_vector):
        """Performs sub-gradient iterations

        :param direction_vector: Initial vector for sub-gradient iterations
        :type direction_vector: ndarray
        :return: Final direction vector
        :rtype: tuple
        """

        y = BlockVector()

        # create an array of rhs of global constraints
        b = np.zeros(shape=self.problem.block_model.cuts.num_of_global_cuts)
        for j, cut in enumerate(self.problem.block_model.cuts.global_cuts):
            b[j] = cut.rhs

        iteration = 0
        alpha = 1
        logger.info('\nSubgradient steps')
        logger.info('{0: <15}{1: <30}{2: <30}'
                    .format('Subgra.iter', 'Lagrange bound', 'alpha'))

        direction_vector_prev = copy.deepcopy(direction_vector)
        lag_sol_prev = float('-inf')

        violation = np.zeros(
            shape=self.problem.block_model.cuts.num_of_global_cuts)

        while iteration < self.settings.cg_sub_gradient_max_iter:
            time_generate_column_sub_gradient_list = {}
            new_columns_generated = [0] * self.problem.block_model.num_blocks
            iteration += 1
            lag_sol = 0
            direction = np.concatenate(([1], direction_vector))
            for k in range(self.problem.block_model.num_blocks):
                tic = time.time()
                feasible_point, primal_bound, reduced_cost, new_point, _ = \
                    self.generate_column(k, direction)
                # add to training data manually
                dir_orig_space = \
                    self.problem.block_model.trans_into_orig_space(k,
                                                                   direction)
                self.problem.training_data(k, dir_orig_space, feasible_point)
                len_data = self.problem.get_size_training_data(k)
                print('length data: ', len_data, 'in block:', k)

                self.ndata[k] = len_data
                #
                time_generate_column_sub_gradient_list[k] = round(
                    time.time() - tic, 2)
                if new_point is True:
                    new_columns_generated[k] += 1
                y.set_block(k, feasible_point)
                lag_sol += primal_bound
            lag_sol -= np.dot(direction_vector, b)

            logger.info('{0: <15}{1: <30}{2: <30}'
                        .format(iteration, self.result.sense * lag_sol, alpha))

            if iteration == 1:
                violation = self.problem.eval_viol_lin_global_constraints(y)

                lag_sol_prev = lag_sol
                direction_vector_prev = copy.deepcopy(direction_vector)
            else:
                # update the step
                if lag_sol <= lag_sol_prev:
                    alpha *= 0.5
                    direction_vector = copy.deepcopy(direction_vector_prev)
                else:
                    alpha *= 2
                    violation = self.problem.eval_viol_lin_global_constraints(y)

                    lag_sol_prev = lag_sol
                    direction_vector_prev = copy.deepcopy(direction_vector)

            if all(abs(item) <= 1e-1 for item in violation):
                # if no violation, stop
                break

            # update direction
            direction_vector += alpha * violation

        return direction_vector

    def generate_column(self, block_id, direction, heuristic=True,
                        approx_solver=False, x_k=None):
        """Generates the inner point (and corresponding column) either
        with MINLP sub-problem or with NLP sub-problem (too heuristically);
        adds valid local linear cut to heu_oa_master_problem if any

        :param block_id: Block identifier
        :type block_id: int
        :param direction: Direction in image space
        :type direction: ndarray
        :param heuristic: Indicates if the sub-problem must be solved \
        heuristically
        :type heuristic: bool
        :param approx_solver: enables approximate solving of subproblems in \
        column generation
        :type approx_solver: bool
        :param x_k: start_point for solving subproblems
        :type x_k: BlockVector or None
        :return: Inner point, primal bound, reduced cost of new point, \
        bool value indicating whether new point is generated and \
        corresponding column
        :rtype: tuple
        """
        if approx_solver:  # approximate solve minlp subproblem
            feasible_point, primal_bound, reduced_cost, is_new_point, \
             column = self.local_solve_subproblem(
                block_id, direction, x_k=x_k)
        else:
            if self.settings.cg_generate_columns_with_nlp is True:
                feasible_point, primal_bound, reduced_cost, is_new_point, \
                 column = self.local_solve_subproblem(block_id,
                                                      direction,
                                                      x_k=x_k)
                if reduced_cost > -0.01:
                    feasible_point, reduced_cost, primal_bound, _, \
                     is_new_point, column = \
                     self.global_solve_subproblem(
                         block_id, direction, heuristic=heuristic)

            else:
                feasible_point, reduced_cost, primal_bound, _, is_new_point, \
                 column = self.global_solve_subproblem(
                    block_id, direction, heuristic=heuristic)

        reduced_cost = round(reduced_cost, 3)
        dir_orig_space = \
            self.problem.block_model.trans_into_orig_space(block_id,
                                                           direction)
        #bin_index = self.problem.binary_index[block_id]
        # store data for the ML-Model
        #self.problem.training_data(block_id, dir_orig_space, feasible_point)
        # check, if get_size_training_data method works
        #len_data = self.problem.get_size_training_data(block_id)
        #print('length data: ', len_data, 'in block:', block_id)
        #self.ndata[block_id] = len_data

        return feasible_point, primal_bound, reduced_cost, is_new_point, column

    def global_solve_subproblem(self, block_id,
                                dir_im_space,
                                compute_reduced_cost=True,
                                heuristic=True):
        """Solves subproblem, adds inner point, \
        (either in compact form or in the original) and computes reduced cost \
        for the new inner point

        :param block_id: Block identifier
        :type block_id: int
        :param dir_im_space: Direction in image space
        :type dir_im_space: ndarray
        :param compute_reduced_cost: Indicates if reduced cost has to be \
        computed
        :type compute_reduced_cost: bool
        :param heuristic: Indicates if the sub-problem must be solved \
        heuristically
        :type heuristic: bool
        :return: Inner point (feasible point), reduced cost value, \
        primal bound of the sub-problem, dual bound of the sub-problem, \
        bool value indicating whether new inner point was generated, \
        corresponding column to the inner point
        :rtype: tuple
        """
        self.result.cg_num_minlp_problems += 1

        # can be None in case of initialization of the inner master problem
        reduced_cost = None
        is_new_point = False

        # transform into original space the direction
        dir_orig_space = \
            self.problem.block_model.trans_into_orig_space(block_id,
                                                           dir_im_space)

        # solve the sub-problem
        feasible_point, primal_bound, dual_bound, _ = \
            self.problem.sub_problems[block_id].global_solve(
                direction=dir_orig_space, result=self.result)



        column = None
        if compute_reduced_cost is True:
            # compute reduced cost

            # get the best inner point with given direction
            best_point, best_obj_val = self.problem.get_min_inner_point(
                block_id, dir_orig_space)
            reduced_cost = primal_bound - best_obj_val  # absolute reduced cost

            if reduced_cost < 0:
                # add the inner point with negative absolute reduced cost
                is_new_point, _, column = \
                    self.problem.add_inner_point(block_id, feasible_point)
        else:
            is_new_point, _, column = \
                self.problem.add_inner_point(
                    block_id, feasible_point,
                    self.settings.cg_min_inner_point_distance)

        if column is None:
            column = self.problem.block_model.trans_into_im_space(
                block_id, feasible_point)

        # check if subproblem is solved to optimality
        gap = abs(primal_bound - dual_bound) / (
                max(abs(primal_bound), abs(dual_bound)) + 1e-5)
        if gap > 0.0001:
            self.result.optimal_subproblems = False

        return feasible_point, reduced_cost, primal_bound, dual_bound, \
            is_new_point, column

    def get_slack_directions(self, slacks):
        """Computes new direction based on the slack values of IA master problem

        :param slacks: Slack values stored as a list of tuples
        :type slacks: list
        :return: New direction in image space
        :rtype: ndarray
        """

        # get maximum slack value
        max_slack_value = max(max(item) for item in slacks)

        direction_image_space = np.zeros(
            shape=self.problem.block_model.cuts.num_of_global_cuts + 1)
        for m, slack in enumerate(slacks):
            # check if slacks are zero
            if any(item != 0 for item in slack) and \
                    max(slack) > 0.1 * max_slack_value:
                direction_image_space[m + 1] = 1

        return direction_image_space

    def local_solve_subproblem(self, block_id, dir_im_space, x_k=None):
        """Solves MINLP subproblem approximately, adds inner point \
        (either in compact form or in the original) and computes reduced cost
        for the new inner point

        :param block_id: Block identifier
        :type block_id: int
        :param dir_im_space: Direction in image space
        :type dir_im_space: ndarray
        :param x_k: start_point for solving subproblems
        :type x_k: BlockVector or None
        :return: Inner point (feasible point), reduced cost value, \
        primal bound of the sub-problem, dual bound of the sub-problem, \
        bool value indicating whether new inner point was generated, \
        corresponding column to the inner point
        :rtype: tuple
        """
        self.result.cg_num_unfixed_nlp_problems += 1

        dir_orig_space = self.problem.block_model.trans_into_orig_space(
            block_id, dir_im_space)

        if x_k is not None:
            best_point = x_k
            best_point_obj_val = np.dot(dir_orig_space, best_point)
        else:
            best_point, best_point_obj_val = self.problem.get_min_inner_point(
                block_id, dir_orig_space)

        # solve the sub-problem
        tilde_y, new_point_obj_val, _, sol_is_feasible = \
            self.problem.sub_problems[block_id].local_solve(
                direction=dir_orig_space, result=self.result)

        reduced_cost = 0
        column = None
        is_new_point = False
        if sol_is_feasible is True:
            reduced_cost = (new_point_obj_val - best_point_obj_val) / (
                    max(abs(new_point_obj_val),
                        abs(best_point_obj_val)) + 1e-5)
            is_new_point, _, column = \
                self.problem.add_inner_point(
                    block_id, tilde_y, min_inner_point_dist=
                    self.settings.cg_min_inner_point_distance)
        else:
            tilde_y = None
            new_point_obj_val = None

        return tilde_y, new_point_obj_val, reduced_cost, is_new_point, column

    def find_solution_init(self, iter_index=None):
        """Method to generate a feasible solution and therefore reducing
        the slacks to zero in innerMaster problem

        :param iter_index: number of main iteration when the method is called
        :type: int
        """
        logger.info('\n=======================================================')
        logger.info('Find solution - init')

        # solve IA master problem
        _, _, w_ia, _, _, obj_value_ia = \
            self.problem.master_problems.solve_ia(self.settings.lp_solver)
        self.result.cg_relaxation = obj_value_ia

        self.problem.original_problem.local_solve_fast(w_ia, self.result,
                                                       self.problem,
                                                       iter_index)

    def find_solution(self, iter_index=None):
        """Primal heuristics method

        :param iter_index: number of main iteration when the method is called
        :type: int
        """
        logger.info('\n=======================================================')
        logger.info('Find solution - projection from ia solution - '
                    'local search')

        _, tilde_y, _, _, _, obj_value_ia = \
            self.problem.master_problems.solve_ia(
                self.settings.lp_solver)  # solve the inner problem
        self.result.cg_relaxation = obj_value_ia

        self.problem.original_problem.local_solve(tilde_y, self.result,
                                                  self.problem, iter_index)

    def print_z_values(self, z):
        """Prints sorted z-weights blockwise

        :param z: Weight vector
        :type z: BlockVector
        """
        non_zero_z = {}
        for k in range(self.problem.block_model.num_blocks):
            non_zero_z[k] = []
            for i, item in enumerate(z.get_block(k)):
                if item > 0:
                    non_zero_z[k].append((i, item))
            non_zero_z[k].sort(key=lambda x: x[1], reverse=True)

        logger.info('z values')
        logger.info(
            'Pairs (index z_k values), such that z_k > 0, sorted by z_k values')
        for k in range(self.problem.block_model.num_blocks):
            logger.info('block {0}: {1}'.format(k, non_zero_z[k]))

    def init_ML(self, block_id, train_set):
        training_data = self.problem.get_training_data()
        phase = self.phase_list[block_id]
        return self.problem.sub_problems[block_id].ml_sub_solver_init_train(block_id, training_data, phase, train_set)

    def update_Surrogate_Model(self, block_id):
        training_data = self.problem.get_training_data()
        return self.problem.sub_problems[block_id].ml_update(block_id, training_data)

    def test_ML(self, block_id):
        training_data = self.problem.get_training_data()
        return self.problem.sub_problems[block_id].ml_sub_solver_test_init_train(block_id, training_data)

    def ml_sub_solve(self, block_id, dir_im_space):
        direction = self.problem.block_model.trans_into_orig_space(block_id, dir_im_space)
        return self.problem.sub_problems[block_id].ml_sub_solver(block_id, direction)

    def plot_train_data(self):
        """method for printing data (inputs & outputs)
        :param:
        """
        plot = True
        if plot:
            test = False
            training_data = self.problem.get_training_data()
            print('=====================Plotting====================')
            for k in range(self.problem.block_model.num_blocks):
                X, y = self.problem.sub_problems[k].split_data(k, training_data, test, shuffle_data=False)

                X_list = []
                i = 0
                for i in range(X.shape[1]):
                    X_list.append('d['+str(i)+']')
                    i += 1
                y_list = []
                i = 0
                for i in range(y.shape[1]):
                    y_list.append('y['+str(i)+']')
                    i += 1

                #training set
                ia_init = np.arange(self.phase_list[k][0])
                X_ia_init = np.take(X, ia_init, axis=0)
                y_ia_init = np.take(y, ia_init, axis=0)
                appcolgen = np.arange(self.phase_list[k][0], self.phase_list[k][1])
                X_appcolgen = np.take(X, appcolgen, axis=0)
                y_appcolgen = np.take(y, appcolgen, axis=0)
                fwcolgen = np.arange(self.phase_list[k][1], self.phase_list[k][2])
                X_fwcolgen = np.take(X, fwcolgen, axis=0)
                y_fwcolgen = np.take(y, fwcolgen, axis=0)

                #validation set
                val_set = np.arange(self.phase_list[k][2], X.shape[0])
                X_val = np.take(X, val_set, axis=0)
                y_val = np.take(y, val_set, axis=0)

                label = X_list + y_list
                X_app_fw = np.concatenate((X_appcolgen, X_fwcolgen), axis=0)
                y_app_fw = np.concatenate((y_appcolgen, y_fwcolgen), axis=0)
                #use appcolgen and fwcolgen phase for corr matrix
                #Xy = np.concatenate((X_app_fw, y_app_fw), axis=1)
                #use all data for corr matrix
                Xy = np.concatenate((X_app_fw, y_app_fw), axis=1)
                df = pd.DataFrame(Xy, columns=label)
                dfcorr = df.corr()

                plt.figure(figsize=(21, 14), dpi=200)
                ax = sns.heatmap(dfcorr, annot=True, center=0)
                ax.set_title('Block' + str(k))
                plt.savefig('Block' + str(k) + '_Heatmap')
                i = 0
                for i in range(X.shape[1]):
                    fig1, ax1 = plt.subplots(1, 1, figsize=(14, 10), dpi=200)
                    #ax1.scatter(linspace, X[:, i], label='d[' + str(i) + ']')
                    #ax1.plot(linspace, X[:, i], 'r--')
                    ax1.plot(ia_init, X_ia_init[:, i], 'b--')
                    ax1.scatter(ia_init, X_ia_init[:, i], color='magenta', label='ia_init')
                    ax1.plot(appcolgen, X_appcolgen[:, i], 'b--')
                    ax1.scatter(appcolgen, X_appcolgen[:, i], color='blue', label='approx colgen')
                    ax1.plot(fwcolgen, X_fwcolgen[:, i], 'b--', label='training set')
                    ax1.scatter(fwcolgen, X_fwcolgen[:, i], color='red', label='fwcolgen')
                    ax1.plot(val_set, X_val[:, i], 'r--', label='main')
                    #ax1.plot(val_set, X_val[:, i], 'g>', label='New Point')


                    plt.grid()
                    plt.legend()
                    plt.title('Block '+str(k))
                    ax1.set_xlabel('iterations')
                    ax1.set_ylabel('d['+str(i)+']')

                    plt.savefig('Block'+str(k)+'Direction_comp'+str(i))

                    #plt.show()
                for j in range(y.shape[1]):
                    fig2, ax1 = plt.subplots(1, 1, figsize=(14, 10),dpi=200)
                    #ax1.scatter(linspace, y[:, j],  label='y['+str(j)+']')
                    #ax1.plot(linspace, y[:, j], 'g--')
                    ax1.plot(ia_init, y_ia_init[:, j], 'b--')
                    ax1.scatter(ia_init, y_ia_init[:, j], color='magenta', label='ia_init')
                    ax1.plot(appcolgen, y_appcolgen[:, j], 'b--')
                    ax1.scatter(appcolgen, y_appcolgen[:, j], color='blue', label='approx colgen')
                    ax1.plot(fwcolgen, y_fwcolgen[:, j], 'b--', label='training set')
                    ax1.scatter(fwcolgen, y_fwcolgen[:, j], color='red', label='fwcolgen')
                    ax1.plot(val_set, y_val[:, j], 'r--', label='main')
                    #ax1.plot(val_set, y_val[:, j], 'g>', label='New Point')


                    plt.grid()
                    plt.legend()
                    plt.title('Block ' + str(k))
                    plt.xlabel('iterations')
                    plt.ylabel('y['+str(j)+']')
                    plt.savefig('Block'+str(k)+'Points_bin' + str(j))
                    #plt.show()

    def eval_prediction(self, block_id, dir_im_space, y_clf, x_ia, binary_index):
        '''
        method to call global solve is required after comparing to prediction

        :param: direction, new direction from LP-IA
        :type: ndarray
        :param: y_clf , predicted binaries from SG
        :type: ndarray
        :param: tdata, training data (corr. direction & points)
        :type: dict


        '''

        #append predictions  to plot list
        if self.predictions[block_id] is None:
            self.predictions[block_id] = y_clf.reshape(1, -1)
        else:
            self.predictions[block_id] = np.concatenate((self.predictions[block_id], y_clf.reshape(1, -1)), axis=0)
            #self.predictions[block_id].append((self.n_subproblems_main[block_id], y_clf))

        # get training data
        #training_data = self.problem.get_training_data()
        # transform direction from im_space to orig_space
        dir_orig = self.problem.block_model.trans_into_orig_space(block_id, dir_im_space)
        # make 2D-Array
        dir_orig_space = dir_orig.reshape(1, -1)

        # get previous directions by splitting in- and outputs
        #X, y = self.problem.sub_problems[block_id].split_data(block_id, training_data)
        X_train, y_train, X_test, y_test = self.problem.sub_problems[block_id].get_training_data(block_id)
        # scaler = self.problem.sub_problems[block_id].get_scaler
        # call anomaly detection (Scaling and PCA)
        T_tr, T_n, n_components = self.anomaly_detection(block_id, X_train, dir_orig_space)
        print('shape X_train', X_train.shape)
        print('shape T_tr', T_tr.shape)
        # hotellings_t2 test
        #alpha = 0.85
        alpha = self.alpha
        df = n_components
        t2_bool = self.hotellings_t2(block_id=block_id, T=T_tr, x_n=T_n, n_components=n_components, alpha=alpha, df=df)
        # SPE (Squared Prediction Error)
        spe_bools = self.spe(block_id=block_id, T=T_tr, t_n=T_n, n_components=n_components)
        if t2_bool:
            update_model=True
            print("pca >> t2 outlier")
            self.anomaly[block_id].append(self.n_subproblems_main[block_id])
        else:
            update_model=False
            '''
        for t2_outlier in t2_bools:
            if t2_outlier:
                update_model = True
                print("pca >> t2 outlier")
                self.anomaly[block_id].append(self.n_subproblems_main[block_id])
                break
            else:
                update_model = False
        '''
        # setting start vector for NLP Solver
        sol = x_ia[block_id, :]

        for n, idx in enumerate(binary_index):
            sol[idx] = y_clf[n]

        # local solve / new point = solution is feasible
        feasible_point, primal_bound, reduced_cost, is_new_point, column = self.local_solve_subproblem(block_id,
                                                                                                     dir_im_space,
                                                                                                     x_k=sol)
        min_point, min_obj_val = self.problem.get_min_inner_point(block_id, dir_orig_space)

        if is_new_point:

            print('New point: ', is_new_point)

            self.newpoints[block_id].append((self.n_subproblems_main[block_id], y_clf))
            # primal bound = new point obj val
            if primal_bound <= min_obj_val:
                update_model = False

        if update_model:

            print('Update Surrogate Model')

            # correction by global solver
            feasible_point, reduced_cost, primal_bound, _, \
                is_new_point, column = self.global_solve_subproblem(block_id, dir_im_space)

            # extract binaries
            y_corr = np.zeros((1, len(binary_index))).flatten()
            for k, index in enumerate(binary_index):
                y_corr[k] = feasible_point[index]
            self.corrections[block_id].append((self.n_subproblems_main[block_id], y_corr))

            # add correction to list of new points if it is a new point
            # only if global solver puts out a new point, the point will be added to training set
            if is_new_point:
                self.newpoints[block_id].append((self.n_subproblems_main[block_id], y_corr))

            # transform direction
            #dir_orig_space = \
             #   self.problem.block_model.trans_into_orig_space(block_id,
              #                                                 dir_im_space)
            # add training data to contatiner in approx_data
            self.problem.training_data(block_id, dir_orig, feasible_point)
            # add the most recent dir and point to Surrogate Model training data
            #self.problem.sub_problems[block_id].add_train_data(block_id, training_data=self.problem.get_training_data())
            # get new principal components with updated training data
            #X_train, _ = self.problem.sub_problems[block_id].get_training_data(block_id)
            #T_tr, _, _ = self.anomaly_detection(block_id, X_train, dir_orig_space)

        self.PC_tr[block_id] = T_tr
        return update_model, feasible_point, reduced_cost, primal_bound, is_new_point, column

    def hotellings_t2(self, block_id, T, x_n, n_components, alpha, df):
        '''
        :param direction: new direction (in original space)
        :type: ndarray
        :param T: Principal Components
        :type: ndarray
        :param df: degree of freedom
        :type: integer
        :param alpha: level of reliability
        :type: float
        :param n_components: number of components to be used in PCA
        :type: integer

        '''

        n_components = np.minimum(n_components, T.shape[1])

        T = T[:, 0:int(n_components)]

        y_n = x_n[:, 0:self.PC_main[block_id].shape[1]]
        ### new calculation
        nx, p = T.shape
        ny, _ = y_n.shape

        delta = np.mean(T, axis=0) - np.mean(y_n, axis=0)
        print('delta', delta)
        print('mean T', np.mean(T, axis=0))
        print('mean y_n', np.mean(y_n, axis=0))
        Sx = np.cov(T, rowvar=False)

        Sy = np.cov(y_n, rowvar=False)
        S_pooled = ((nx - 1) * Sx + (ny - 1) * Sy) / (nx + ny - 2)
        t2 = (nx * ny) / (nx + ny) * np.matmul(np.matmul(delta.transpose(), np.linalg.inv(S_pooled)), delta)


        #p_value_chi2 = chi2.ppf(q=1 - alpha, df=df)

        mean, var = np.mean(T), np.var(T)
        # score of training data
        T_score = (T - mean) ** 2 / var
        self.t_score[block_id] = T_score
        # score of the new direction normalized value of score
        y_score = (y_n - mean) ** 2 / var
        print('y_score',y_score)
        anomaly_score_threshold = stats.chi2.ppf(q=(1 - alpha), df=df)
        k = 0
        t2_bools = []

        #print('anomaly_score ', anomaly_score_theshold)
        for k in range(y_n.shape[0]):
            if y_score.flatten()[k] > anomaly_score_threshold:
                outlier = True
            else:
                outlier = False
            t2_bools.append(outlier)

        if self.y_score[block_id] is None:
            self.y_score[block_id] = y_score
        else:
            self.y_score[block_id] = np.concatenate((self.y_score[block_id], y_score), axis=0)

        self.a_threshold[block_id] = anomaly_score_threshold
        self.t2[block_id].append(t2)
        t2_bool = t2 >= anomaly_score_threshold
        print('T2: ',t2)
        print('p value chi2: ', anomaly_score_threshold)
        print('outlier: ', t2_bool)

        return t2_bool

    def spe(self, block_id, T, t_n, n_components):
        """ calculate spe (pca) and detect anomaly"""
        x_dim = T.shape[1]
        if n_components < x_dim:

            T_tilde = T[:, 0:self.PC_main[block_id].shape[1]]
            spe = np.sum(T_tilde * T_tilde, axis=1)
            spe2over3 = np.cbrt(spe * spe)
            mean, std = np.mean(spe2over3), np.std(spe2over3)
            spe_threshold = np.power(mean + std, 3)
            self.spe_threshold[block_id] = spe_threshold
            t_n_tilde = t_n[:, 0:self.PC_main[block_id].shape[1]]
            spe_n = np.sum(t_n_tilde * t_n_tilde, axis=1)

            spe_bool = spe_n > spe_threshold
            self.spe_main[block_id] = spe

            if self.spe_score[block_id] is None:
                self.spe_score[block_id] = spe_n.reshape(1, -1)
            else:
                self.spe_score[block_id] = np.concatenate((self.spe_score[block_id], spe_n.reshape(1, -1)), axis=0)
        else:
            spe_bool = False

        return spe_bool

    def anomaly_detection(self, block_id, X, dir_orig_space):
        #get scaler from surrogate model
        scale_dict = self.problem.sub_problems[block_id].get_scaler()
        #scaler = StandardScaler().fit(X)
        scaler = scale_dict[block_id]
        # scale all previous directions
        x_tr = scaler.transform(X)
        # scale new direction
        x_n = scaler.transform(dir_orig_space)
        # PCA init

        pca_model = {}
        pca_model[block_id] = PCA().fit(x_tr)
        # PCA all previous directions
        T_tr = pca_model[block_id].transform(x_tr)

        # PCA new direction
        T_n = pca_model[block_id].transform(x_n)

        explained_variance_ratio = pca_model[block_id].explained_variance_ratio_
        accumulated_var_ratio = explained_variance_ratio[0]
        ratio_threshold = 0.95
        for n in range(x_tr.shape[1]):
            if accumulated_var_ratio >= ratio_threshold:
                n_components = n + 1
                break
            else:
                accumulated_var_ratio += explained_variance_ratio[n + 1]

        if self.PC_main[block_id] is None:
            self.PC_main[block_id] = T_n[:, 0:n_components]
        else:
            self.PC_main[block_id] = np.concatenate((self.PC_main[block_id], T_n[:, 0:self.PC_main[block_id].shape[1]]), axis=0)

        return T_tr, T_n, n_components

    def write_text(self, block_id):

        current_time = time.ctime()
        path = r"C:\Users\Finn Swonke\Documents\HAW\Semester 3\MP\Project\decogo\tests\solver\refactory_colgen\Results.txt"

        with open(path, 'a') as f:
            if block_id == 0:
                f.write('\n')
                f.write('Loadcase: L4S4'+'\n')
                f.write('Time: ' + str(current_time)+'\n')
                f.write('CASE: 0/1/1'+'\n')
                f.write('Primal Bound: ' + str(self.result.primal_bound) + '\n')
                f.write('Dual Bound: ' + str(self.result.cg_relaxation) + '\n')
                f.write('Total number of columns: ' + str(self.result.total_number_columns) + '\n')
            f.write('Block: ' + str(block_id) + '\n')
            f.write('New Points: ' + str(len(self.newpoints[block_id])) + '\n')
            f.write('Predictions: ' + str(len(self.predictions[block_id])) + '\n')
            f.write('Corrections: ' + str(len(self.corrections[block_id])) + '\n')
            #f.write('Accuracy: ' + str(avg))

            f.write(""+'\n')
            f.close()

    def ML_ColGen(self, block_id, direction, heuristic=True,
                        approx_solver=False, x_ia=None):
        """Generates the inner point (and corresponding column) either
        with MINLP sub-problem or with NLP sub-problem (too heuristically);
        adds valid local linear cut to heu_oa_master_problem if any

        :param block_id: Block identifier
        :type block_id: int
        :param direction: Direction in image space
        :type direction: ndarray
        :param heuristic: Indicates if the sub-problem must be solved \
        heuristically
        :type heuristic: bool
        :param approx_solver: enables approximate solving of subproblems in \
        column generation
        :type approx_solver: bool
        :param x_k: start_point for solving subproblems
        :type x_k: BlockVector or None
        :return: Inner point, primal bound, reduced cost of new point, \
        bool value indicating whether new point is generated and \
        corresponding column
        :rtype: tuple
        """

        if self.result.main_iterations == 0:
            if approx_solver:  # approximate solve minlp subproblem
                feasible_point, primal_bound, reduced_cost, is_new_point, \
                 column = self.local_solve_subproblem(
                    block_id, direction, x_k=x_ia[block_id, :])
            else:
                if self.settings.cg_generate_columns_with_nlp is True:
                    feasible_point, primal_bound, reduced_cost, is_new_point, \
                     column = self.local_solve_subproblem(block_id,
                                                          direction,
                                                          x_k=x_ia[block_id, :])
                    if reduced_cost > -0.01:
                        feasible_point, reduced_cost, primal_bound, _, \
                         is_new_point, column = \
                         self.global_solve_subproblem(
                             block_id, direction, heuristic=heuristic)

                else:
                    feasible_point, reduced_cost, primal_bound, _, is_new_point, \
                     column = self.global_solve_subproblem(
                        block_id, direction, heuristic=heuristic)

            reduced_cost = round(reduced_cost, 3)

            dir_orig_space = \
                self.problem.block_model.trans_into_orig_space(block_id,
                                                               direction)

            # store data for the ML-Model
            self.problem.training_data(block_id, dir_orig_space, feasible_point)

            # check, if get_size_training_data method works
            len_data = self.problem.get_size_training_data(block_id)
            print('length data: ', len_data, 'in block:', block_id)

            self.ndata[block_id] = len_data
        else:

            # accumulating data from global solver as validation data

            if approx_solver:  # approximate solve minlp subproblem
                feasible_point, primal_bound, reduced_cost, is_new_point, \
                 column = self.local_solve_subproblem(
                    block_id, direction, x_k=x_ia[block_id, :])
            else:
                if self.settings.cg_generate_columns_with_nlp is True:
                    feasible_point, primal_bound, reduced_cost, is_new_point, \
                     column = self.local_solve_subproblem(block_id,
                                                          direction,
                                                          x_k=x_ia[block_id, :])
                    if reduced_cost > -0.01:
                        feasible_point, reduced_cost, primal_bound, _, \
                         is_new_point, column = \
                         self.global_solve_subproblem(
                             block_id, direction, heuristic=heuristic)

                else:
                    feasible_point, reduced_cost, primal_bound, _, is_new_point, \
                     column = self.global_solve_subproblem(
                        block_id, direction, heuristic=heuristic)

            reduced_cost = round(reduced_cost, 3)

            # get binaries from global solver

            bin_index = self.problem.sub_problems[block_id].binary_index[block_id]
            y_bin = []
            for i, index in enumerate(bin_index):
                y_bin.append(round(feasible_point[index], 1))
            self.y_main[block_id].append((self.n_subproblems_main[block_id], np.array(y_bin)))


            # collecting directions in orig space for plotting only
            dir_orig_space = \
                self.problem.block_model.trans_into_orig_space(block_id,
                                                               direction)
            self.X_main_plot[block_id].append((self.n_subproblems_main[block_id], dir_orig_space))
            # collect x_ia in main iteration

            self.n_subproblems_main[block_id] += 1
            # ml sub solve
            '''
            bin_pred, bin_index = self.ml_sub_solve(block_id, direction)
            
            update_model, _, _, _, is_new_point, _ = self.eval_prediction(block_id=block_id, dir_im_space=direction,
                                                                           y_clf=bin_pred,
                                                                           x_ia=x_ia,
                                                                           binary_index=bin_index)
            
            #if update_model:
                #self.init_ML(block_id)
                #self.update_Surrogate_Model(block_id)
            '''


        len_data = self.problem.get_size_training_data(block_id)
        self.X_main[block_id].append((len_data, direction))
        self.x_ia[block_id].append((len_data, x_ia))
        return feasible_point, primal_bound, reduced_cost, is_new_point, column

    def plot_main(self, block_id, exp_type):
        #if len(self.newpoints[block_id]) != 0:
        # init array with iteration steps

        # get number of binary variables
        bin_index = self.problem.sub_problems[block_id].binary_index[block_id]
        n_bins = len(bin_index)
        # get training data
        X_train, y_train, X_test, y_test = self.problem.sub_problems[block_id].get_training_data(block_id)
        n_train = X_train.shape[0]
        iter_train = np.arange(n_train)
        # initiate dict´s for storing components of points in lists
        y_np = {}
        y_pred = {}
        y_corr = {}
        y_val = {}
        X_val = {}
        for k in range(n_bins):
            y_np[k] = []
            y_pred[k] = []
            y_corr[k] = []
            y_val[k] = []

        k=0
        for k in range(X_train.shape[1]):
            X_val[k] = []
        k=0
        iter_npoint = []
        iter_pred = []
        iter_corr = []
        iter_val = []
        iter_anomaly = []

        for i in range(len(self.newpoints[block_id])):
            iter_npoint.append(self.newpoints[block_id][i][0])
            for k in range(n_bins):
                y_np[k].append(self.newpoints[block_id][i][1][k])
        i=0
        k=0
        for i in range(len(self.predictions[block_id])):
            iter_pred.append(self.predictions[block_id][i][0])
            for k in range(n_bins):
                y_pred[k].append(self.predictions[block_id][i][1][k])
        i=0
        k=0
        for i in range(len(self.corrections[block_id])):
            iter_corr.append(self.corrections[block_id][i][0])
            for k in range(n_bins):
                y_corr[k].append(self.corrections[block_id][i][1][k])
        k=0
        i=0
        for i in range(len(self.y_main[block_id])):
            iter_val.append(self.y_main[block_id][i][0])
            for k in range(n_bins):
                y_val[k].append(self.y_main[block_id][i][1][k])
        k=0
        i=0
        for i in range(len(self.anomaly[block_id])):
            iter_anomaly.append(self.anomaly[block_id][i])
        k=0
        i=0
        # get scaler from surrogate model
        scale_dict = self.problem.sub_problems[block_id].get_scaler()
        # scaler = StandardScaler().fit(X)
        scaler = scale_dict[block_id]

        for i in range(len(self.X_main_plot[block_id])):
            for k in range(self.X_main_plot[block_id][i][1].shape[0]):

                X_val[k].append(self.X_main[block_id][i][1][k])


        # scale all previous directions
        #x_tr = scaler.transform(X)

        i=0
        # get scores of training data
        T_scores = self.t_score[block_id]

        scores = []
        anomalies = []
        corr_true = []
        #print(y_val)
        #print(y_pred)
        for k in range(n_bins):
            positiv = 0
            anomaly_hit = 0
            corr_hit = 0
            for i in range(len(self.y_main[block_id])):
                if y_val[k][i] == y_pred[k][i]:
                    positiv += 1
                else:
                    # list with subprblems in which anomaly occured
                    if i in iter_anomaly:
                        anomaly_hit += 1
            i=0
            '''
            for idx, i in enumerate(iter_corr):
                
                # count corrections which are equal to validation
                if self.corrections[block_id][idx][0] == self.y_main[block_id][i][0]:

                    if y_corr[k][idx] == y_val[k][i]:
                        corr_hit += 1
                #i += 1

            corr_true.append(corr_hit)
            '''
            acc_s = round(positiv/len(y_val[k]), 2)
            acc_a = anomaly_hit
            scores.append(acc_s)
            anomalies.append(acc_a)
        train_scores = []
        print('y_train shape',y_train.shape)

        bin_preds = self.sub_solve_train(block_id)
        print('bin_pred shape', bin_preds.shape)
        for j in range(y_train.shape[1]):
            positiv = 0
            for i in range(y_train.shape[0]):
                if bin_preds[i, j] == y_train[i, j]:
                    positiv += 1

            acc_t = round(positiv/y_train.shape[0],2)
            train_scores.append(acc_t)


        if n_bins < 0:
            plt.figure(figsize=(12, 3), tight_layout=True)
        else:
            plt.figure(figsize=(12, 16), tight_layout=True)

        for i in range(n_bins):

            plt.subplot(n_bins + 1, 1, i + 1)
            plt.plot(np.array(iter_train), bin_preds[:, i], 'm*')
            plt.plot(np.array(iter_val)+n_train, np.array(y_val[i]), 'r--', label='Validation', alpha=0.6)
            plt.plot(np.array(iter_pred)+n_train, np.array(y_pred[i]), 'm*', label='Prediction')
            plt.vlines(np.array(iter_anomaly)+n_train, ymin=0, ymax=1, color='lime', alpha=0.5, label='Anomaly')
            #plt.plot(np.array(iter_npoint)+n_train, np.array(y_np[i]), 'g>', label='New Point', alpha=0.4)
            #plt.plot(np.array(iter_corr)+n_train, np.array(y_corr[i]), 'ro', label='Correction', alpha=0.4)
            plt.plot(iter_train, y_train[:, i], 'b--', label='Training')
            plt.ylim(ymin=-.08, ymax=1.08)
            plt.title('y[' + str(i) + ']; Acc_Val: ' + str(scores[i]) + ' Acc_Train: ' + str(train_scores[i]) +
                      '; ' + str(anomalies[i]) + '/'+ str(len(iter_anomaly)) + ' anomalies /w False Pred; '
                    'Train_Data: '+str(len(iter_train)) + ' Test_Data: '+str(len(iter_val))) #+
                      #str(corr_true[i])+'/'+str(len(iter_corr))+'Corrections right')
            plt.xlabel('subproblems')
            plt.grid()
        plt.legend(loc='upper left')#bbox_to_anchor=(0.5,-0.5))
        plt.savefig('eval_Block_['+str(block_id)+']_'+str(exp_type) +'.png', dpi=300)



        # plotting training data directions and points only for blocks with 1 binary variable
        plot = True
        if plot:
            if n_bins > 1:
                plt.figure(figsize=(40, 18), tight_layout=True)
            else:
                plt.figure(figsize=(12, 10), tight_layout=True)
            # directions (X)
            for i in range(X_train.shape[1]):
                ax = plt.subplot(X_train.shape[1] + 1, 2, 2 * i + 1)
                plt.grid()
                plt.plot(np.array(iter_train), X_train[:, i], 'b--', label='Training')
                ax.twinx()
                plt.plot(np.array(iter_val)+n_train, np.array(X_val[i]), 'r--', label='Validation')
                #plt.vlines(np.array(iter_anomaly) + n_train, ymin=0, ymax=1, color='lime', alpha=0.5,
                           #label='Anomaly')
                #
                #plt.plot(np.array(iter_val)+n_train, np.array(self.y_score[block_id][:, i]), 'm+', label='max. y_score', alpha=0.5)
                #plt.plot(np.array(iter_val) + n_train, np.array(self.a_threshold[block_id]), 'k_', label='anomaly_threshold')
                plt.title('d[' + str(i) + ']')
                plt.xlabel('subproblems')
                #plt.yscale('log')

            plt.legend()
            # points
            for i in range(y_train.shape[1]):
                plt.subplot(y_train.shape[1] + 1, 2, 2 * i + 2)
                plt.plot(iter_train, y_train[:, i], 'b--', label='Training')
                plt.plot(np.array(iter_val)+n_train, np.array(y_val[i]), 'r--', label='Validation')
                plt.vlines(np.array(iter_anomaly) + n_train, ymin=0, ymax=1, color='lime', alpha=0.5,
                           label='Anomaly')
                plt.title('y[' + str(i) + ']')
                plt.xlabel('subproblems')
                plt.grid()
            plt.legend()
            plt.savefig('Block_[' + str(block_id) + '] Training_Data '+str(exp_type)+'.png', dpi=500)
        #'''
        plot = True
        if plot:

            n_components = self.PC_main[block_id].shape[1]
            plt.figure(figsize=(25, 6*n_components), tight_layout=True)

            n_samples = self.PC_main[block_id].shape[0]
            for i in range(n_components):

                ax = plt.subplot(n_components, 1, i+1)
                plt.plot(iter_train, self.PC_tr[block_id][:, i], 'b--', label='PC_train')
                plt.plot(iter_train, T_scores[:, i], 'm+')
                plt.plot(np.array(iter_val) + n_train, self.y_score[block_id][:, i], 'm+', label='y_score')
                #plt.plot(np.array(iter_val)+n_train, self.spe_score[block_id], 'kx', label='SPE')
                #plt.plot(iter_train, self.spe_main[block_id], 'kx')
                plt.hlines(y=self.a_threshold[block_id], xmin=0, xmax=n_train+len(iter_val), label='T² threshold', color='teal', alpha=0.5)
                #plt.hlines(y=self.spe_threshold[block_id], xmin=0, xmax=n_train + len(iter_val), label='SPE threshold',
                 #          color='darkorange', alpha=0.5)
                plt.grid()
                plt.xlabel('subproblems')
                ax2 = ax.twinx()
                plt.plot(np.array(iter_val)+n_train, self.PC_main[block_id][:, i], 'r--', label='PC_'+str(i)+'_test')

                plt.title('Block_'+str(block_id)+' PC_' + str(i)+ 'Train_Data: '+str(len(iter_train)) + ' Test_Data: '+ str(len(iter_val)))

            ax.legend(loc='upper left')
            ax2.legend(loc='upper right')
            plt.savefig('Block'+str(block_id)+'PC_'+str(exp_type)+'.png', dpi=200)

    def ML_ColGen_test(self, block_id, direction, x_ia):
        '''
        :param: block_id
        :type: in
        :param: direction; direction image space
        :type: ndarray
        :param:
        :type:
        '''
        bin_pred, bin_index = self.ml_sub_solve(block_id, direction)

        update_model, feasible_point, reduced_cost, primal_bound, \
            is_new_point, column = self.eval_prediction(block_id=block_id, dir_im_space=direction,
                                                                      y_clf=bin_pred,
                                                                      x_ia=x_ia,
                                                                      binary_index=bin_index)
        update_model = False
        if update_model:
            #self.init_ML(block_id)
            self.update_Surrogate_Model(block_id)
        return feasible_point, primal_bound, reduced_cost, is_new_point, column

    def testing(self):
        # testing Surrogate Model for each block
        train_sets = [45] # train_set will be train_set += min(round(X.shape[0]*0.8, 1), len(bin_index)*5)
        alphas = [0.2]

        num_blocks = self.problem.block_model.num_blocks

        num_blocks = 4
        # experiment loop train sets
        for train_set in train_sets:
            for block_id in range(num_blocks):
                bin_index = self.problem.sub_problems[block_id].binary_index[block_id]
                if len(bin_index) > 0:
                    # train model
                    self.init_ML(block_id, train_set)  # test size, random_seed
                    X_train, y_train, X_test, y_test = self.problem.sub_problems[block_id].get_training_data(block_id)
                    n_train = X_train.shape[0]
                    n_test = X_test.shape[0]
                    start = n_train
                    end = n_train+n_test

            # experiment loop alphas
            for alpha in alphas:
                # solving with SG Model
                for block_id in range(num_blocks):
                    bin_index = self.problem.sub_problems[block_id].binary_index[block_id]
                    if len(bin_index) > 0:
                        print('Block: ', block_id)
                        self.alpha = alpha

                        # ML_ColGen_test loop over number of directions of subproblems
                        for i in range(n_train, n_train+n_test):
                        #for i in range(len(self.X_main[block_id])):
                            self.n_subproblems_main[block_id] = i

                            dir_im_space = self.X_main[block_id][i][1]
                            #dir_im_space = X_test[i, :]
                            x_ia = self.x_ia[block_id][i][1]

                            self.ML_ColGen_test(block_id, dir_im_space, x_ia)
                        # plot_main
                        #self.plot_main(block_id, exp_type=train_set)
                        self.plot(block_id, alpha)
                        #self.plot_PC(block_id)
                        self.plot_t2(block_id, alpha)
                        # clear lists
                        self.predictions[block_id] = None
                        self.newpoints[block_id] = []
                        self.corrections[block_id] = []
                        self.anomaly[block_id] = []


                        self.n_subproblems_main[block_id] = 1
                        self.y_score[block_id] = None
                        self.a_threshold[block_id] = None
                        self.spe_theshold[block_id] = None
                        self.PC_main[block_id] = None
                        self.PC_tr[block_id] = None
                        self.t_score[block_id] = None
                        self.spe_score[block_id] = None
                        self.spe_main[block_id] = None
                        self.t2[block_id] = []

    def sub_solve_train(self, block_id, X_train):
        #X_train, y_train, X_test, y_test = self.problem.sub_problems[block_id].get_training_data(block_id)
        bin_preds = None
        for i in range(X_train.shape[0]):
            direction = X_train[i, :]
            bin_pred, bin_index = self.problem.sub_problems[block_id].ml_sub_solver(block_id, direction)
            if bin_preds is None:
                bin_preds = bin_pred.reshape(1, -1)
            else:
                bin_preds = np.concatenate((bin_preds, bin_pred.reshape(1, -1)), axis=0)

        return bin_preds

    def plot(self, block_id, alpha):

        X_train, y_train, X_test, y_test = self.problem.sub_problems[block_id].get_training_data(block_id)
        bin_preds_train = self.sub_solve_train(block_id, X_train)
        bin_preds_test = self.predictions[block_id]
        anomalies = np.array(self.anomaly[block_id])
        print('anomalies', anomalies)
        subproblems_train = np.arange(X_train.shape[0])
        subproblems_test = np.arange(X_test.shape[0])
        train_scores = []
        test_scores = []
        for j in range(y_train.shape[1]):
            positiv = 0
            for i in range(y_train.shape[0]):
                if bin_preds_train[i, j] == y_train[i, j]:
                    positiv += 1
            acc_t = round(positiv/y_train.shape[0], 2)
            train_scores.append(acc_t)
            positiv=0
            for i in range(y_test.shape[0]):
                if bin_preds_test[i, j] == y_test[i, j]:
                    positiv += 1
            acc_t = round(positiv / y_test.shape[0], 2)
            test_scores.append(acc_t)

        # Plot input output of training data + predictions and anomalies
        plt.figure(figsize=(25, 16), tight_layout=True)
        orig = False
        if orig:
            #Plot original input
            for i in range(X_train.shape[1]):
                plt.subplot(X_train.shape[1]+1, 2, 2*i + 1)
                plt.plot(subproblems_train, X_train[:, i], 'r--', label='Train_Set')
                plt.plot(subproblems_test+X_train.shape[0], X_test[:, i], 'b--', label='Test_Set')
                plt.grid()
                plt.title('d['+str(i)+']')
                plt.xlabel('subproblems')
                plt.ylabel('d[' + str(i)+']')
            plt.legend()

        else:
        #plot principal components
            n_components = self.PC_main[block_id].shape[1]
            for i in range(n_components):
                ax = plt.subplot(n_components+1, 2, 2*i + 1)
                plt.plot(subproblems_train, self.PC_tr[block_id][:, i], 'b--', label='train_set')
                plt.plot(subproblems_test+X_train.shape[0], self.PC_main[block_id][:, i], 'r--', label='test set')
                plt.plot(subproblems_train, self.PC_tr[block_id][:, i], 'bd')
                plt.plot(subproblems_test + X_train.shape[0], self.PC_main[block_id][:, i], 'rd')
                plt.grid()
                plt.title('PC '+str(i))
                plt.xlabel('subproblems')
                plt.ylabel('PC'+str(i))
            plt.legend()

        for i in range(y_test.shape[1]):
            plt.subplot(X_test.shape[1] + 1, 2, 2 * i + 2)
            plt.plot(subproblems_train, y_train[:, i], 'r--')
            plt.plot(subproblems_test + X_train.shape[0], y_test[:, i], 'b--')
            plt.plot(subproblems_train, bin_preds_train[:, i], 'm*', label='Prediction')
            plt.plot(subproblems_test + X_train.shape[0], bin_preds_test[:, i], 'm*')
            plt.vlines(anomalies, ymin=0, ymax=1, color='black', label='Anomaly')
            plt.grid()
            plt.xlabel('subproblems')
            plt.ylabel('y[' + str(i) + ']')
            plt.title('y[' + str(i) + ']; Training: '+str(train_scores[i])+ '; Testing: '+str(test_scores[i]))
        plt.legend()
        plt.savefig('PreMain_train_test Block ' + str(block_id) + 'alpha=' + str(alpha) + '.png', dpi=300)

    def plot_PC(self, block_id):
        plt.figure(figsize=(6, 6), tight_layout=True)
        ax = plt.subplot(1, 1, 1)
        plt.plot(self.PC_tr[block_id][:, 0], self.PC_tr[block_id][:, 1], 'r*', label='PC_train')
        plt.plot(self.PC_main[block_id][:, 0], self.PC_main[block_id][:, 1], 'b*', label='PC_test')

        plt.grid()
        plt.xlabel('PC 1')
        plt.ylabel('PC 2')
        plt.legend()
        plt.savefig('PrincipalComponents Block ' + str(block_id) + '.png', dpi=300)

    def plot_t2(self, block_id, alpha):
        a = [0.05,0.15,0.25,0.35,0.45,0.55]
        p_values = []
        colors = pl.cm.gist_rainbow(np.linspace(0, 1, len(a)))
        for idx, value in enumerate(a):
            p_values.append(stats.chi2.ppf(q=(1 - value), df=2))
        X_train, y_train, X_test, y_test = self.problem.sub_problems[block_id].get_training_data(block_id)
        subproblems_test = np.arange(X_test.shape[0])
        subproblems_train = np.arange(X_train.shape[0])
        plt.figure(figsize=(8, 5), tight_layout=True)
        plt.plot(subproblems_test+X_train.shape[0], self.t2[block_id], 'b--', label='T2')
        plt.plot(subproblems_test + X_train.shape[0], self.t2[block_id], 'bd',)
        for idx, p_value in enumerate(p_values):
            plt.hlines(p_value, xmin=X_train.shape[0], xmax=X_train.shape[0]+X_test.shape[0], color=colors[idx], label='P-Value Chi2; alpha='+str(a[idx]))
        plt.hlines(self.a_threshold[block_id], xmin=X_train.shape[0], xmax=X_train.shape[0]+X_test.shape[0],
                   label='P-Value Chi2; alpha:'+str(alpha), color='black', linestyles='dashed')
        plt.grid()
        plt.legend()
        plt.xlabel('subproblems')
        plt.ylabel('T2')
        plt.savefig('T2_pValueChi2 Block' + str(block_id) + 'alpha'+str(alpha) + '.png', dpi=300)

    def plot_directions(self, block_id):
        plt.fig