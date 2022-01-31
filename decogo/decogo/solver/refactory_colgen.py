"""Class which implements Column Generation with user-defined input model"""

import copy
import logging
import math
import time

import numpy as np
import matplotlib.pyplot as plt
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

        self.tdata = {}
        self.phase_list = {}
        #initiation of lists for phase list
        for k in range(self.problem.block_model.num_blocks):
            self.phase_list[k] = []
        self.ndata = {}

    def solve(self):
        """
        Inner approximation algorithm
        """

        self.ia_init()
        print('=======================')
        print('IA_init done')
        print('=======================')
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
        while True:
            print('=====================================================')
            print('                 main iteration                      ')
            print('=====================================================')
            self.result.main_iterations += 1
            print('iteration', self.result.main_iterations)
            print('=====================Plotting====================')
            #self.plot_train_data(1, self.tdata)
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
            point_list = {}
            pred_list = {}
            for k in range(self.problem.block_model.num_blocks):
                point_list[k] = []
                pred_list[k] = []
                if self.problem.block_model.sub_models[k].linear is False:
                    fpoint, _, delta_k, new_point, _ = \
                        self.generate_column(k, reduced_cost_direction)
                    if self.result.main_iterations == 2:
                        self.init_ML(k)
                    if self.result.main_iterations >= 3:
                        #pred, y_test = self.test_ML(k, training_data)
                        #dir= direction (duals from LP_IA Masterproblem (orig space))
                        dir = self.problem.block_model.trans_into_orig_space(k, reduced_cost_direction)
                        bin_pred, bin_glob, bin_index = self.ml_sub_solve(k, dir, fpoint, x_ia, self.result)
                        update_model= self.eval_prediction(block_id=k, dir_im_space=reduced_cost_direction, y_clf=bin_pred, x_ia=x_ia
                                                           , binary_index=bin_index)
                        #print('feasbile point from subsolver')
                        #print(fpoint)
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
                            fpoint, _, delta_k, new_point, _ = \
                                self.generate_column(k,
                                                     reduced_cost_direction,
                                                     heuristic=False)
                            if self.result.main_iterations == 2:
                                self.init_ML(k)
                            if self.result.main_iterations >= 3:
                                #pred, y_test = self.test_ML(k, training_data)
                                dir = self.problem.block_model.trans_into_orig_space(k, reduced_cost_direction)

                                bin_pred, bin_glob, bin_index = self.ml_sub_solve(k, dir, fpoint, x_ia, self.result)
                                update_model = self.eval_prediction(block_id=k, dir_im_space=reduced_cost_direction, y_clf=bin_pred,
                                                                    x_ia=x_ia,
                                                                    binary_index=bin_index)
                                #print('feasbile point from subsolver')
                                #print(fpoint)
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
        print('====================================================')
        print('phase_list')
        print(self.phase_list)
        print('====================================================')

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
                _, _, reduced_cost_list[k], new_point, _ = \
                    self.generate_column(k, reduced_cost_direction,
                                         approx_solver=approx_solver,
                                         x_k=x_ia.get_block(k))
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
                        _, _, _, new_point, _ = \
                            self.generate_column(k, slack_direction,
                                                 approx_solver=approx_solver,
                                                 x_k=x_ia.get_block(k))
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
                    _, _, reduced_cost_list[k], new_point, r_k = \
                        self.generate_column(k, direction)
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
                #logger.info('Training Data {0} Block {1}, reducedcost > -0.01'.format(len(data), block_id))

            else:
                feasible_point, reduced_cost, primal_bound, _, is_new_point, \
                 column = self.global_solve_subproblem(
                    block_id, direction, heuristic=heuristic)

        reduced_cost = round(reduced_cost, 3)

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

        #store data for the ML-Model
        self.problem.training_data(block_id, dir_orig_space, feasible_point)

        #check, if get_size_training_data method works
        len_data = self.problem.get_size_training_data(block_id)
        print('length data: ',len_data, 'in block:',block_id)
        self.ndata[block_id] = len_data

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

    def init_ML(self, block_id):
        training_data = self.problem.get_training_data()
        return self.problem.sub_problems[block_id].ml_sub_solver_init_train(block_id, training_data)

    def test_ML(self, block_id, training_data):
        return self.problem.sub_problems[block_id].ml_sub_solver_test_init_train(block_id, training_data)

    def ml_sub_solve(self, block_id, direction, point, x_ia, result):

        return self.problem.sub_problems[block_id].ml_sub_solver(block_id, direction, point, x_ia, result)

    def plot_train_data(self, block_id, training_data):
        """method for printing data (inputs & outputs)
        :param:
        """
        test = False

        for k in range(self.problem.block_model.num_blocks):
            X, y = self.problem.sub_problems[k].split_data(k, training_data, test)
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
            label = X_list+y_list #X_list.extend(y_list)
            Xy = np.concatenate((X, y), axis=1)
            df = pd.DataFrame(Xy, columns=label)
            dfcorr = df.corr()
            #print('correlation matrix')
            #print(dfcorr)
            plt.figure(figsize=(12, 8))
            ax = sns.heatmap(dfcorr)
            ax.set_title('Block'+str(k))
            plt.savefig('Block'+str(k)+'_Heatmap')
            linspace = np.linspace(0, X.shape[0], X.shape[0])
            ia_init = np.linspace(0, self.phase_list[k][0]-1, self.phase_list[k][0], dtype=int)
            appcolgen = np.linspace(self.phase_list[k][0], self.phase_list[k][1]-1, self.phase_list[k][1] - self.phase_list[k][0], dtype=int)
            fwcolgen = np.linspace(self.phase_list[k][1], self.phase_list[k][2]-1, self.phase_list[k][2] - self.phase_list[k][1], dtype=int)

            print('appcolgen', appcolgen.shape)
            print('X 0-1 ', X[self.phase_list[k][0]:self.phase_list[k][1],1].shape)
            print('fwcolgen: ', fwcolgen.shape)
            print(fwcolgen)
            print('X 1-2: ', X[self.phase_list[k][1]:self.phase_list[k][2]+1,1].shape)
            print(X[self.phase_list[k][1]:self.phase_list[k][2]+1,1])
            print('xshape ',X.shape)
            print('X:')
            print(X)
            i = 0
            for i in range(X.shape[1]):
                fig1, ax1 = plt.subplots(1, 1)
                #ax1.scatter(linspace, X[:, i], label='d[' + str(i) + ']')
                #ax1.plot(linspace, X[:, i], 'r--')
                ax1.plot(ia_init, X[0:self.phase_list[k][0], i], 'r--')
                ax1.scatter(ia_init, X[0:self.phase_list[k][0], i], color='green', label='ia_init')
                ax1.plot(appcolgen, X[self.phase_list[k][0]:self.phase_list[k][1],i], 'r--')
                ax1.scatter(appcolgen, X[self.phase_list[k][0]:self.phase_list[k][1], i], color='blue', label='approx colgen')
                ax1.plot(fwcolgen, X[self.phase_list[k][1]:self.phase_list[k][2]+1, i], 'r--')
                ax1.scatter(fwcolgen, X[self.phase_list[k][1]:self.phase_list[k][2]+1, i], color='red', label='fwcolgen')


                plt.grid()
                plt.legend()
                plt.title('Block '+str(k))
                ax1.set_xlabel('iterations')
                ax1.set_ylabel('d['+str(i)+']')

                plt.savefig('Block'+str(k)+'Direction_comp'+str(i))

                #plt.show()
            for j in range(y.shape[1]):
                fig2, ax1 = plt.subplots(1, 1)
                #ax1.scatter(linspace, y[:, j],  label='y['+str(j)+']')
                #ax1.plot(linspace, y[:, j], 'g--')
                ax1.plot(ia_init, y[0:self.phase_list[k][0],j], 'g--')
                ax1.scatter(ia_init, y[0:self.phase_list[k][0], j], color='green', label='ia_init')
                ax1.plot(appcolgen, y[self.phase_list[k][0]:self.phase_list[k][1],j], 'g--')
                ax1.scatter(appcolgen, y[self.phase_list[k][0]:self.phase_list[k][1], j], color='blue', label='approx colgen')
                ax1.plot(fwcolgen, y[self.phase_list[k][1]:self.phase_list[k][2]+1,j], 'g--')
                ax1.scatter(fwcolgen, y[self.phase_list[k][1]:self.phase_list[k][2]+1, j], color='red', label='fwcolgen')


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
        # get training data
        training_data = self.problem.get_training_data()
        # transform direction from im_space to orig_space
        dir_orig_space = self.problem.block_model.trans_into_orig_space(block_id, dir_im_space)
        # make 2D-Array
        dir_orig_space = dir_orig_space.reshape(1, -1)
        # get previous directions by splitting in- and outputs
        X, y = self.problem.sub_problems[block_id].split_data(block_id, training_data)
        print('last added direction')
        print(X[-1, :])
        print('X shape')
        print(X.shape)
        print('last added point')
        print(y[-1, :])
        print('y shape')
        print(y.shape)

        #scaler = self.problem.sub_problems[block_id].get_scaler
        # call anomaly detection (Scaling and PCA)
        T_tr, T_n, n_components = self.anomaly_detection(block_id, X, dir_orig_space)

        # hotellings_t2 test
        y_proba, y_score, t2_bools, param = self.hotellings_t2(T_tr, T_n, n_components)

        for t2_outlier in t2_bools:
            if t2_outlier:
                update_model = True
                print("pca >> t2 outlier, update_model = ", update_model)
                break
            else:
                update_model = False
        sol = x_ia[block_id, :]

        j = 0
        for idx in binary_index:
            sol[idx] = y_clf[j]
            j += 1
        tilde_y, new_point_obj_val, reduced_cost, is_new_point, column = self.local_solve_subproblem(block_id,
                                                                                                     dir_im_space,
                                                                                                     x_k=sol)
        min_point, min_obj_val = self.problem.get_min_inner_point(block_id, dir_orig_space)
        print('new_point_obj_val', new_point_obj_val)
        print('min_obj_val', min_obj_val)
        if is_new_point:
            print('=====================')
            print('is new point', is_new_point)
            print('=====================')
            #data = self.problem.training_data(block_id, dir_orig_space, tilde_y)
            #len_data = self.problem.get_size_training_data(block_id)
            if new_point_obj_val <= min_obj_val:
                update_model = False
        if update_model:
            print('=====================')
            print('model is updated with global solving')
            print('=====================')
            feasible_point, reduced_cost, primal_bound, dual_bound, \
            is_new_point, column = self.global_solve_subproblem(self, block_id,
                                    dir_im_space,
                                    compute_reduced_cost=True,
                                    heuristic=True)
            # append new direction & point
            self.problem.training_data(block_id, dir_orig_space, feasible_point)
        return update_model

    def hotellings_t2(self, X, x_n, df=3, n_components=5, alpha=0.05):
        '''
        :param direction: new direction (in original space)
        :type: ndarray
        :param x: all previous collected directions
        :type: ndarray
        :param df: degree of freedom
        :type: integer
        :param alpha: level of reliability
        :type: float
        :param n_components: number of components to be used in PCA
        :type: integer

        '''
        n_components = np.minimum(n_components, X.shape[1])
        X = X[:, 0:n_components]

        x_n = x_n[:, 0:n_components]
        y_n = x_n

        mean, var = np.mean(X), np.var(X)
        param = (mean, var)
        y_score = (y_n - mean) ** 2 / var
        y_proba = 1 - stats.chi2.cdf(y_score, df=df)

        anomaly_score_theshold = stats.chi2.ppf(q=(1 - alpha), df=df)
        k = 0
        t2_bools = []
        #print('y_score ',y_score)
        #print('anomaly_score ', anomaly_score_theshold)
        for k in range(y_n.shape[0]):
            if y_score.flatten()[k] > anomaly_score_theshold:
                t = True
            else:
                t = False
            t2_bools.append(t)


        return y_proba, y_score, t2_bools, param

    def anomaly_detection(self, block_id, X, dir_orig_space):
        scaler = StandardScaler().fit(X)
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
        return T_tr, T_n, n_components
