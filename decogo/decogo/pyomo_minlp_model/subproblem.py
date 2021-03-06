"""Implements and manages all sub-problems"""

import logging
from abc import ABC

import numpy as np
from pyomo.core import ConcreteModel, Param, RangeSet, Var, ConstraintList, \
    Objective, minimize, maximize
from pyomo.core.expr.visitor import identify_variables, replace_expressions
from pyomo.environ import Binary, Integers, Reals, SolverStatus, \
    TerminationCondition
from pyomo.opt import SolverFactory
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
from tensorflow.keras.layers import Dense, Input, Dropout
from tensorflow.keras.models import Sequential
import tensorflow as tf
import matplotlib.pyplot as plt

logger = logging.getLogger('decogo')


class PyomoSubProblemBase(ABC):
    """Base class for construction of Pyomo model. Here are implemented
    methods for creating local linear and nonlinear constraints

    :param block_model: Block model
    :type block_model: BlockModel
    :param block_id: Block identifier
    :type block_id: int
    :param model: Pyomo model
    :type model: ConcreteModel
    """

    def __init__(self, sub_models, cuts, block_id):
        """Constructor method"""
        super().__init__()
        self.block_id = block_id
        self.sub_model = sub_models[block_id]
        self.cuts = cuts

        self.model = ConcreteModel()

        self.model.Y = Var(RangeSet(cuts.block_sizes[block_id]))
        for n in range(1, self.sub_model.block_size + 1):
            lb = self.sub_model.variables[n - 1].lower_bound
            self.model.Y[n].value = lb

            self.model.Y[n].setlb(lb)

            ub = self.sub_model.variables[n - 1].upper_bound
            self.model.Y[n].setub(ub)

            if self.sub_model.variables[n - 1].type == "Binary":
                self.model.Y[n].domain = Binary
                self.model.Y[n].value = 0
            elif self.sub_model.variables[n - 1].type == "Integers":
                self.model.Y[n].domain = Integers
            else:
                self.model.Y[n].domain = Reals

        self.model.lin_con = ConstraintList()

        def lin_con_rule(model, k, m):
            return sum(cuts.local_cuts[k][m].lhs[self.block_id, n] *
                       model.Y[n + 1] for n in range(sub_models[k].block_size))

        for m, local_constr in enumerate(cuts.local_cuts[block_id]):
            if local_constr.relation == "<=":
                self.model.lin_con.add(expr=lin_con_rule(self.model, block_id,
                                                         m) <= local_constr.rhs)
            elif local_constr.relation == "=":
                self.model.lin_con.add(expr=lin_con_rule(self.model, block_id,
                                                         m) == local_constr.rhs)
            elif local_constr.relation == ">=":
                self.model.lin_con.add(expr=lin_con_rule(self.model, block_id,
                                                         m) >= local_constr.rhs)

        self.model.nonlin_constr = ConstraintList()
        for constr in self.sub_model.nonlin_constr:
            expr = constr.expr.clone()
            vars_in_expr = identify_variables(expr)
            substitution_map = {}  # map for replacing the objects in
            # the expression
            for var in vars_in_expr:
                index = self.sub_model.vars_in_block.index(
                    var.name)
                substitution_map[id(var)] = self.model.Y[index + 1]
            new_expr = replace_expressions(expr, substitution_map)
            self.model.nonlin_constr.add(expr=new_expr)

    def solve(self, solver_name, start_point=None, solver_options=None):
        """Base method for calling the external solver to solve the model

        :param solver_name: External solver name
        :type solver_name: str
        :param start_point: Starting point for the solver, defaults to ``None``
        :type start_point: ndaray or None
        :param solver_options: Options for external solver, defaults to ``None``
        :type solver_options: list
        :return: Solution point, primal bound, dual bound, flag if the primal \
        bound is feasible
        :rtype: tuple
        """

        if start_point is not None:

            for i in range(len(self.model.Y)):
                self.model.Y[i + 1].value = start_point[i]

        # solving the problem
        opt = SolverFactory(solver_name)

        # # set solver options
        # # options are pairs: (parameter name, value)
        # if solver_options is not None:
        #     for key, value in solver_options:
        #         opt.options[key] = value
        #
        # results = opt.solve(self.model, tee=False)

        def recursive_solve(opt_recursive, options=None, iter_i=10):
            solver_error = False
            try:
                if options is not None:
                    for key, value in solver_options:
                        opt_recursive.options[key] = value
                result_recursive = opt_recursive.solve(self.model, tee=False)
            except ValueError as error:
                logger.info(error)
                if iter_i - 2 < 6:
                    logger.info('error: cannot solve')
                    solver_error = True
                    result_recursive = None
                else:
                    logger.info('solve: max_iter = {0}'.format(iter_i - 2))
                    result_recursive, solver_error = \
                        recursive_solve(opt_recursive, options=('max_iter',
                                                                iter_i - 2),
                                        iter_i=iter_i - 2)
            return result_recursive, solver_error

        results, error_in_solver = recursive_solve(opt, solver_options)

        sol_is_feasible = True
        dual_bound = None
        primal_bound = None
        y_new = np.zeros(shape=len(self.model.Y))
        if error_in_solver:
            sol_is_feasible = False
        else:
            if results.solver.status == SolverStatus.ok:
                sol_is_feasible = True
            elif results.solver.status == SolverStatus.warning:
                if results.solver.termination_condition == \
                        TerminationCondition.infeasible:
                    sol_is_feasible = False
                    logger.info(self.model.name + ' is infeasible')
                elif results.solver.termination_condition == \
                        TerminationCondition.maxIterations:
                    sol_is_feasible = False

            for i in range(len(self.model.Y)):
                y_new[i] = self.model.Y[i + 1].value

            primal_bound = self.model.obj.expr()

            # by default dual bound is equal to primal bound
            dual_bound = primal_bound
            if solver_name == 'scip' and solver_options is not None:
                log_lines = opt._log.split('\n')
                for line in log_lines:
                    if line.startswith('Dual Bound'):
                        dual_bound = float(line.split(':')[1].strip())

        return y_new, primal_bound, dual_bound, sol_is_feasible

    def add_linear_constraint(self):
        """Adds new linear constraint"""

        def lin_con_rule(model, constr):
            return sum(constr.lhs[self.block_id, n] * model.Y[n + 1]
                       for n in
                       range(self.cuts.block_sizes[self.block_id]))

        constr = self.cuts.local_cuts[self.block_id][
            -1]  # this takes the last one
        if constr.relation == "<=":
            self.model.lin_con.add(
                expr=lin_con_rule(self.model, constr) <= constr.rhs)
        elif constr.relation == "=":
            self.model.lin_con.add(
                expr=lin_con_rule(self.model, constr) == constr.rhs)
        elif constr.relation == ">=":
            self.model.lin_con.add(
                expr=lin_con_rule(self.model, constr) >= constr.rhs)

    def set_nlp(self):
        """Relaxes integer varaibles to continious"""
        for i, var_domain in enumerate(
                self.sub_model.variables):
            if var_domain.type == "Integers":
                self.model.Y[i + 1].domain = Reals
            elif var_domain.type == "Binary":
                self.model.Y[i + 1].domain = Reals

    def unset_nlp(self):
        """Sets to integer type for the variables which are integer in the
        original formulation"""
        for i, var_domain in enumerate(
                self.sub_model.variables):
            if var_domain.type == "Integers":
                self.model.Y[i + 1].domain = Integers
            elif var_domain.type == "Binary":
                self.model.Y[i + 1].domain = Binary

    def update_var_lower_bound(self, index):
        """Updates the lower bound of the variable. The value is taken form
        the BlockModel

        :param index: Index (block_id, index)
        :type index: tuple
        """
        k, i = index
        self.model.Y[i + 1].setlb(
            self.sub_models[k].variables[i].lower_bound)

    def update_var_upper_bound(self, index):
        """Updates the upper bound of the variable. The value is taken form
        the BlockModel

        :param index: Index (block_id, index)
        :type index: tuple
        """
        k, i = index
        self.model.Y[i + 1].setub(
            self.sub_models[k].variables[i].upper_bound)

    def fix_integers(self, x):
        """Fixes integer variables with given value

        :param x: Given array
        :type x: ndarray
        """
        for i, var in enumerate(
                self.sub_model.variables):
            if var.type == 'Integer' or var.type == 'Binary':
                self.model.Y[i + 1].value = x[i]
                self.model.Y[i + 1].fix()

    def unfix_integers(self):
        """Unfixes all integer variables"""
        for i, var in enumerate(
                self.sub_model.variables):
            if var.type == 'Integers' or var.type == 'Binary':
                self.model.Y[i + 1].unfix()

    def add_objective(self):
        """Sets default objective function, i.e. :math:`c_k^Tx_k`, where
        :math:`c_k` - partial objective from the original model
        """

        def objective_rule(model):
            ans_obj = 0

            ans_obj += sum(
                self.cuts.obj.c[self.block_id, n] * model.Y[n + 1]
                for n in range(self.cuts.block_sizes[self.block_id]))

            ans_obj += self.cuts.obj.const

            return ans_obj

        self.model.del_component('obj')
        self.model.obj = Objective(rule=objective_rule, sense=minimize)

    def set_objective(self, direction):
        """Sets the objective function with given direction vector

        :param direction: Given vector
        :type direction: ndarray
        """

        new_obj_expr = sum(direction[n] * self.model.Y[n + 1]
                           for n in range(
            self.sub_model.block_size))

        self.model.del_component('obj')

        self.model.obj = Objective(expr=new_obj_expr)


class PyomoMinlpSubProblem(PyomoSubProblemBase):
    """Class for defining the following sub-problem

    .. math::
        \\begin{equation}
        \\begin{split}
            \\min \\ &d_k^Ty_k, \\newline
            &y_k \\in X_k, d_k \\in \\mathbb{R}^{n_k}
        \\end{split}
        \\end{equation}
    """

    def __init__(self, sub_models, cuts, block_id):
        """Constructor method"""
        super().__init__(sub_models, cuts, block_id)


class PyomoProjectionSubProblem(PyomoSubProblemBase):
    """Class for defining a projection sub-problem

    .. math::
        \\begin{equation}
        \\begin{split}
            \\min \\ &||y_k-x_k||^2,\\newline
            &y_k \\in G_k, x_k \\text{ is fixed}
        \\end{split}
        \\end{equation}
    """

    def __init__(self, sub_models, cuts, block_id):
        """Constructor method"""
        super().__init__(sub_models, cuts, block_id)

        self.model.X = Param(RangeSet(self.cuts.block_sizes[block_id]),
                             mutable=True, initialize=0)

        self.model.obj = Objective(expr=self.obj_rule())
        self.model.name = 'Projection sub problem'

    def obj_rule(self):
        """Defines the objective function

        :return: Objective expression
        :rtype: Expression
        """
        expr = sum(pow((self.model.Y[n + 1] - self.model.X[n + 1]), 2)
                   for n in
                   range(self.sub_model.block_size))

        return expr

    def set_projection_point(self, point_to_project):
        """Sets projection point, i.e. :math:`x_k`

        :param point_to_project: Projection point
        :type point_to_project: ndarray
        """
        for i in range(len(self.model.X)):
            self.model.X[i + 1].value = point_to_project[i]


class PyomoResourceProjectionSubProblem(PyomoSubProblemBase):
    """Class for defining a projection sub-problem

    .. math::
        \\begin{equation}
        \\begin{split}
            \\min \\ &||A_kx_k - w_k||^2,\\newline
            &y_k \\in G_k, w_k \\text{ is fixed}
        \\end{split}
        \\end{equation}
    """

    def __init__(self, sub_models, cuts, block_id):
        """Constructor method"""
        super().__init__(sub_models, cuts, block_id)

        self.model.w = Param(RangeSet(self.cuts.num_of_global_cuts + 1),
                             mutable=True, initialize=0)

        self.model.obj = Objective(expr=self.obj_rule())
        self.model.name = 'Resource projection sub problem'

    def obj_rule(self):
        """Defines the objective function

        :return: Objective expression
        :rtype: Expression
        """
        expr = 0
        for j in range(self.cuts.num_of_global_cuts + 1):
            if j == 0:
                expr += \
                    (sum(self.cuts.obj.c[self.block_id, i] * self.model.Y[i + 1]
                         for i in range(self.cuts.block_sizes[self.block_id])) -
                     self.model.w[j + 1]) ** 2
            else:
                expr += \
                    (sum(self.cuts.global_cuts[j - 1].lhs[self.block_id, i] *
                         self.model.Y[i + 1]
                    for i in range(self.cuts.block_sizes[self.block_id])) -
                         self.model.w[j + 1]) ** 2

        return expr

    def set_projection_point(self, point_to_project):
        """Sets projection point, i.e. :math:`w_k`

        :param point_to_project: Projection point
        :type point_to_project: ndarray
        """
        for i in range(len(self.model.w)):
            self.model.w[i + 1].value = point_to_project[i]


class PyomoLineSearchSubProblem(PyomoSubProblemBase):
    """Class defines line search sub-problem between exterior point
    :math:`x_k^{ext}` and interior point :math:`x_k^{int}`

    .. math::
        \\begin{equation}
        \\begin{split}
            \\max \\ & \\alpha, \\newline
            &y_k = \\alpha x_k^{ext} + (1 - \\alpha) x_k^{int}, \\newline
            &y_k \\in X_k
        \\end{split}
        \\end{equation}
    """

    def __init__(self, sub_models, cuts, block_id):
        super().__init__(sub_models, cuts, block_id)

        self.model.lin_con.deactivate()

        # variable for finding the point on the line between x_hat and x_star
        self.model.alpha = Var(bounds=(0, 1), initialize=0, within=Reals)

        # objective
        self.model.obj = Objective(expr=self.model.alpha, sense=maximize)

        self.model.interior_point = Param(
            RangeSet(self.cuts.block_sizes[block_id]), mutable=True,
            initialize=0)
        self.model.exterior_point = Param(
            RangeSet(self.cuts.block_sizes[block_id]), mutable=True,
            initialize=0)

        self.model.line_search_con = ConstraintList()
        for i in range(self.cuts.block_sizes[self.block_id]):
            self.model.line_search_con.add(
                self.model.Y[i + 1] == self.model.alpha *
                self.model.exterior_point[i + 1] +
                (1 - self.model.alpha) * self.model.interior_point[i + 1])

    def set_interior_exterior_points(self, exterior_point, interior_point):
        """Sets values of two endpoints

        :param exterior_point: Exterior point
        :type exterior_point: ndarray
        :param interior_point: Interior point
        :type interior_point: ndarray
        """
        for i in range(self.cuts.block_sizes[self.block_id]):
            self.model.exterior_point[i + 1].value = exterior_point[i]
            self.model.interior_point[i + 1].value = interior_point[i]

    def solve(self, solver_name, start_point=None, solver_options=None):
        """Solves the subproblem and gets value of :math:`\\alpha`. For more
        info, see base method :meth:`SubProblemBase.solve`
        """
        y_new, primal_bound, dual_bound, sol_is_feasible = \
            super().solve(solver_name, start_point=start_point,
                          solver_options=solver_options)

        alpha = self.model.alpha.value

        return alpha, y_new


class SurrogateModel:

    def __init__(self, block_id, binary_index):
        '''Constructor for the Surrogate Model
        :param:
        :type:
        :param:
        :type:
        :param:
        :type:
        '''
        self.block_id = block_id
        self.clf_batch = {}
        self.binary_index = binary_index
        self.scaler = {}
        self.test_split = 0.1
        self.X_train = {}
        self.y_train = {}
        self.X_validation = {}
        self.y_validation = {}

    def init_train(self, block_id, training_data, phase, split):
        '''Method for initial training of the Surrogate Model
        :param: block_id
        :type: int
        :param: training_data
        :type: dict -> includes a list of tuples with corresponding directions and points for each block
        '''
        bin_index = self.binary_index[block_id]
        if len(bin_index) > 0:
            print('============================================ ')
            print('             initiated training              ')
            print('============================================ ')

            # split training_data into input/output
            X, y = self.split_data(block_id, training_data, shuffle_data=False)

            # setting up hyperparameters for NN depending on number of binaries
            num_bin = min(30, len(bin_index)+1)

            epochs = min(500, 12*len(bin_index))
            if len(bin_index) > 1:
                loss = 'categorical_crossentropy'
                activation = 'sigmoid'
            else:
                loss = 'binary_crossentropy'
                activation = 'sigmoid'


            # seperate to train & test set (training set is last added data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=split, shuffle=False)

            X_val = X_test[:10, :]
            y_val = y_test[:10, :]
            self.X_validation[block_id] = X_test
            self.y_validation[block_id] = y_test

            self.clf_batch[block_id] = Sequential()

            # Input Layer
            self.clf_batch[block_id].add(Dense(X_train.shape[1], input_dim=X_train.shape[1]))

            # Hidden Layer
            self.clf_batch[block_id].add(Dense(num_bin, activation='sigmoid'))


            # Output Layer
            self.clf_batch[block_id].add(Dense(y_train.shape[1], activation=activation))

            # Settings
            self.clf_batch[block_id].compile(loss=loss, optimizer='adam',
                                             metrics=['acc'])

            #scale data (standardize)
            self.scaler[block_id] = StandardScaler().fit(X_train, y_train)
            X_tr_scaled = self.scaler[block_id].transform(X_train)
            X_val_scaled = self.scaler[block_id].transform(X_val)

            # add to attributes
            self.X_train[block_id] = X_train
            self.y_train[block_id] = y_train

            print('fit model to block:', block_id)
            model = self.clf_batch[block_id].fit(X_tr_scaled, y_train,
                                                 validation_data=(X_val_scaled, y_val), epochs=epochs, verbose=0)
            plot = True
            if plot:
            #Plot training progress
                plt.figure(figsize=(14, 5), dpi=200)

                plt.subplot(1, 2, 1)
                plt.title('Loss')
                plt.plot(model.history['loss'], label='Train')
                plt.plot(model.history['val_loss'], label='Validation')
                plt.legend()
                plt.xlabel('epochs')
                plt.grid()
                plt.subplot(1, 2, 2)
                plt.title('Accuracy')
                plt.plot(model.history['acc'], label='Train')
                plt.plot(model.history['val_acc'], label='Validation')
                plt.legend()
                plt.xlabel('epochs')
                #plt.ylim(0, 1)
                plt.grid()
                plt.savefig('Acc&Loss_block_'+str(block_id)+'test_set'+str(split)+'.png', format='png', dpi=200)
        else:
            print('no binary variables in block', block_id)

    def predict(self, block_id, X, y = None):
        '''Method for prediction of feasible points with Surrogate MOdel
        :param:
        :type:
        :param:
        :type:
        '''
        # transform input
        transformed_direction = self.scaler[block_id].transform(X)
        # predict method
        pred = self.clf_batch[block_id].predict(transformed_direction)
        # rounding
        prediction = np.around(pred)


        return prediction

    def test_init_train(self, block_id, training_data):
        '''
        :param: block_id
        :type: int
        :param: direction
        :type: ndarray
        :param: point
        :type: ndarray
        :returns: point
        :rtype: list
        '''
        print('============================================ ')
        print('            testing block', block_id, '      ')
        print('============================================ ')
        test = True
        X_test, y_test = self.split_data(block_id, training_data, test)
        index = self.binary_index[block_id]

        if len(index) > 0:
            prediction = self.predict(block_id, X_test, y_test)
            score = self.get_score(prediction, y_test)

            return prediction, y_test, score
        else:
            prediction = None
            y_test = None
            score = None

            return prediction, y_test, score

    def split_data(self, block_id, training_data, test=False, shuffle_data=True):
        """
        :param:
        """
        #list of binary indexes for a given block
        index = self.binary_index[block_id]

        blockdata = training_data[block_id]
        y = np.zeros((len(blockdata), len(index)))
        X = np.zeros((len(blockdata), len(blockdata[0][0])))
        if test == True:
            n_test = round(self.test_split*X.shape[0])
            n_train = X.shape[0]-n_test
        else:
            n_train = X.shape[0]
        for i in range(0, len(blockdata)):
            j = 0
            for idx in index:
                y[i, j] = round(blockdata[i][1][idx])
                j += 1
            for k in range(len(blockdata[0][0])):
                X[i, k] = blockdata[i][0][k]

        if test == False:
            X = X[0:n_train, :]
            y = y[0:n_train, :]
        elif test == True:
            X = X[n_train:X.shape[0], :]
            y = y[n_train:y.shape[0], :]

        if shuffle_data:
            X, y = shuffle(X, y, random_state=42)

        return X, y

    def sub_solve(self, block_id, direction):
        """ takes direction to predict binaries; takes point from for comparing prediction & solution
        :param block_id: Block Identifier
        :type block_id:int
        :param: direction
        :type: direction, ndarray

        """
        print('============================================ ')
        print('    sub solve block', block_id, '            ')
        print('============================================ ')

        index = self.binary_index[block_id]
        pred = self.predict(block_id, np.array([direction]))
        bin_pred = pred.flatten()

        return bin_pred, index

    def get_score(self, prediction, test):
        '''computes a average score of the predictions
        :param
        '''
        l = []
        for i in range(prediction.shape[0]):
            positiv = 0
            for j in range(prediction.shape[1]):
                if prediction[i, j] == test[i, j]:
                    positiv += 1
            if prediction.shape[1] > 0:
                score = positiv / prediction.shape[1]
                l.append(score)
        score_arr = np.array(l)
        score_mean = np.mean(score_arr)
        score_var = np.var(score_arr)
        return score_mean, score_var

    def update_model(self, block_id, dir_orig_space, feasible_point):
        ''' Update Surrogate Model with new data only
        :param block_id, Block Identifier
        :type block_id: int
        :param dir_orig_space: direction in original space
        :type dir_orig_space: ndarray
        :param feasible_point: Feasible Point
        :type feasible_point: ndarray
        '''

        X = dir_orig_space

        X_scaled = self.scaler[block_id].transform(X)
        y = feasible_point

        bin_index = self.binary_index[block_id]
        if len(bin_index) > 1:
            loss = 'categorical_crossentropy'
        else:
            loss = 'binary_crossentropy'
        self.clf_batch[block_id].compile(loss=loss, optimizer='adam',
                                         metrics=['acc'])
        self.clf_batch[block_id].fit(X_scaled, y, epochs=10, verbose=0)

    def add_train_data(self, block_id, dir_orig_space, feasible_point):
        '''Adds training data to the Surrogate Model
        :param block_id, Block Identifier
        :type block_id: int
        :param dir_orig_space: direction in original space
        :type dir_orig_space: ndarray
        :param feasible_point: Feasible Point
        :type feasible_point: ndarray
        '''

        last_X = dir_orig_space.reshape(1, -1)

        last_y = feasible_point

        self.X_train[block_id] = np.concatenate((self.X_train[block_id], last_X), axis=0)
        self.y_train[block_id] = np.concatenate((self.y_train[block_id], last_y), axis=0)
