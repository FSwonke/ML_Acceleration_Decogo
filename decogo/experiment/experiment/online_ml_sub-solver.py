import numpy as np
import matplotlib.pyplot as plt
import os
print(os.getcwd())
import sys
path = os.path.dirname(os.getcwd())
while os.path.basename(path) != 'decogo':
    path = os.path.dirname(path)
sys.path.insert(0, path)
print('path',path)
from sklearn.neural_network import MLPRegressor
from sklearn.neural_network import MLPClassifier
from experiment_dataset import ExperimentData
from sklearn.preprocessing import StandardScaler

from scipy import stats

from sklearn.decomposition import PCA

from decogo.model.block_model import BlockModel
from decogo.problem.decomposed_problem import \
    DecomposedProblem
from tests.examples.tu.DESSLib_testmodel.DESS_blockstructure import \
    generate_model
from decogo.pyomo_minlp_model.input_model import PyomoInputModel

import copy


class StreamBatchData:
    """" Store stream data and normalization for update of surrogate model"""

    def __init__(self, x, y, block_id=None, ia_sol=None, decogo_model=None):
        self.decogo_model = decogo_model
        self.block_id = block_id
        self.x = x
        self.regr = None
        self.clf = None
        self.stream_index = None
        self.points = []
        self.global_solve_points = []
        self.add_all_global_solve_points(y_orig=y)
        continuous_index, binary_index = self._variable_index(y)
        self.continuous_index = continuous_index
        self.binary_index = binary_index
        self.y = y[:, continuous_index]
        self.ia_sol = ia_sol
        if binary_index:
            self.y_clf = y[:, binary_index]
        else:
            self.y_clf = None
        if ia_sol is not None:
            self.ia_sol_clf = ia_sol[:, binary_index]
            for i in range(self.ia_sol_clf.shape[0]):
                for j in range(self.ia_sol_clf.shape[1]):
                    self.ia_sol_clf[i, j] = round(self.ia_sol_clf[i, j])
        else:
            self.ia_sol_clf = None

    def add_all_global_solve_points(self, y_orig):
        """ add point of pre-training data during online training """
        for n in range(y_orig.shape[0]):
            point = y_orig[n, :]
            new_point = self.is_new_point_global_solve(point)
            if new_point:
                self.global_solve_points.append((point,))

    def check_points(self):
        """ count number of global point, ml point, overlapped points"""
        num_global_points = len(self.global_solve_points)
        num_global_points_ml = 0
        num_ml_points = len(self.points)
        for point in self.global_solve_points:
            new_point = self.is_new_point(point[0],
                                          min_inner_point_distance=1e1)
            if not new_point:
                num_global_points_ml += 1

        return num_global_points, num_global_points_ml, num_ml_points

    def offline_train(self, stream_index, regr=None, clf=None):
        """ testing off-line training of regr/clf model and its prediction"""
        self.stream_index = stream_index
        self.x_scaler = StandardScaler().fit(self.x[0:stream_index, :])
        self.y_scaler = StandardScaler().fit(self.y[0:stream_index, :])
        x_tr_scaled = self.x_scaler.transform(self.x[0:stream_index, :])
        y_tr_scaled = self.y_scaler.transform(self.y[0:stream_index, :])
        y_clf_tr = self.y_clf[0:stream_index, :]
        if regr is not None:
            self.regr = regr
            self.regr.fit(x_tr_scaled, y_tr_scaled)
            y_v_pred = self.regr_score()

        if clf is not None and self.y_clf is not None:
            self.clf = clf
            self.clf.fit(x_tr_scaled, y_clf_tr)
            y_clf_v_pred = self.clf_score()

        direction_v = self.x[stream_index:, :]
        direction_tr = self.x[0:stream_index, :]
        reduce_cost_v = self.compute_reduce_cost(direction_v,
                                                 y_v_pred,
                                                 y_clf_v_pred)
        reduce_cost_tr = self.compute_reduce_cost(direction_tr,
                                                  self.y[0:stream_index, :],
                                                  self.y_clf[0:stream_index, :])
        dir_len = direction_v.shape[0]
        dir_dim = direction_v.shape[1]
        fig = plt.figure(figsize=(10, 8))
        title = 'Block-{0}: reduced cost and constraint violation ' \
                'of predicted points'.format(self.block_id)
        fig.suptitle(title)
        ax = fig.add_subplot(2, 1, 1)
        ax.stem(range(self.stream_index, dir_len + self.stream_index),
                reduce_cost_v, label='reduce cost')
        ax.legend(loc='lower right')

        if decogo_evaluator is not None:
            y_v_violation = []
            for i in range(dir_len):
                point = []
                for j in range(dir_dim):
                    for l, index in enumerate(self.continuous_index):
                        if j == index:
                            point.append(y_v_pred[i, l])
                    for l, index in enumerate(self.binary_index):
                        if j == index:
                            point.append(y_clf_v_pred[i, l])
                max_vio = self.violation_nonlinear_constraints(sol=point)

                y_v_violation.append(max_vio)
            ax = fig.add_subplot(2, 1, 2)
            y_v_violation = np.array(y_v_violation) + 1
            ax.stem(range(self.stream_index, dir_len + self.stream_index),
                    y_v_violation,
                    label='max violation of nonlinear constraints')
            ax.set_yscale('log')
            ax.set_ylabel('log scaled violation')
            ax.set_xlabel('point index')
            ax.legend(loc='upper right')
            ax.plot([self.stream_index, dir_len + self.stream_index - 1],
                    [1, 1], 'r-')
        fig.tight_layout()
        fig.show()

    def online_clf_train(self, start_index, clf):
        """ Online train and evaluation of surrogate model """
        # online training data set
        print('binaryIndex:',self.binary_index)
        x_tr = self.x[0:start_index, :]
        y_tr = self.y_clf[0:start_index, :]
        point_dim = self.x.shape[1]
        # add existing point
        new_point_index = []
        y_clf_new_point = None
        for l in range(start_index):
            point = self._obtain_y(l)
            is_new_point = self.add_point(point)
            if is_new_point:
                new_point_index.append(l)

        y_clf_new_point = y_tr[new_point_index, :]

        # online train and evaluation
        y_pred_index = []
        y_pred_val = None
        y_corr_index = []
        y_rounding_index = []
        y_corr_val = None
        y_rounding_val = None
        for m in range(start_index, self.x.shape[0]):
            print('index {0} \n'.format(m))
            x_scaler = StandardScaler().fit(x_tr)
            x_tr_scaled = x_scaler.transform(x_tr)
            if y_tr.shape[1] == 1:
                clf.fit(x_tr_scaled, y_tr.reshape(-1))
            else:
                clf.fit(x_tr_scaled, y_tr)
            new_direction = self.x[m, :]
            print('x_tr',x_tr)
            print('=======================')
            print('y_tr', y_tr)
            x_new = x_scaler.transform(new_direction.reshape(1, -1))
            #PREDICT
            y_clf_new = clf.predict(x_new)
            print('y_clf_new',y_clf_new)
            y_clf_new = y_clf_new.reshape(1, -1)
            y_pred_index.append(m)
            if y_pred_val is not None:
                y_pred_val = np.concatenate((y_pred_val,
                                             y_clf_new), axis=0)
            else:
                y_pred_val = copy.copy(y_clf_new)

            update_model, is_new_point = \
                self.evaluate_prediction(new_direction, y_clf_new, m,
                                         x_scaler, x_tr)
            if is_new_point:
                new_point_index.append(m)
                if y_clf_new_point is not None:
                    y_clf_new_point = np.concatenate((y_clf_new_point,
                                                      y_clf_new), axis=0)
                else:
                    y_clf_new_point = copy.copy(y_clf_new)

            if update_model:
                _, tilde_y, _ = \
                    self.global_solve(new_direction, index=m)
                tilde_y_clf = tilde_y[self.binary_index]
                y_pred_val[-1] = tilde_y_clf
                tilde_y_clf = tilde_y_clf.reshape(1, -1)

                y_corr_index.append(m)
                if y_corr_val is not None:
                    y_corr_val = np.concatenate((y_corr_val,
                                                 tilde_y_clf), axis=0)
                else:
                    y_corr_val = copy.copy(tilde_y_clf)

                print("global solve: {0}".format(tilde_y))
                new_point_index.append(m)
                if y_clf_new_point is not None:
                    y_clf_new_point = np.concatenate((y_clf_new_point,
                                                      tilde_y_clf), axis=0)
                else:
                    y_clf_new_point = copy.copy(tilde_y_clf)
                # update training data
                x_tr = np.concatenate((x_tr,
                                       new_direction.reshape(1, -1)), axis=0)
                y_tr = np.concatenate((y_tr, tilde_y_clf), axis=0)

        # result visualization
        y_num = self.y_clf.shape[1]
        var_index = self.binary_index
        sub_plot_num = 2
        initial_set = [l for l in range(0, start_index + 1)]
        pred_set = [l for l in range(start_index, self.y_clf.shape[0])]
        y_true = self.y_clf[pred_set, :]
        score = y_true == y_pred_val
        accuracy = np.average(score)
        for i, index in enumerate(var_index):
            if i % sub_plot_num == 0:
                fig = plt.figure(figsize=(8, 6))
                title = 'Prediction of binaries in Block {0}; ' \
                        'Accuracy: {1}'.format(self.block_id,
                                               round(accuracy, 3))
                fig.suptitle(title)
            ax = fig.add_subplot(sub_plot_num, 1, i % sub_plot_num + 1)
            ax.plot(initial_set, self.y_clf[initial_set, i], 'b--',
                    label='Training set')
            ax.plot(pred_set, y_true[:, i], 'r--',
                    label='Validation set')

            ax.plot(y_pred_index, y_pred_val[:, i], 'm*',
                    label='Prediction')

            if y_corr_index:
                ax.plot(y_corr_index, y_corr_val[:, i], 'ro',
                        label='Correction')
            ax.plot(new_point_index, y_clf_new_point[:, i], 'g>',
                    label='New point')
            ax.set_ylabel('y[{0}]'.format(index))
            ax.set_xlabel('Iterations')
            ax.legend(loc='center right')
            if i % sub_plot_num == sub_plot_num - 1 or i == y_num - 1:
                fig.tight_layout()
                fig.show()

        # num_global_points, num_global_points_ml, num_ml_points_new = \
        #     self.check_points()

    def regr_score(self, show_fig=True, sub_plot_num=2):
        """ regression model for offline training """
        block_id = self.block_id
        x_v_scaled = self.x_scaler.transform(self.x[self.stream_index:, :])
        y_v_scaled = self.y_scaler.transform(self.y[self.stream_index:, :])
        y_v_pred_scaled = self.regr.predict(x_v_scaled)
        if len(y_v_pred_scaled.shape) == 1:
            y_v_pred_scaled = y_v_pred_scaled[:, np.newaxis]
        y_v_pred = self.y_scaler.inverse_transform(y_v_pred_scaled)
        score_v = self.regr.score(x_v_scaled, y_v_scaled)

        x_tr_scaled = self.x_scaler.transform(self.x[0:self.stream_index, :])
        y_tr_scaled = self.y_scaler.transform(self.y[0:self.stream_index, :])
        y_tr_pred_scaled = self.regr.predict(x_tr_scaled)
        if len(y_tr_pred_scaled.shape) == 1:
            y_tr_pred_scaled = y_tr_pred_scaled[:, np.newaxis]
        y_tr_pred = self.y_scaler.inverse_transform(y_tr_pred_scaled)
        score_tr = self.regr.score(x_tr_scaled, y_tr_scaled)
        if show_fig:
            y_num = self.y.shape[1]
            y_len = self.y.shape[0]
            var_index = self.continuous_index

            for i, index in enumerate(var_index):
                if i % sub_plot_num == 0:
                    fig = plt.figure(figsize=(10, 8))
                    title = 'Block-{0} self validation R2:{1}; ' \
                            'cross validation R2: {2}'.format(
                                block_id,
                                round(score_tr, 2), round(score_v, 2))
                    fig.suptitle(title)
                ax = fig.add_subplot(sub_plot_num, 1, i % sub_plot_num + 1)
                ax.plot(self.y[:, i], 'r--', label='actual point')
                ax.plot(range(0, self.stream_index),
                        y_tr_pred[:, i], 'b--', label='training set')
                ax.plot(range(self.stream_index, y_len),
                        y_v_pred[:, i], 'm--', label='validation set')
                ax.set_xlabel('points[{0}]'.format(index))
                ax.legend(loc='upper right')
                if i % sub_plot_num == sub_plot_num - 1 or i == y_num - 1:
                    fig.tight_layout()
                    fig.show()
                    # fig.suptitle('Hist of x in block {0} '
                    #                     '(num: {1})'.format(
                    #                        block_id, length))
                    # fig.tight_layout()
                    # fig.savefig(self.directory +
                    #                    'hist_block_{0}_x_{1}'.format(
                    #                        block_id,
                    #                        math.ceil((i + 1) / sub_plot_num)))
                    # plt.close(fig)
                    # plt.clf()
        return y_v_pred

    def clf_score(self, show_fig=True, sub_plot_num=2):
        """ classification model for offline training """
        block_id = self.block_id
        x_v_scaled = self.x_scaler.transform(self.x[self.stream_index:, :])
        y_clf_v = self.y_clf[self.stream_index:, :]
        y_clf_v_pred = self.clf.predict(x_v_scaled)
        if len(y_clf_v_pred.shape) == 1:
            y_clf_v_pred = y_clf_v_pred[:, np.newaxis]
        score_clf_v = self.clf.score(x_v_scaled, y_clf_v)

        x_tr_scaled = self.x_scaler.transform(self.x[0:self.stream_index, :])
        y_clf_tr = self.y_clf[0:self.stream_index, :]
        y_clf_tr_pred = self.clf.predict(x_tr_scaled)
        if len(y_clf_tr_pred.shape) == 1:
            y_clf_tr_pred = y_clf_tr_pred[:, np.newaxis]
        score_clf_tr = self.clf.score(x_tr_scaled, y_clf_tr)
        if show_fig:
            y_num = self.y_clf.shape[1]
            y_len = self.y_clf.shape[0]
            var_index = self.binary_index

            for i, index in enumerate(var_index):
                if i % sub_plot_num == 0:
                    fig = plt.figure(figsize=(10, 8))
                    title = 'Block-{0}-   self validation accuracy:{1}; ' \
                            'cross validation accuracy: {2}'.format(
                                block_id,
                                round(score_clf_tr, 2), round(score_clf_v, 2))
                    fig.suptitle(title)
                ax = fig.add_subplot(sub_plot_num, 1, i % sub_plot_num + 1)
                ax.plot(self.y_clf[:, i], 'ro--', label='actual point')
                if self.ia_sol_clf is not None:
                    ax.plot(self.ia_sol_clf[:, i] - 0.01, 'kv-.',
                            label='rounded ia sol')
                ax.plot(range(0, self.stream_index),
                        y_clf_tr_pred[:, i], 'bo', label='training set')
                ax.plot(range(self.stream_index, y_len),
                        y_clf_v_pred[:, i], 'mo', label='validation set')
                ax.set_xlabel('points[{0}]'.format(index))
                ax.legend(loc='center right')
                if i % sub_plot_num == sub_plot_num - 1 or i == y_num - 1:
                    fig.tight_layout()
                    fig.show()
                    # fig.suptitle('Hist of x in block {0} '
                    #                     '(num: {1})'.format(
                    #                        block_id, length))
                    # fig.tight_layout()
                    # fig.savefig(self.directory +
                    #                    'hist_block_{0}_x_{1}'.format(
                    #                        block_id,
                    #                        math.ceil((i + 1) / sub_plot_num)))
                    # plt.close(fig)
                    # plt.clf()
        return y_clf_v_pred

    @staticmethod
    def _variable_index(points):
        """ check index of binary and continuous variable"""
        p_num = points.shape[1]
        binary_index = []
        continuous_index = []
        for l in range(p_num):
            used = set()
            unique = [j for j in points[:, l] if
                      j not in used and (used.add(j) or True)]
            if len(unique) == 2:
                if max(unique) == 1 and min(unique) == 0:
                    binary_index.append(l)
                else:
                    continuous_index.append(l)
            else:
                continuous_index.append(l)

        return continuous_index, binary_index

    def compute_reduce_cost(self, direction, y, y_clf):
        """compute reduce cost of prediction"""
        dir_dim = direction.shape[1]
        dir_len = direction.shape[0]
        reduce_cost = np.array([0.0 for i in range(dir_len)])
        for i in range(dir_dim):
            for j, index in enumerate(self.continuous_index):
                if i == index:
                    reduce_cost += np.prod([direction[:, index],
                                            y[:, j]], axis=0)
            for j, index in enumerate(self.binary_index):
                if i == index:
                    reduce_cost += np.prod([direction[:, index],
                                            y_clf[:, j]], axis=0)
        return reduce_cost

    def violation_nonlinear_constraints(self, sol):
        """ calculate sol's max violation of nonlinear constraints in block
        block_id"""
        max_viol, _ = self.decogo_model.violation_nonlinear_constraints(
            block_id=self.block_id, x=sol)
        return max_viol

    def local_nlp_solve(self, direction, sol):
        """ call local solve of sub-problem"""
        block_id = self.block_id
        tilde_y, new_point_obj_val = self.decogo_model.local_nlp_opt(block_id,
                                                                     direction,
                                                                     sol)
        new_point = self.add_point(tilde_y, global_corr=False,
                                   min_inner_point_distance=10e-3)
        # print("---------------------------------------------------------------")
        # print("local ml solve: {0}".format(tilde_y))
        return new_point, tilde_y, new_point_obj_val

    def global_solve(self, direction, index=None):
        """ simulate global solve of sub-problem """
        if index is None:
            index = self._search_direction(direction)

        tilde_y = self._obtain_y(index)
        # print("global solve: {0}".format(tilde_y))
        new_point_obj_val = np.dot(tilde_y, direction)
        new_point = self.add_point(tilde_y, min_inner_point_distance=10e-3)
        # new_point = False
        return new_point, tilde_y, new_point_obj_val

    def _obtain_y(self, index_y):
        sol_dim = self.x.shape[1]
        y_i = np.zeros(sol_dim)
        for l in range(sol_dim):
            for n, index in enumerate(self.continuous_index):
                if l == index:
                    y_i[l] = self.y[index_y, n]
            for n, index in enumerate(self.binary_index):
                if l == index:
                    y_i[l] = self.y_clf[index_y, n]
        return y_i

    def _search_direction(self, direction):
        index = []
        for index_x in range(self.x.shape[0]):
            if direction == self.x[index_x, :]:
                index.append(index_x)
        obj_val = []
        y_val = []

        for index_y in index:
            tilde_y = self._obtain_y(index_y)
            obj_val.append(np.dot(tilde_y, direction))
            y_val.append(tilde_y)

        count_obj_val = obj_val.count(obj_val[0]) == len(obj_val)
        count_y_val = y_val.count(y_val[0]) == len(y_val)
        if count_y_val:
            index_y = index[0]
        else:
            if count_obj_val:
                index_y = index[0]
            else:
                min_obj_val_index = obj_val.index(min(obj_val))
                index_y = index[min_obj_val_index]

        return index_y

    def evaluate_prediction(self, direction, y_clf_val, index, x_scaler, x_tr):
        # testing pca clustering
        x_tr_scaled = x_scaler.transform(self.x[0:index, :])
        # x_tr_scaled = x_scaler.transform(x_tr)

        x_n = direction.reshape(1, -1)
        x_n_scaled = x_scaler.transform(x_n)

        pca_model = PCA().fit(x_tr_scaled)
        T_tr = pca_model.transform(x_tr_scaled)
        t_n = pca_model.transform(x_n_scaled)
        # print('t_n:')
        # print(t_n)
        explained_variance_ratio = pca_model.explained_variance_ratio_
        accumulated_var_ratio = explained_variance_ratio[0]
        ratio_threshold = 0.95
        for n in range(x_tr_scaled.shape[1]):
            if accumulated_var_ratio >= ratio_threshold:
                n_components = n + 1
                break
            else:
                accumulated_var_ratio += explained_variance_ratio[n+1]

        _, _, t2_bools, _, param = \
            hotellings_t2(T_tr, t_n, n_components=n_components)
        t2_bools = t2_bools[0]
        for t2_outlier in t2_bools:
            if t2_outlier:
                update_model = True
                print("pca >> t2 outlier, index {0}".format(index))
                break
            else:
                update_model = False

        # spe_bool = spe(T_tr, t_n,  n_components=n_components)
        # if spe_bool:
        #     print("pca >> spe outlier, index {0}".format(index))
        # if update_model:
        #     update_model = spe_bool
        # sub-problem fast solve
        sol = np.zeros(x_tr.shape[1])
        for j in range(x_tr.shape[1]):
            for n, l in enumerate(self.binary_index):
                if j == l:
                    sol[j] = y_clf_val[:, n]
            for n, l in enumerate(self.continuous_index):
                if j == l:
                    sol[j] = self.ia_sol[index, j]

        _, min_obj_val = self.get_min_inner_point(direction)

        new_point, tilde_y, new_point_obj_val = self.local_nlp_solve(direction,
                                                                     sol)
        # check if it is new point, check the reduced cost;
        # testing update
        if new_point:
            if new_point_obj_val <= min_obj_val:
                update_model = False

        return update_model, new_point

    def add_point(self, point, min_inner_point_distance=None,
                  global_corr=True):
        """Adds the new point and column. If ``min_inner_point_distance``
        is ``None``, the column is always added (based on the reduced cost
        computation)

        """

        new_point = self.is_new_point(point, min_inner_point_distance)

        if new_point is True:
            self.points.append((point,))
            if global_corr:
                print("Point {0} (global solve): {1}".format(len(self.points),
                                                             point))
            else:
                print("Point {0} (local solve): {1}".format(len(self.points),
                                                            point))

        return new_point

    def is_new_point(self, point, min_inner_point_distance=None):
        min_dist = float('inf')
        # find minimum distance to columns
        if self.points:
            for p in self.points:
                dist = np.linalg.norm(point - p, ord=np.inf)
                if dist < min_dist:
                    min_dist = dist

            new_point = False
            if min_dist >= 1e-10:
                # makes sure that completely identical column is not added
                # to the inner points
                if min_inner_point_distance is not None:
                    if min_dist > min_inner_point_distance:
                        new_point = True
                    else:
                        new_point = False
                else:
                    new_point = True
        else:
            new_point = True

        return new_point

    def is_new_point_global_solve(self, point, min_inner_point_distance=None):
        min_dist = float('inf')
        # find minimum distance to columns
        if self.global_solve_points:
            for p in self.global_solve_points:
                dist = np.linalg.norm(point - p, ord=np.inf)
                if dist < min_dist:
                    min_dist = dist

            new_point = False
            if min_dist >= 1e-10:
                # makes sure that completely identical column is not added
                # to the inner points
                if min_inner_point_distance is not None:
                    if min_dist > min_inner_point_distance:
                        new_point = True
                    else:
                        new_point = False
                else:
                    new_point = True
        else:
            new_point = True

        return new_point

    def get_min_inner_point(self, dir_orig_space):
        """Get the inner point based on the minimum value regarding the
        direction in original space, i.e. :math:`points = {\\mathrm{argmin\ }}
        d^Tx, x \\in S`, where :math:`S` is the set of inner points
        """
        min_point = None
        min_obj_val = float('inf')

        for point in self.points:
            obj_val = np.dot(dir_orig_space, point[0])
            if obj_val < min_obj_val:
                min_obj_val = obj_val
                min_point = point[0]

        return min_point, min_obj_val


def spe(T, t_n, n_components=2):
    """ calculate spe (pca) and detect anomaly"""
    x_dim = T.shape[1]
    if n_components < x_dim:
        T_tilde = T[:, n_components:]
        t_n_tilde = t_n[:, n_components:]
        spe = np.sum(T_tilde * T_tilde, axis=1)
        spe2over3 = np.cbrt(spe * spe)
        mean, std = np.mean(spe2over3), np.std(spe2over3)
        spe_n = np.sum(t_n_tilde * t_n_tilde, axis=1)
        spe_threshold = np.power(mean + std, 3)
        spe_bool = spe_n > spe_threshold
    else:
        spe_bool = False

    return spe_bool


def hotellings_t2(X, x_n, alpha=0.05, df=1, n_components=5, verbose=3):
    """Test for outlier using hotelling T2 test.

    Description
    -----------
    Test for outliers using chi-square tests for each of the n_components.
    The resulting P-value matrix is then combined using fishers method per sample.
    The results can be used to priortize outliers as those samples that are an outlier
    across multiple dimensions will be more significant then others.

    Parameters
    ----------
    X : numpy-array.
        Principal Components.
    alpha : float, (default: 0.05)
        Alpha level threshold to determine outliers.
    df : int, (default: 1)
        Degrees of freedom.
    n_components : int, (default: 5)
        Number of PC components to be used to compute the Pvalue.
    param : 2-element tuple (default: None)
        Pre-computed mean and variance in the past run. None to compute from scratch with X.
    Verbose : int (default : 3)
        Print to screen. 0: None, 1: Error, 2: Warning, 3: Info, 4: Debug, 5: Trace

    Returns
    -------
    outliers : pd.DataFrame
        dataframe containing probability, test-statistics and boolean value.
    y_bools : array-like
        boolean value when significant per PC.
    param : 2-element tuple
        computed mean and variance from X.
    """
    n_components = np.minimum(n_components, X.shape[1])
    X = X[:, 0:n_components]

    x_n = x_n[:, 0:n_components]
    y_n = x_n

    mean, var = np.mean(X), np.var(X)
    param = (mean, var)
    if verbose>=3:
        print('[pca] >Outlier detection using Hotelling T2 test with alpha=[%.2f] and n_components=[%d]' %(alpha, n_components))
    y_score = (y_n - mean) ** 2 / var
    # Compute probability per PC whether datapoints are outside the boundary
    y_proba = 1 - stats.chi2.cdf(y_score, df=df)
    # Set probabilities at a very small value when 0. This is required for the Fishers method. Otherwise inf values will occur.
    y_proba[y_proba==0]=1e-300

    # Compute the anomaly threshold
    anomaly_score_threshold = stats.chi2.ppf(q=(1 - alpha), df=df)
    # Determine for each samples and per principal component the outliers
    y_bools = y_score >= anomaly_score_threshold

    # Combine Pvalues across the components
    Pcomb = []
    # weights = np.arange(0, 1, (1/n_components) )[::-1] + (1/n_components)
    for i in range(0, y_proba.shape[0]):
        # Pcomb.append(stats.combine_pvalues(y_proba[i, :], method='stouffer', weights=weights))
        Pcomb.append(stats.combine_pvalues(y_proba[i, :], method='fisher'))

    Pcomb = np.array(Pcomb)
    # outliers = pd.DataFrame(data={'y_proba':Pcomb[:, 1], 'y_score': Pcomb[:, 0], 'y_bool': Pcomb[:, 1] <= alpha})
    # Return
    return y_proba, y_score, y_bools, Pcomb, param


class DecogoEvaluation:
    """"correction and evaluation methods for surrogate model"""

    def __init__(self, input_model):
        block_model = BlockModel(input_model)
        self.problem = DecomposedProblem(block_model,
                                         input_model.sub_problems,
                                         input_model.original_problem)

    def violation_nonlinear_constraints(self, block_id, x):
        """"evaluate violation of nonlinear constraints given
        sub-problem solution estimate x"""

        """Determines if the solution is feasible regarding the nonlinear
        constraints by computing the maximum violation

        :param x: Given point
        :type x: BlockVector
        :return: Maximum violation and index of most violated constraint as \
        tuple (viol, block_id, index)
        :rtype: tuple
        """
        violation = 0
        idx = None

        for i, constr in enumerate(
                self.problem.block_model.sub_models[block_id].nonlin_constr):
            viol, val = constr.eval(x)
            if viol > violation:
                violation = viol
                idx = block_id, i

        return violation, idx

    def local_nlp_opt(self, block_id, dir_orig_space, x_k=None):
        """" correct estimate x by solving nlp sub-problem """
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

        integer_point = x_k

        # solve the sub-problem
        tilde_y, new_point_obj_val, _, sol_is_feasible = \
            self.problem.sub_problems[block_id].local_solve(
                direction=dir_orig_space,
                start_point=integer_point)

        if sol_is_feasible is False:
            tilde_y = None
            new_point_obj_val = None

        return tilde_y, new_point_obj_val

    def local_solve(self, block_id, dir_orig_space, x_k=None):
        """" correct estimate x by solving nlp sub-problem """
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

        best_point = x_k

        # solve the sub-problem
        tilde_y, new_point_obj_val, _, sol_is_feasible = \
            self.problem.sub_problems[block_id].local_solve_clf(
                direction=dir_orig_space,
                start_point=best_point)

        if sol_is_feasible is False:
            tilde_y = None
            new_point_obj_val = None

        return tilde_y, new_point_obj_val


class SurrogateModel:

    def __init__(self, block_id, binary_index):
        '''Constructor for the Surrogate Model
        '''
        self.block_id = block_id
        self.clf_batch = {}
        self.binary_index = binary_index
        self.scaler = {}

    def init_train(self, block_id, training_data):
        '''Method for initial training of the Surrogate Model
        param: block_id
        type: int
        param: training_data
        type: dict -> includes a list of tuples for each block
        '''
        self.clf_batch[block_id] = MLPClassifier(hidden_layer_sizes=(10, 10),
                                                 activation='relu',
                                                 max_iter=100,
                                                 alpha=1e-5)
        # list of binary indexes for a given block
        index = self.binary_index[block_id]

        y = []
        # type blockdata: list of tuples, corresponding
        blockdata = training_data[block_id, :]
        for i in range(len(blockdata)):
            if i == 0:
                X = blockdata[i][0]
                for idx in index:
                    y.append(blockdata[i][1][idx])
            else:
                X = np.concatenate((X, blockdata[i][0]))
                for idx in index:
                    y.append(blockdata[i][1][idx])
            # scale data (standardize)
            self.scaler[block_id] = StandardScaler().fit(X)
            X_scaled = self.scaler[block_id].transform(X)

            self.clf_batch[block_id].fit(X_scaled, y)

    def predict(self, block_id, direction):
        # transform input
        transformed_direction = self.scaler[block_id].transform(direction)
        # predict method
        prediction = self.clf_batch[block_id].predict(transformed_direction)
        # inverse transform
        # inversetransform_pred = self.scaler[block_id].inverse_transform(prediction)

        return prediction

    def test_init_train(self):
        pass


if __name__ == '__main__':
    # input pyomo model
    superstructure = 'S4'
    load_case = 'L4'
    load_cases_per_block = 6
    data_instance = 1

    input_model = generate_model(superstructure, load_case,
                                 load_cases_per_block,
                                 data_instance)
    pyomo_model = PyomoInputModel(input_model)
    decogo_evaluator = DecogoEvaluation(pyomo_model)

    # historical data
    data_S4L4 = ExperimentData(instance='S4L4no_estimate_dual_run_1',
                               add_fig=False)
    block_index = 0
    num_index = list(range(23, 74))  # block 0
    # num_index = list(range(21, 74))  # block 4
    # num_index = list(range(15, 75))   # block 1
    data_length = len(num_index)
    y_clean, _ = data_S4L4.point_clean(block_id=block_index)
    print('y_clean: ', y_clean)
    y = y_clean[num_index, :]
    print('y',y)
    x = data_S4L4.x[block_index][num_index, :]

    ia_sol = data_S4L4.ia_sol[block_index][num_index, :]
    stream_list = [10]
    regr_batch = {}
    clf_batch = {}

    stream_batch = StreamBatchData(x, y, ia_sol=ia_sol,
                                   decogo_model=decogo_evaluator,
                                   block_id=block_index)

    for i, num in enumerate(stream_list):

        clf_batch[i] = MLPClassifier(solver='lbfgs', alpha=1e-5,
                                     hidden_layer_sizes=(25, 25),
                                     max_iter=40000,
                                     random_state=1)

        stream_batch.online_clf_train(start_index=num, clf=clf_batch[i])

