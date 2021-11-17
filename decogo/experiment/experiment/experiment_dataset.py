import numpy as np
import matplotlib.pyplot as plt
import math
from sklearn import preprocessing
import copy


class ExperimentData:
    """ Container of experiment data for testing ml-based sub-solver;
    import data from csv files; preprocessing and visualization of sub-problem
    data
    """
    def __init__(self, exact_solver=True, instance='S4L4_run_1',
                 max_block=100, add_fig=True):
        self.x = {}
        self.y = {}
        self.ia_sol = {}
        self.points = {}
        self.dir = {}

        self.instance = instance
        for i in range(max_block):
            input_txt = "block_{0}_input_orig_space.csv".format(i)
            output_txt = "block_{0}_output_orig_space.csv".format(i)
            ia_sol_txt = "block_{0}_ia_orig_space.csv".format(i)
            if exact_solver:
                self.directory = ".\\data\\{0}\\exact_solver\\".format(
                    self.instance)
            try:
                self.x[i] = np.loadtxt(self.directory + input_txt,
                                       delimiter=',')
                self.y[i] = np.loadtxt(self.directory + output_txt,
                                       delimiter=',')
                self.ia_sol[i] = np.loadtxt(self.directory + ia_sol_txt,
                                            delimiter=',')
            except OSError:
                break

            if add_fig:
                # self.block_data_hist(block_id=i)
                self.column_count(block_id=i)
                self.dir_count(block_id=i)

    def column_count(self, block_id=0, min_dist_val=1e-10, add_fig=True):
        self.points[block_id] = []
        y_val = self.y[block_id]
        y_num = y_val.shape[1]
        length = y_val.shape[0]
        y_point_index = []
        for i in range(length):
            min_dist = float('inf')
            for counter, point in enumerate(self.points[block_id]):
                dist = np.linalg.norm(y_val[i] - point)
                if dist < min_dist:
                    min_dist = dist
                if min_dist < min_dist_val:
                    y_point_index.append(counter + 1)
                    break
            if min_dist > min_dist_val:
                self.points[block_id].append(y_val[i])
                if y_point_index:
                    y_point_index.append(counter + 2)
                else:
                    y_point_index.append(1)

        if add_fig:
            fig = plt.figure(figsize=(8, 6))
            ax = fig.add_subplot(1, 1, 1)
            ax.plot(y_point_index, 'ro', label='point series')
            ax.set_title('Block {0}: {1} variables, '
                         '{2} labels of points'.format(block_id, y_num,
                                                       max(y_point_index)))
            # plt.yticks(np.arange(1, max(y_point_index) + 1))
            ax.set_xlabel('Point Num.')
            ax.set_ylabel('Point Label')
            ax.yaxis.grid(color='k', linestyle='--', linewidth=0.5)
            fig.tight_layout()
            # fig.show()
            fig.savefig(self.directory + 'point_block_{0}'.format(block_id))
            plt.close(fig)
            plt.clf()

        return self.points[block_id], y_point_index

    def dir_count(self, block_id=0, min_dist_val=1e-10, add_fig=True):
        self.dir[block_id] = []
        y_val = self.x[block_id]
        y_num = y_val.shape[1]
        length = y_val.shape[0]
        y_point_index = []
        for i in range(length):
            min_dist = float('inf')
            for counter, point in enumerate(self.dir[block_id]):
                dist = np.linalg.norm(y_val[i] - point)
                if dist < min_dist:
                    min_dist = dist
                if min_dist < min_dist_val:
                    y_point_index.append(counter + 1)
                    break
            if min_dist > min_dist_val:
                self.dir[block_id].append(y_val[i])
                if y_point_index:
                    y_point_index.append(counter + 2)
                else:
                    y_point_index.append(1)

        if add_fig:
            fig = plt.figure(figsize=(8, 6))
            ax = fig.add_subplot(1, 1, 1)
            ax.plot(y_point_index, 'ro', label='dir series')
            ax.set_title('Block {0}: {1} variables, '
                         '{2} labels of dir'.format(block_id, y_num,
                                                       max(y_point_index)))
            # plt.yticks(np.arange(1, max(y_point_index) + 1))
            ax.set_xlabel('Dir Num.')
            ax.set_ylabel('Dir Label')
            ax.yaxis.grid(color='k', linestyle='--', linewidth=0.5)
            fig.tight_layout()
            # fig.show()
            fig.savefig(self.directory + 'dir_block_{0}'.format(block_id))
            plt.close(fig)
            plt.clf()

        return self.dir[block_id], y_point_index

    def point_clean(self, block_id=0, min_dist_val=1e-10):
        """ replace point with existing point that has better obj. value """
        points = []
        y_val = copy.copy(self.y[block_id])
        x_val = self.x[block_id]
        length = y_val.shape[0]
        y_index = 0
        for i in range(length):
            min_dist = float('inf')
            for l, val in enumerate(y_val[i]):
                if np.abs(val) < 1e-10:
                    y_val[i][l] = 0
                if np.abs(val - 1) < 1e-10:
                    y_val[i][l] = 1
            for counter, point in enumerate(points):
                dist = np.linalg.norm(y_val[i] - point)
                if dist < min_dist:
                    min_dist = dist
                if min_dist < min_dist_val:
                    y_index = counter
                    break
            if min_dist > min_dist_val:
                points.append(y_val[i])

            if i > 1:
                min_point, min_obj_val, index = \
                    self.get_min_inner_point(points, x_val[i])
                primal_bound = np.dot(x_val[i], y_val[i])
                if primal_bound + min(abs(primal_bound),
                                      abs(min_obj_val)) * 1e-12 \
                        >= min_obj_val:
                    y_val[i] = min_point

                if i in range(22, 28):
                    print('point_index= {0}, '
                          'min_index = {1}'.format(y_index, index))
                    print('direction: {0}, min_point_index: {1}, '
                          'min_obj_val: {2}'.format(x_val[i], index,
                                                    round(min_obj_val, 4)))

        return y_val, points,

    def get_min_inner_point(self, points, dir_orig_space):
        """Get the inner point based on the minimum value regarding the
        direction in original space, i.e. :math:`points = {\\mathrm{argmin\ }}
        d^Tx, x \\in S`, where :math:`S` is the set of inner points

        :param block_id: Block identifier
        :type block_id: int
        :param dir_orig_space: Given direction
        :type dir_orig_space: ndarray
        :return: Point and corresponding minimum value
        :rtype: tuple
        """
        min_point = 0
        min_obj_val = float('inf')
        index = 0

        for i, point in enumerate(points):
            obj_val = np.dot(dir_orig_space, point)
            if obj_val + min(abs(obj_val),
                             abs(min_obj_val)) * 1e-12 < min_obj_val:
                min_obj_val = obj_val
                min_point = point
                index = i

        return min_point, min_obj_val, index

    def block_data_visualize(self, block_id=0, variable=None):
        pass

    def block_data_hist(self, block_id=0, sub_plot_num=3):
        x_val = self.x[block_id]
        y_val = self.y[block_id]
        x_num = x_val.shape[1]
        length = x_val.shape[0]
        for i in range(x_num):
            if i % sub_plot_num == 0:
                fig_hist_x = plt.figure(figsize=(10, 8))
                fig_hist_y = plt.figure(figsize=(10, 8))
            ax_hist_x = fig_hist_x.add_subplot(sub_plot_num, 1,
                                               i % sub_plot_num + 1)
            ax_hist_x.hist(x_val[:, i])
            ax_hist_x.set_xlabel('x[{0}]'.format(i))
            ax_hist_y = fig_hist_y.add_subplot(sub_plot_num, 1,
                                               i % sub_plot_num + 1)
            ax_hist_y.hist(y_val[:, i])
            ax_hist_y.set_xlabel('points[{0}]'.format(i))
            if i % sub_plot_num == sub_plot_num - 1 or i == x_num - 1:
                fig_hist_x.suptitle('Hist of x in block {0} '
                                    '(num: {1})'.format(
                                       block_id, length))
                fig_hist_x.tight_layout()
                fig_hist_x.savefig(self.directory +
                                   'hist_block_{0}_x_{1}'.format(
                                       block_id,
                                       math.ceil((i + 1) / sub_plot_num)))
                fig_hist_y.suptitle('Hist of points in block {0} '
                                    '(num: {1})'.format(
                                       block_id, length))
                fig_hist_y.tight_layout()
                fig_hist_y.savefig(self.directory +
                                   'hist_block_{0}_y_{1}'.format(
                                       block_id,
                                       math.ceil((i + 1) / sub_plot_num)))
                plt.close(fig_hist_x)
                plt.close(fig_hist_y)
                plt.clf()

    def preprocess(self, block_id=0, y_var=None, x_var=None,
                   num_index=None, clean=True):
        if clean:
            y_val, _ = self.point_clean(block_id=block_id)
        else:
            y_val = self.y[block_id]
        if x_var is None:
            if num_index is None:
                x_train = self.x[block_id]
            else:
                x_train = self.x[block_id][num_index, :]
        else:
            if num_index is None:
                x_train = self.x[block_id][:, x_var]
            else:
                x_train = self.x[block_id][num_index, x_var]
        x_scaler = preprocessing.StandardScaler().fit(x_train)
        x_scaled = x_scaler.transform(x_train)
        if y_var is None:
            if num_index is None:
                y_train = y_val
            else:
                y_train = y_val[num_index, :]
        else:
            y_train = y_val[:, y_var]
            if num_index is not None:
                y_train = y_train[num_index, :]

        y_scaler = preprocessing.StandardScaler().fit(y_train)
        y_scaled = y_scaler.transform(y_train)
        return x_scaled, y_scaled, x_scaler, y_scaler

    def block_output_visualize(self, block_id=0, var_index=None, num_index=None,
                               y_pre=None, sub_plot_num=3,
                               title=None, clean=True, show_sample_index=False):
        if clean:
            y_val, _ = self.point_clean(block_id=block_id)
        else:
            y_val = self.y[block_id]
        if var_index is None:
            y_num = y_val.shape[1]
            var_index = list(range(y_num))
        else:
            y_num = len(var_index)

        for i, index in enumerate(var_index):
            if i % sub_plot_num == 0:
                fig = plt.figure(figsize=(7, 5))
                if title is not None:
                    fig.suptitle(title)
            ax = fig.add_subplot(sub_plot_num, 1, i % sub_plot_num + 1)
            if num_index is None:
                ax.plot(y_val[:, index], 'r--o', label='y[{0}]'.format(index))
            else:
                if show_sample_index:
                    ax.plot(num_index, y_val[num_index, index], 'r--',
                            label='y[{0}]'.format(index))
                else:
                    ax.plot(y_val[num_index, index], 'r--',
                            label='y[{0}]'.format(index))
            if y_pre is not None:
                if show_sample_index and num_index is not None:
                    ax.plot(num_index, y_pre[:, i],
                            'bo', label='predict point')
                else:
                    ax.plot(y_pre[:, i], 'bo', label='predict point')
            ax.set_xlabel('Iterations')
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

    def block_input_visualize(self, block_id=0, var_index=None,
                              num_index=None, sub_plot_num=3,
                              show_sample_index=False):
        x_val = self.x[block_id]
        if var_index is None:
            x_num = x_val.shape[1]
            var_index = list(range(x_num))
        else:
            x_num = len(var_index)

        for i, index in enumerate(var_index):
            if i % sub_plot_num == 0:
                fig = plt.figure(figsize=(7, 5))
            ax = fig.add_subplot(sub_plot_num, 1, i % sub_plot_num + 1)
            if num_index is None:
                ax.plot(x_val[:, index], 'r--', label='d[{0}]'.format(index))
            else:
                if show_sample_index:
                    ax.plot(num_index, x_val[num_index, index], 'r--',
                            label='d[{0}]'.format(index))
                else:
                    ax.plot(x_val[num_index, index], 'r--',
                            label='d[{0}]'.format(index))

            ax.set_xlabel('Iterations'.format(index))
            ax.legend(loc='upper right')
            if i % sub_plot_num == sub_plot_num - 1 or i == x_num - 1:
                fig.tight_layout()
                fig.show()


if __name__ == '__main__':
    data1 = ExperimentData(instance='S4L4no_estimate_dual_run_1', add_fig=False)
    block_index = 0

    num_index = list(range(23, 74))  # block 0
    # num_index = list(range(21, 75))  # block 4
    # num_index = None

    data1.block_output_visualize(block_id=block_index,
                                 num_index=num_index, clean=True,
                                 show_sample_index=True)
    data1.block_input_visualize(block_id=block_index,
                                num_index=num_index, show_sample_index=True)



