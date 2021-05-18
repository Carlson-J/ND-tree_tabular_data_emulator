# import numpy as np
#
#
# def get_extent(X, Y=None, already_log=False):
#     """
#     Returns the log10() of the extent of X and Y.
#     :param X: (2d array or dict) meshgrid density array or table dict. If Y is not included it is assumed X
#         is a dict.
#     :param Y: (2d array) meshgrid temp array
#     :param already_log: (bool) if X and Y are already in log space don't take log.
#     :return: [lower_den, upper_den, lower_temp, upper_temp]
#     """
#     # if a table is the input unpack first
#     if isinstance(X, dict):
#         Y = X['temp']
#         X = X['den']
#     # Return extent and do log transform if needed
#     if already_log:
#         return np.array([X[0, 0], X[0, -1], Y[0, 0], Y[-1, 0]])
#     else:
#         return np.log10(np.array([X[0, 0], X[0, -1], Y[0, 0], Y[-1, 0]]))
#
def create_mask():

    return
#
# class Mask:
#
#     def __init__(self, X, Y, x_range, y_range, log_transform_XY=True, require_square=True):
#         """
#         A mask for 2d data that has a corresponding X, Y 2d domain, such as generated from np.meshgrid.
#         It is assumed that the points are on a regular square grid, i.e., all points are evenly spaced in x and y dims.
#         :param X: (2d array)
#         :param Y: (2d array)
#         :param x_range: [x_low, x_high]
#         :param y_range: [y_low, y_high]
#         :param log_transform_XY: (bool) if x_range and y_range are in log domain but X and Y are not.
#         """
#         # transform and grab arrays
#         if log_transform_XY:
#             x_array = np.log10(X[0, :])
#             y_array = np.log10(Y[:, 0])
#         else:
#             x_array = X[0, :]
#             y_array = Y[:, 0]
#
#         # unpack ranges
#         self.lb = x_range[0]
#         self.rb = x_range[1]
#         self.bb = y_range[0]
#         self.tb = y_range[1]
#
#         # Find boundaries in X
#         EPS = 10 ** -13
#         SMALL = 10 ** -200
#         # All of the complexity is to make sure that if the boundary or an entry in x or y is zero it is put in the right cell.
#         self.l_i = next((i for i, x in enumerate(x_array) if
#                          x + EPS * (abs(self.lb) + abs(self.rb)) >= self.lb * (1 - EPS * np.sign(x)) - SMALL), None)
#         self.r_i = next((i for i, x in enumerate(x_array) if
#                          x - EPS * (abs(self.rb) + abs(self.lb)) >= self.rb * (1 + EPS * np.sign(x)) + SMALL), None)
#         self.b_i = next((i for i, y in enumerate(y_array) if
#                          y + EPS * (abs(self.bb) + abs(self.tb)) >= self.bb * (1 - EPS * np.sign(y)) - SMALL), None)
#         self.t_i = next((i for i, y in enumerate(y_array) if
#                          y - EPS * (abs(self.tb) + abs(self.bb)) >= self.tb * (1 + EPS * np.sign(y)) + SMALL), None)
#         # check for far right or top
#         if self.r_i is None:
#             self.r_i = len(x_array)
#         if self.t_i is None:
#             self.t_i = len(y_array)
#         # Check that domain is square in index space.
#         self.Nx = self.r_i - self.l_i
#         self.Ny = self.t_i - self.b_i
#         if require_square and self.r_i - self.l_i != self.t_i - self.b_i:
#             raise RuntimeError(f"Masked region not square! {self.r_i - self.l_i} != {self.t_i - self.b_i}")
#
#     def __call__(self, X, flt=False, return_logical_mask=False):
#         """
#         return the masked portion of X. If X is a dict, a dictionary is returned where the domain and all vars have
#         been masked.
#         :param X: (2d array) or (dict) table
#         :param flt: (bool) return flattened matrix
#         :param return_logical_mask: (bool) return a logical mask of same shape as X
#         :return: (2d array) a subset of X
#         """
#         if isinstance(X, dict):
#             # assume it is a table
#             table_new = {}
#             table_new['den'] = X['den'][self.b_i:self.t_i, self.l_i:self.r_i]
#             table_new['temp'] = X['temp'][self.b_i:self.t_i, self.l_i:self.r_i]
#             table_new['Table_Values'] = {}
#             for key in X['Table_Values'].keys():
#                 table_new['Table_Values'][key] = X['Table_Values'][key][self.b_i:self.t_i, self.l_i:self.r_i]
#             return table_new
#
#         if return_logical_mask:
#             mask = np.zeros_like(X, dtype=np.bool_)
#             mask[self.b_i:self.t_i, self.l_i:self.r_i] = True
#             return mask
#         if flt:
#             return X[self.b_i:self.t_i, self.l_i:self.r_i].flatten()
#         else:
#             return X[self.b_i:self.t_i, self.l_i:self.r_i]
#
#
# class IndexMask(Mask):
#     def __init__(self, left_i, right_i, bottom_i, top_i):
#         """
#         A child of the Mask class that has the index boundaries as an input.
#         :param left_i: the left index of the mask
#         :param right_i: the right index of the mask
#         :param bottom_i: the bottom index of the mask
#         :param top_i: the top index of the mask
#         """
#         self.l_i = left_i
#         self.r_i = right_i
#         self.b_i = bottom_i
#         self.t_i = top_i
