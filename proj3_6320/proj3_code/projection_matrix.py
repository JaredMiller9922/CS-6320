import numpy as np
import cv2
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

from scipy.optimize import least_squares
from scipy.optimize import minimize
from scipy.linalg import rq

import time


def projection(P: np.ndarray, points_3d: np.ndarray) -> np.ndarray:
    """
        Computes projection from [X,Y,Z,1] in homogenous coordinates to
        (x,y) in non-homogenous image coordinates.

        Args:
        -  P: 3x4 projection matrix
        -  points_3d : n x 4 array of points [X_i,Y_i,Z_i,1] in homogenouos coordinates
                       or n x 3 array of points [X_i,Y_i,Z_i]

        Returns:
        - projected_points_2d : n x 2 array of points in non-homogenous image coordinates
    """

    #######################################################################
    # YOUR CODE HERE                                                      #
    #######################################################################
    projected_points_2d = np.empty((len(points_3d),2))

    if points_3d.shape[1] == 4:
        projected_points_2d = np.empty((points_3d.shape[0],2))
        for i in range(0, len(points_3d)):
            x_i = (P[0] @ points_3d[i]) / (P[2] @ points_3d[i])
            y_i = np.dot(P[1], points_3d[i]) / np.dot(P[2], points_3d[i])
            # np.append(projected_points_2d, (x_i,y_i))
            projected_points_2d[i] = [x_i,y_i]
    else:
        # First append a 1 column
        ones_col = np.ones(points_3d.shape[0]).reshape(-1,1)
        points_3d = np.concatenate((points_3d, ones_col), axis=1)
        for i in range(0, len(points_3d)):
            x_i = (P[0] @ points_3d[i]) / (P[2] @ points_3d[i])
            y_i = np.dot(P[1], points_3d[i]) / np.dot(P[2], points_3d[i])
            # np.append(projected_points_2d, (x_i,y_i))
            projected_points_2d[i] = [x_i,y_i]

    #######################################################################
    #                           END OF YOUR CODE                          #
    #######################################################################

    return projected_points_2d


def objective_func(x, **kwargs): 
    """
        Calculates the difference in image (pixel coordinates) and returns
        it as a 2*n_points vector

        Args:
        -        x: numpy array of 11 parameters of P in vector form
                    (remember you will have to fix P_34=1) to estimate the reprojection error
        - **kwargs: dictionary that contains the 2D and the 3D points. You will have to
                   	retrieve these 2D (using the key ‘pts2d’) and 3D(using the key ‘pts3d’) points and then
		            use them to compute the reprojection error.
        Returns:
        -     diff: A 2*N_points-d vector (1-D numpy array) of differences between
                    projected and actual 2D points. (the difference between all the x
                    and all the y coordinates)

    """

    #######################################################################
    # YOUR CODE HERE                                                      #
    #######################################################################
    points_2d = kwargs['pts2d']
    points_3d = kwargs['pts3d']

    # Fix P_34 = 1
    x = np.append(x, 1)
    # Reshape x to work with projection equation
    x = x.reshape(3,4)
    projected_points_2d = projection(x, points_3d)
    diff = projected_points_2d - points_2d
    # Reshape into a row vector
    diff = diff.reshape(-1)
    #######################################################################
    #                           END OF YOUR CODE                          #
    #######################################################################

    return diff


def estimate_camera_matrix(pts2d: np.ndarray,
                           pts3d: np.ndarray,
                           initial_guess: np.ndarray) -> np.ndarray:
    '''
        Calls least_squres form scipy.least_squares.optimize and
        returns an estimate for the camera projection matrix

        Args:
        - pts2d: n x 2 array of known points (x_i, y_i) in image coordinates
        - pts3d: n x 3 array of known points in 3D, (X_i, Y_i, Z_i, 1)
        - initial_guess: 3x4 projection matrix initial guess

        Returns:
        - P: 3x4 estimated projection matrix

        Note: Because of the requirements of scipy.optimize.least_squares
              you will have to pass the projection matrix P as a vector.
              Since we will fix P_34 to 1 you will not need to pass all 12
              matrix parameters.

              You will also have to put pts2d and pts3d into a kwargs dictionary
              that you will add as an argument to least squares.

              We recommend that in your call to least_squares you use
              - method='lm' for Levenberg-Marquardt
              - verbose=2 (to show optimization output from 'lm')
              - max_nfev=50000 maximum number of function evaluations
              - ftol \
              - gtol  --> convergence criteria
              - xtol /
              - kwargs -- dictionary with additional variables
                          for the objective function
    '''

    start_time = time.time()

    #######################################################################
    # YOUR CODE HERE                                                      #
    #######################################################################
    dictionary = {'pts2d':pts2d,
              'pts3d':pts3d}
    initial_guess = initial_guess.reshape(-1)[:-1]

    res = least_squares(fun = objective_func, x0 = initial_guess, method = 'lm', verbose = 2, max_nfev=50000, kwargs=dictionary)

    P_vec = res.x
    P_vec = np.append(P_vec, 1)

    P = P_vec.reshape(3,4)
    #######################################################################
    #                           END OF YOUR CODE                          #
    #######################################################################

    print("Time since optimization start", time.time() - start_time)

    return P

def decompose_camera_matrix(P: np.ndarray) -> (np.ndarray, np.ndarray):
    '''
        Decomposes the camera matrix into the K intrinsic and R rotation matrix

        Args:
        -  P: 3x4 numpy array projection matrix

        Returns:

        - K: 3x3 intrinsic matrix (numpy array)
        - R: 3x3 orthonormal rotation matrix (numpy array)

        hint: use scipy.linalg.rq()
    '''

    #######################################################################
    # YOUR CODE HERE                                                      #
    #######################################################################
    # QR decomposistion on left-most 3x3 matrix of P
    left_block = P[:,:3]
    K,R = rq(left_block)

    #######################################################################
    #                           END OF YOUR CODE                          #
    #######################################################################

    return K, R

def calculate_camera_center(P: np.ndarray,
                            K: np.ndarray,
                            R: np.ndarray) -> np.ndarray:
    """
    Returns the camera center matrix for a given projection matrix.

    Args:
    -   P: A numpy array of shape (3, 4) representing the projection matrix

    Returns:
    -   cc: A numpy array of shape (3,) representing the camera center
            location in world coordinates
    """

    #######################################################################
    # YOUR CODE HERE                                                      #
    #######################################################################
    K_p = np.linalg.inv(K)
    # Kt = P4 -> K^-1 
    t = K_p @ P[:, -1]
    # x_c = R (X_w - t) 
    # x_c = Rx_w - tR (call tR is t)
    # x_c = Rx_w + t
    # 0 = Rx_cc + t
    # x_cc = -Rt
    cc = -R.T @ t
    #######################################################################
    #                           END OF YOUR CODE                          #
    #######################################################################

    return cc
