from nd_emulator.nd_tree import ND_Tree, create_children_nodes, load_emulator
import pytest
import numpy as np
import matplotlib.pyplot as plt



@pytest.mark.dependency()
def test_init(dataset_2d):
    """
    GIVEN: nd tree paramaters
    WHEN: tree is built
    THEN: No errors occur
    """
    model_classes = [{'type': 'nd-linear'}]
    max_depth = 3
    error_threshold = 0.1
    data, domain, spacing = dataset_2d
    try:
        ND_Tree(data, max_depth, domain, spacing, error_threshold, model_classes)
    except Exception as err:
        assert False, f'initializing ND_Tree raised an exception: {err}'


def test_2d_interpolation(dataset_2d_log):
    """
    GIVEN: 2d function evaluations and a domain.
    WHEN: Create an emulator from data
    THEN: Correctly sorts and interpolates data
    """
    EPS = 10**-14
    N = 200
    data, domain, spacing = dataset_2d_log
    error_threshold = 0
    max_depth = 2
    model_classes = [{'type': 'nd-linear', 'transforms': [None]*2}]
    # Create emulator
    emulator = ND_Tree(data, max_depth, domain, spacing, error_threshold, model_classes)
    # Compute new values over domain
    X, Y = np.meshgrid(np.linspace(domain[0][0], domain[0][1], N), np.logspace(np.log10(domain[1][0])
                                                                                 , np.log10(domain[1][1]), N))
    input = np.array([X.flatten(), Y.flatten()]).T
    f_interp = emulator(input).reshape([N, N])
    f_true = X*Y
    error = abs(f_true - f_interp)
    # resize and plot
    plt.imshow(error, origin='lower')
    plt.title("Should not see any grid structure")
    plt.colorbar()
    plt.show()

    # check if error is low
    assert np.all(error <= EPS)


def test_saving_nd_tree(dataset_4d_log):
    """
    GIVEN: 4d data and params for emulator
    WHEN: Emulator is saved, destroyed, and loaded
    THEN: Emulator produces same values before and after
    """
    EPS = 10**-15
    SAVE_LOCATION = './saved_emulator.hdf5'
    data, domain, spacing = dataset_4d_log
    error_threshold = 0
    max_depth = 2
    model_classes = [{'type': 'nd-linear', 'transforms': ['linear']*4}]
    # Create emulator
    emulator = ND_Tree(data, max_depth, domain, spacing, error_threshold, model_classes)
    inputs = np.random.uniform(0.1, 0.5, size=[100, len(spacing)])
    output_true = emulator(inputs)
    # save it
    emulator.save(SAVE_LOCATION)


def test_1d_interpolation():
    """
    GIVEN: 1d domain and function values
    WHEN: build tree
    THEN: refines correctly
    """
    # create function and domain
    domain = [[0,2]]
    x = np.linspace(domain[0][0],domain[0][1],2**5 + 1)
    y = 3*x
    y[x>1] = np.sin(x[x>1])
    data = {'f': y}
    spacing = ['linear']

    # build tree
    EPS = 10 ** -13
    N = 200
    error_threshold = 1e-2
    max_depth = 5
    model_classes = [{'type': 'nd-linear', 'transforms': [None]}]
    # Create emulator
    emulator = ND_Tree(data, max_depth, domain, spacing, error_threshold, model_classes)
    # Compute new values over domain
    x_test = np.linspace(domain[0][0],domain[0][1], N).reshape([N,1])
    f_interp = emulator(x_test)
    f_true = 3*x_test
    f_true[x_test>1] = np.sin(x_test[x_test > 1])
    error = abs(f_true.flatten() - f_interp)
    # Plot errors
    plt.plot(x_test, error, '--', label='error')
    plt.plot(x_test, f_true, label='true')
    plt.plot(x_test, f_interp, '--', label='interp')
    plt.plot(x, y, '.', label='data')
    plt.legend()
    plt.show()


def test_2d_convergence(dataset_2d_non_linear):
    EPS = 10 ** -4
    N = 100
    data, domain, spacing = dataset_2d_non_linear
    error_threshold = 0
    model_classes = [{'type': 'nd-linear', 'transforms': [None] * 2}]

    # Compute true values
    X, Y = np.meshgrid(np.linspace(domain[0][0], domain[0][1], N), np.linspace(domain[1][0], domain[1][1], N))
    input = np.array([X.flatten(), Y.flatten()]).T
    f_true = np.cos(X)*2 + np.sin(Y)

    # error arrays
    num_depths = 6
    l1_error = np.zeros(num_depths)
    lI_error = np.zeros(num_depths)

    for i in range(0, num_depths):
        # Create emulator
        emulator = ND_Tree(data, i+1, domain, spacing, error_threshold, model_classes)
        # test at different points
        f_interp = emulator(input).reshape([N, N])
        # compute error
        error = abs(f_true - f_interp)
        l1_error[i] = np.mean(error)
        lI_error[i] = np.max(error)
    # resize and plot
    plt.figure()
    plt.plot(np.arange(1, num_depths+1), l1_error, label='L1 Norm')
    plt.plot(np.arange(1, num_depths+1), lI_error, label='LI Norm')
    plt.title("Should look linear with a negative slope.")
    plt.xlabel("Depth")
    plt.ylabel("Error")
    plt.yscale("log")
    plt.legend()
    plt.show()
    # check if error is low
    assert np.all(error <= EPS)


def test_4d_convergence(dataset_4d_log_non_linear):
    """
    Given: 4d data set
    WHEN: Different max depths are given
    THEN: The error converges to 0
    """
    EPS = 10**-1
    data, domain, spacing = dataset_4d_log_non_linear
    # create test data
    low = np.array(domain)[:, 0]
    high = np.array(domain)[:, 1]
    N = 100
    inputs = np.random.uniform(low, high, size=[N, 4])
    f_true = (inputs[:, 0]**2 * inputs[:, 1] * inputs[:, 2] * inputs[:, 3] + 1).flatten()
    error_threshold = 0
    model_classes = [{'type': 'nd-linear', 'transforms': [None] * 4}]

    # error arrays
    num_depths = 2
    l1_error = np.zeros(num_depths)
    lI_error = np.zeros(num_depths)

    for i in range(0, num_depths):
        # Create emulator
        emulator = ND_Tree(data, i + 1, domain, spacing, error_threshold, model_classes)
        # test at different points
        f_interp = emulator(inputs).flatten()
        # compute error
        error = abs(f_true - f_interp)/abs(f_true)
        l1_error[i] = np.mean(error)
        lI_error[i] = np.max(error)

    plt.figure()
    plt.plot(np.arange(1, num_depths+1), l1_error, label='L1 Norm')
    plt.plot(np.arange(1, num_depths+1), lI_error, label='LI Norm')
    plt.title("Should look linear with a negative slope.")
    plt.xlabel("Depth")
    plt.ylabel("Error")
    plt.yscale("log")
    plt.legend()
    plt.show()
    # check if error is low
    assert np.all(l1_error[-1] <= EPS)

# def test_2d_log_transforms(dataset_2d_log_non_linear):
#     """
#     GIVEN: 2d function evaluations and a domain.
#     WHEN: Create an emulator from data with log transforms
#     THEN: Correctly sorts and interpolates data
#     """
#     EPS = 10 ** -14
#     N = 200
#     data, domain, spacing = dataset_2d_log_non_linear
#     error_threshold = 0
#     max_depth = 2
#     model_classes = [{'type': 'nd-linear', 'transforms': [None, 'log']}]
#     # Create emulator
#     emulator = ND_Tree(data, max_depth, domain, spacing, error_threshold, model_classes)
#     # Compute new values over domain
#     X, Y = np.meshgrid(np.linspace(domain[0][0], domain[0][1], N), np.logspace(np.log10(domain[1][0])
#                                                                                , np.log10(domain[1][1]), N))
#     input = np.array([X.flatten(), Y.flatten()]).T
#     f_interp = emulator(input).reshape([N, N])
#     f_true = X * Y
#     error = abs(f_true - f_interp)
#     # resize and plot
#     plt.imshow(error, origin='lower')
#     plt.title("Should not see any grid structure")
#     plt.colorbar()
#     plt.show()
#
#     # check if error is low
#     assert np.all(error <= EPS)