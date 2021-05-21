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


def test_create_children_nodes(default_root_node_2d):
    """
    GIVEN: a node
    WHEN: split nodes
    THEN: node has correct number of children and parameters
    """
    EPS = 10**-15
    node = default_root_node_2d
    num_dims = len(node['mask'])
    spacing = ['linear', 'linear']

    # create new children nodes
    create_children_nodes(node, spacing)

    # check to make sure everything is correct
    assert len(node['children']) == 2**num_dims
    for i in range(num_dims):
        node2 = node['children'][i]
        assert list(node['id']) + [i] == node2['id']
        assert node2['children'] is None
        assert node2['model'] is None
        assert node2['error'] is None
        assert node2['children'] is None
        assert node2['mask'] is not None
    # check new domains
    assert np.all(abs(node['children'][0]['domain'] - np.array([[0, 0.5], [0, 0.5]])) <= EPS)
    assert np.all(abs(node['children'][1]['domain'] - np.array([[0.5, 1], [0, 0.5]])) <= EPS)
    assert np.all(abs(node['children'][2]['domain'] - np.array([[0, 0.5], [0.5, 1]])) <= EPS)
    assert np.all(abs(node['children'][3]['domain'] - np.array([[0.5, 1], [0.5, 1]])) <= EPS)


def test_create_children_nodes_transforms(default_root_node_2d_log):
    """
    GIVEN: a node
    WHEN: split nodes
    THEN: node has correct number of children and parameters
    """
    EPS = 10**-15
    node, spacing = default_root_node_2d_log
    num_dims = len(node['mask'])

    # create new children nodes
    create_children_nodes(node, spacing)

    # check to make sure everything is correct
    assert len(node['children']) == 2**num_dims
    for i in range(num_dims):
        node2 = node['children'][i]
        assert list(node['id']) + [i] == node2['id']
        assert node2['children'] is None
        assert node2['model'] is None
        assert node2['error'] is None
        assert node2['children'] is None
        assert node2['mask'] is not None
    # check new domains
    assert np.all(abs(node['children'][0]['domain'] - np.array([[0, 0.5], [1, 10**0.5]])) <= EPS)
    assert np.all(abs(node['children'][1]['domain'] - np.array([[0.5, 1], [1, 10**0.5]])) <= EPS)
    assert np.all(abs(node['children'][2]['domain'] - np.array([[0, 0.5], [10**0.5, 10]])) <= EPS)
    assert np.all(abs(node['children'][3]['domain'] - np.array([[0.5, 1], [10**0.5, 10]])) <= EPS)


def test_2d_interpolation(dataset_2d_log):
    """
    GIVEN: 2d function evaluations and a domain.
    WHEN: Create an emulator from data
    THEN: Correctly sorts and interpolates data
    """
    EPS = 10**-13
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
    plt.colorbar()
    plt.show()
    plt.imshow(f_true, origin='lower')
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
    # load emulator
    emulator = load_emulator(SAVE_LOCATION)
    # see if the results are the same
    output_trial = emulator(inputs)
    assert np.all(abs(output_trial - output_true) <= EPS)


