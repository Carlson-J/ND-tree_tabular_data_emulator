from nd_emulator.nd_tree import ND_Tree, create_children_nodes
import pytest
import numpy as np


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