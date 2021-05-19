from nd_emulator.nd_tree import ND_Tree, create_children_nodes
import pytest


@pytest.mark.dependency()
def test_init(dataset_2d):
    """
    GIVEN: nd tree paramaters
    WHEN: tree is built
    THEN: No errors occur
    """
    model_classes = ['nd-linear']
    max_depth = 3
    error_threshold = 0.1
    data, domain, dims, spacing = dataset_2d
    try:
        ND_Tree(data, max_depth, domain, dims, spacing, error_threshold, model_classes)
    except Exception as err:
        assert False, f'initializing ND_Tree raised an exception: {err}'


def test_create_children_nodes(default_root_node_2d):
    """
    GIVEN: a node
    WHEN: split nodes
    THEN: node has correct number of children and parameters
    """
    node = default_root_node_2d
    num_dims = len(node['mask'])

    # create new children nodes
    create_children_nodes(node)

    # check to make sure everything is correct
    assert len(node['children']) == num_dims
    for i in range(num_dims):
        node2 = node['children'][i]
        assert node['id'] + f'{i}' == node2['id']
        for j in range(num_dims):
            assert node2['domain'][j] == [0, 0.5]
        assert node['children'] is None
        assert node['model'] is None
        assert node['error'] is None
        assert node['children'] is None
        assert node['mask'] is not None
