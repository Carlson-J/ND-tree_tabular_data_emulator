from nd_emulator.compact_mapping import load_compact_mapping, CompactMapping, save_compact_mapping
from nd_emulator.parameter_struct import Parameters
import numpy as np


def test_loading_and_saving():
    """
    GIVEN: CompactMapping struct
    WHEN: it is saved and loaded
    THEN: it is the same as before it was saved
    """
    # Create param and mapping objects
    params = Parameters(2, np.array(['linear', 'log']), np.array([20, 3]), 0.1,
                        np.array([{'type': 'nd-linear', 'transforms': None}, {'type': 'nd-linear', 'transforms': None}]),
                        10, False, np.array([[0, 1], [5, 2]]))
    compact_mapping = CompactMapping(np.array([1, 4, 6]), np.array([2, 5, 7]), np.array([0, 1]),
                                     [np.array([[1.3, 1.5]]), np.array([[1.1, 1, 9]])], params)

    # save and load it
    save_compact_mapping(compact_mapping, '.', 'test_io')
    compact_mapping_new = load_compact_mapping('./test_io_table.hdf5')

    # compare things
    assert np.all(compact_mapping.offsets == compact_mapping_new.offsets)
    assert np.all(compact_mapping.encoding_array == compact_mapping_new.encoding_array)
    assert np.all(compact_mapping.index_array == compact_mapping_new.index_array)
    for i in range(len(compact_mapping.model_arrays)):
        assert np.all(compact_mapping.model_arrays[i] == compact_mapping_new.model_arrays[i])
    assert np.all(compact_mapping.params.max_depth == compact_mapping_new.params.max_depth)
    assert np.all(compact_mapping.params.spacing == compact_mapping_new.params.spacing)
    assert np.all(compact_mapping.params.error_threshold == compact_mapping_new.params.error_threshold)
    for i in range(len(compact_mapping.params.model_classes)):
        assert np.all(compact_mapping.params.model_classes[i]['type'] == compact_mapping_new.params.model_classes[i]['type'])
    assert np.all(compact_mapping.params.max_test_points == compact_mapping_new.params.max_test_points)
    assert np.all(compact_mapping.params.relative_error == compact_mapping_new.params.relative_error)
    assert np.all(compact_mapping.params.domain == compact_mapping_new.params.domain)