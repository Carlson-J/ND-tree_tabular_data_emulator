import pytest
import numpy as np


@pytest.fixture
def dataset_2d():
    # Create a 2D grid spaced in log and linear space
    Nx0 = 2**3 + 1
    x0 = np.linspace(0, 1, Nx0)
    Nx1 = 2**4 + 1
    x1 = np.logspace(0, 1, Nx1)
    X0, X1 = np.meshgrid(x0, x1)

    return {
        'f': X0*X1,
        'df_x0': X1,
        'df_x1': X0,
        'df_x0_x1': np.ones_like(X1)
    }
