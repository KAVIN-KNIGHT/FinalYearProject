import pytest


def test_torch_and_pyg_imports():
    import torch
    import torch_geometric

    assert torch.__version__ is not None
    assert torch_geometric.__version__ is not None


def test_satsim_imports():
    import satsim
    from satsim.config import SimConfig
    from satsim.logging import setup_logging

    assert satsim.__version__ == "0.1.0"


def test_numeric_and_graph_imports():
    import numpy as np
    import scipy
    import networkx as nx
    import pandas as pd
    import pyarrow as pa
    import sgp4

    assert np.__version__ is not None
