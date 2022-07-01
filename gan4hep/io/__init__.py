import numpy as np
from typing import Tuple, List

__all__ = (
    "DiMuonsReader",
    "HerwigReader",
)
GAN_INPUT_DATA_TYPE = Tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str]]

from gan4hep.io.dimuons import read as DiMuonsReader
from gan4hep.io.herwig import read as HerwigReader

from gan4hep.io.herwig import convert_cluster_decay as HerwigConvert