
import numpy as np
from typing import Union, Any


class SpatialEncoder:
    """Encoder catorical data, particle IDs,
    into a vector separated by equal distance.
    Those particles appearing more often are assigned with large value.
    """
    def __init__(self):
        self.encoder = None

    def encode(self, x: Union[np.ndarray, float]) -> Union[np.ndarray, float]:
        if type(x) == np.ndarray:
            return np.array([self.encoder[x[i]] for i in range(x.shape[0])])
        else:
            return self.encoder[x]

    def __call__(self, x, *args: Any, **kwds: Any) -> Any:
        return self.encode(x)

    def train(self, x):
        all_classes, counts = np.unique(x, return_counts=True)

        # those appearing more are assigned with larger value
        classes = all_classes[np.argsort(counts)]
        num_classes = len(classes)
        self.step = 1/(num_classes-1)
        code = np.arange(0, 1+self.step, self.step)
        self.encoder = dict([(k, v) for k, v in zip(classes, code)])