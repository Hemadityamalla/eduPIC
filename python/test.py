import numpy as np
from dataclasses import dataclass


#Test script for dataclass

@dataclass
class Test:
    name: str
    vals: np.ndarray = np.zeros(1000)