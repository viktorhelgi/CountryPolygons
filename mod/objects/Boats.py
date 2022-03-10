from dataclasses import dataclass, field
import data.inputs as gd
import numpy as np
from math import sqrt





@dataclass
class Boat:
    name : str
    length: float = field(init=False, metadata={'units':'meters'})
    beam: float = field(init=False)
    depth: float = field(init=False)
    power: int = field(init=False)
    bt: float = field(init=False)
    nt: float = field(init=False)
    trim: float = field(init=False)
    deadrise: float = field(init=False)
    craft_displ: float = field(init=False)
    maxspeedmotor: float = field(init=False)
    speed: np.ndarray = field(init=False)
    fuel_burn: np.ndarray = field(init=False)
    def __post_init__(self):
        for key, item in gd.boats[self.name].items():
            setattr(self, key, item)

    @property
    def plan_hradi(self):
        return 0.9 * 1.34 * sqrt(self.length)
    @property
    def load_coeff(self):
        return float(self.craft_displ/(1.10*self.beam**3))


