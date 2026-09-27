from dataclasses import dataclass, field
from typing import Protocol
import numpy as np

@dataclass
class CoarseRequest:
    a: np.ndarray
    b: np.ndarray
    tile_a: dict
    tile_b: dict
    direction: tuple
    prior: object = None
    radius: float = .25
    budget: int = 1
    geometry: str = 'translation'
    mask_a: object = None
    mask_b: object = None

@dataclass
class CoarseResult:
    status: str
    candidates: list = field(default_factory=list)
    reason: str = ''
    diagnostics: dict = field(default_factory=dict)

@dataclass
class RefineRequest:
    a: np.ndarray
    b: np.ndarray
    displacement: np.ndarray
    config: dict

@dataclass
class RefineResult:
    status: str
    points_a: object = None
    points_b: object = None
    diagnostics: dict = field(default_factory=dict)
    frame: str = 'original_tile_pixel_centers_xy'

class CoarseRegistrar(Protocol):
    def register(self, request: CoarseRequest) -> CoarseResult: ...

class OverlapRefiner(Protocol):
    def refine(self, request: RefineRequest) -> RefineResult: ...

@dataclass
class WarpSolveRequest:
    tiles: list
    positions: list
    constraints: list
    config: dict

@dataclass
class TileWarpSet:
    meshes: list
    validation: dict
    convention: str = 'F_i(p)=P_i+p+u_i(p); inverse by fixed-point iteration'

class WarpReconciler(Protocol):
    def solve(self, request: WarpSolveRequest) -> TileWarpSet: ...
