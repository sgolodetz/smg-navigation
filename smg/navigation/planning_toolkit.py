import numpy as np

from smg.pyoctomap import OcTree, OcTreeNode, Vector3

from scipy.interpolate import PchipInterpolator
from typing import Callable, List, Optional, Tuple


# HELPER TYPES

PathNode = Tuple[int, int, int]


# MAIN CLASS

class PlanningToolkit:
    """TODO"""

    # CONSTRUCTOR

    def __init__(self, *, neighbours: Optional[Callable[[PathNode], List[PathNode]]] = None,
                 node_is_free: Optional[Callable[[PathNode, OcTree], bool]]):
        if neighbours is None:
            neighbours = PlanningToolkit.neighbours6
        if node_is_free is None:
            node_is_free = lambda n, t: PlanningToolkit.occupancy_status(n, t) == "Free"

        self.neighbours: Callable[[PathNode], List[PathNode]] = neighbours
        self.node_is_free: Callable[[PathNode, OcTree], bool] = node_is_free

    # PUBLIC STATIC METHODS

    @staticmethod
    def interpolate_path(path: np.ndarray, *, new_length: int = 100) -> np.ndarray:
        x: np.ndarray = np.arange(len(path))
        cs: PchipInterpolator = PchipInterpolator(x, path)
        return cs(np.linspace(0, len(path) - 1, new_length))

    @staticmethod
    def l1_distance(v1: np.ndarray, v2: np.ndarray) -> float:
        return np.linalg.norm(v1 - v2, ord=1)

    @staticmethod
    def l2_distance(v1: np.ndarray, v2: np.ndarray) -> float:
        return np.linalg.norm(v1 - v2)

    @staticmethod
    def neighbours4(node: PathNode) -> List[PathNode]:
        x, y, z = node
        return [
            (x, y, z - 1),
            (x - 1, y, z),
            (x + 1, y, z),
            (x, y, z + 1)
        ]

    @staticmethod
    def neighbours6(node: PathNode) -> List[PathNode]:
        x, y, z = node
        return [
            (x, y, z - 1),
            (x, y - 1, z),
            (x, y + 1, z),
            (x - 1, y, z),
            (x + 1, y, z),
            (x, y, z + 1)
        ]

    @staticmethod
    def neighbours8(node: PathNode) -> List[PathNode]:
        x, y, z = node
        return [
            (x - 1, y, z - 1),
            (x, y, z - 1),
            (x + 1, y, z - 1),
            (x - 1, y, z),
            (x + 1, y, z),
            (x - 1, y, z + 1),
            (x, y, z + 1),
            (x + 1, y, z + 1)
        ]

    @staticmethod
    def neighbours26(node: PathNode) -> List[PathNode]:
        x, y, z = node
        return [
            (x - 1, y - 1, z - 1),
            (x,     y - 1, z - 1),
            (x + 1, y - 1, z - 1),
            (x - 1, y,     z - 1),
            (x,     y,     z - 1),
            (x + 1, y,     z - 1),
            (x - 1, y + 1, z - 1),
            (x,     y + 1, z - 1),
            (x + 1, y + 1, z - 1),
            (x - 1, y - 1, z),
            (x,     y - 1, z),
            (x + 1, y - 1, z),
            (x - 1, y,     z),
            (x + 1, y,     z),
            (x - 1, y + 1, z),
            (x,     y + 1, z),
            (x + 1, y + 1, z),
            (x - 1, y - 1, z + 1),
            (x,     y - 1, z + 1),
            (x + 1, y - 1, z + 1),
            (x - 1, y,     z + 1),
            (x,     y,     z + 1),
            (x + 1, y,     z + 1),
            (x - 1, y + 1, z + 1),
            (x,     y + 1, z + 1),
            (x + 1, y + 1, z + 1)
        ]

    @staticmethod
    def node_to_vpos(node: PathNode, tree: OcTree) -> np.ndarray:
        voxel_size: float = tree.get_resolution()
        half_voxel_size: float = voxel_size / 2.0
        return np.array([node[i] * voxel_size + half_voxel_size for i in range(3)])

    @staticmethod
    def occupancy_status(node: PathNode, tree: OcTree) -> str:
        # FIXME: Use an enumeration for the return values.
        vpos: np.ndarray = PlanningToolkit.node_to_vpos(node, tree)
        octree_node: Optional[OcTreeNode] = tree.search(Vector3(*vpos))
        if octree_node is None:
            return "Unknown"
        else:
            occupied: bool = tree.is_node_occupied(octree_node)
            return "Occupied" if occupied else "Free"

    @staticmethod
    def pos_to_node(pos: np.ndarray, tree: OcTree) -> PathNode:
        voxel_size: float = tree.get_resolution()
        return tuple(np.round(pos // voxel_size).astype(int))

    # PUBLIC METHODS

    def node_is_traversible(self, node: PathNode, tree: OcTree, *, use_clearance: bool) -> bool:
        if not self.node_is_free(node, tree):
            return False

        if use_clearance:
            for neighbour_node in self.neighbours(node):
                if not self.node_is_free(neighbour_node, tree):
                    return False

        return True

    def path_is_traversible(self, path: np.ndarray, source: int, dest: int, tree: OcTree, *,
                            use_clearance: bool) -> bool:
        source_node: PathNode = PlanningToolkit.pos_to_node(path[source, :], tree)
        dest_node: PathNode = PlanningToolkit.pos_to_node(path[dest, :], tree)

        source_vpos: np.ndarray = PlanningToolkit.node_to_vpos(source_node, tree)
        dest_vpos: np.ndarray = PlanningToolkit.node_to_vpos(dest_node, tree)

        # TODO: Fix and optimise this.
        prev_node: Optional[PathNode] = None
        for t in np.linspace(0.0, 1.0, 101):
            pos: np.ndarray = source_vpos * (1 - t) + dest_vpos * t
            node: PathNode = PlanningToolkit.pos_to_node(pos, tree)
            if prev_node is None or node != prev_node:
                prev_node = node
                if not self.node_is_traversible(node, tree, use_clearance=use_clearance):
                    return False

        return True

    def pull_strings(self, path: np.ndarray, tree: OcTree, *, use_clearance: bool) -> np.ndarray:
        pulled_path: List[np.ndarray] = []

        i: int = 0
        while i < len(path):
            pulled_path.append(path[i, :])

            j: int = i + 2
            while j < len(path) and self.path_is_traversible(path, i, j, tree, use_clearance=use_clearance):
                j += 1

            i = j - 1

        return np.vstack(pulled_path)
