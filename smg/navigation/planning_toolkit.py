import numpy as np

from smg.pyoctomap import OcTree, OcTreeNode, Vector3

from scipy.interpolate import Akima1DInterpolator, PchipInterpolator
from typing import Callable, List, Optional, Tuple


# HELPER TYPES

PathNode = Tuple[int, int, int]


# MAIN CLASS

class PlanningToolkit:
    """A toolkit of useful functions for use by path planners that work with Octomap scenes."""

    # CONSTRUCTOR

    def __init__(self, tree: OcTree, *, neighbours: Optional[Callable[[PathNode], List[PathNode]]] = None,
                 node_is_free: Optional[Callable[[PathNode], bool]] = None):
        """
        Construct a planning toolkit.

        .. note::
            If the neighbours function isn't explicitly specified, 6-connected neighbours will be computed.

        :param tree:            The Octomap octree over which paths are to be planned.
        :param neighbours:      An optional function specifying how the neighbours of a path node are to be computed.
        :param node_is_free:    An optional function specifying what counts as a "free" path node.
        """
        self.__tree: OcTree = tree

        if neighbours is None:
            neighbours = PlanningToolkit.neighbours6
        if node_is_free is None:
            node_is_free = lambda n: self.occupancy_status(n) == "Free"

        self.neighbours: Callable[[PathNode], List[PathNode]] = neighbours
        self.node_is_free: Callable[[PathNode], bool] = node_is_free

    # PUBLIC STATIC METHODS

    @staticmethod
    def interpolate_path(path: np.ndarray, *, new_length: int = 100) -> np.ndarray:
        x: np.ndarray = np.arange(len(path))
        # cs: PchipInterpolator = PchipInterpolator(x, path)
        cs: Akima1DInterpolator = Akima1DInterpolator(x, path)
        return cs(np.linspace(0, len(path) - 1, new_length))

    @staticmethod
    def l1_distance(v1: np.ndarray, v2: np.ndarray) -> float:
        """
        Compute the L1 distance between the two specified vectors.

        :param v1:  The first vector.
        :param v2:  The second vector.
        :return:    The L1 distance between the two vectors.
        """
        return np.linalg.norm(v1 - v2, ord=1)

    @staticmethod
    def l1_penalise_vertical(mu: float) -> Callable[[np.ndarray, np.ndarray], float]:
        def inner(v1: np.ndarray, v2: np.ndarray) -> float:
            dx: float = abs(v2[0] - v1[0])
            dy: float = abs(v2[1] - v1[1])
            dz: float = abs(v2[2] - v1[2])
            return dx + mu * dy + dz

        return inner

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

    # PUBLIC METHODS

    def node_is_traversible(self, node: PathNode, *, use_clearance: bool) -> bool:
        if not self.node_is_free(node):
            return False

        if use_clearance:
            for neighbour_node in self.neighbours(node):
                if not self.node_is_free(neighbour_node):
                    return False

        return True

    def node_to_vpos(self, node: PathNode) -> np.ndarray:
        voxel_size: float = self.__tree.get_resolution()
        half_voxel_size: float = voxel_size / 2.0
        return np.array([node[i] * voxel_size + half_voxel_size for i in range(3)])

    def occupancy_status(self, node: PathNode) -> str:
        # FIXME: Use an enumeration for the return values.
        vpos: np.ndarray = self.node_to_vpos(node)
        octree_node: Optional[OcTreeNode] = self.__tree.search(Vector3(*vpos))
        if octree_node is None:
            return "Unknown"
        else:
            occupied: bool = self.__tree.is_node_occupied(octree_node)
            return "Occupied" if occupied else "Free"

    def path_is_traversible(self, path: np.ndarray, source: int, dest: int, *, use_clearance: bool) -> bool:
        source_node: PathNode = self.pos_to_node(path[source, :])
        dest_node: PathNode = self.pos_to_node(path[dest, :])

        source_vpos: np.ndarray = self.node_to_vpos(source_node)
        dest_vpos: np.ndarray = self.node_to_vpos(dest_node)

        # TODO: Fix and optimise this.
        prev_node: Optional[PathNode] = None
        for t in np.linspace(0.0, 1.0, 101):
            pos: np.ndarray = source_vpos * (1 - t) + dest_vpos * t
            node: PathNode = self.pos_to_node(pos)
            if prev_node is None or node != prev_node:
                prev_node = node
                if not self.node_is_traversible(node, use_clearance=use_clearance):
                    return False

        return True

    def pos_to_node(self, pos: np.ndarray) -> PathNode:
        voxel_size: float = self.__tree.get_resolution()
        return tuple(np.round(pos // voxel_size).astype(int))

    def pull_strings(self, path: np.ndarray, *, use_clearance: bool) -> np.ndarray:
        pulled_path: List[np.ndarray] = []

        i: int = 0
        while i < len(path):
            pulled_path.append(path[i, :])

            j: int = i + 2
            while j < len(path) and self.path_is_traversible(path, i, j, use_clearance=use_clearance):
                j += 1

            i = j - 1

        return np.vstack(pulled_path)
