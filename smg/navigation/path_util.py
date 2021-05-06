import numpy as np

from smg.pyoctomap import OcTree, OcTreeNode, Vector3

from scipy.interpolate import PchipInterpolator
from typing import Callable, List, Optional, Tuple


# HELPER TYPES

PathNode = Tuple[int, int, int]


# MAIN CLASS

class PathUtil:
    """Utility functions related to paths."""

    # PUBLIC STATIC HOOKS

    neighbours: Callable[[PathNode], List[PathNode]] = lambda n: PathUtil.neighbours6(n)
    node_is_free: Callable[[PathNode, OcTree], bool] = lambda n, t: PathUtil.occupancy_status(n, t) == "Free"

    # PUBLIC STATIC METHODS

    @staticmethod
    def from_numpy(v: np.ndarray) -> Vector3:
        return Vector3(*v)

    @staticmethod
    def interpolate(path: np.ndarray, *, smoothed_length: int = 100) -> np.ndarray:
        x: np.ndarray = np.arange(len(path))
        cs: PchipInterpolator = PchipInterpolator(x, path)
        return cs(np.linspace(0, len(path) - 1, smoothed_length))

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

    # noinspection PyCallByClass
    @staticmethod
    def node_is_traversible(node: PathNode, tree: OcTree, *, use_clearance: bool) -> bool:
        if not PathUtil.node_is_free(node, tree):
            return False

        if use_clearance:
            for neighbour_node in PathUtil.neighbours(node):
                if not PathUtil.node_is_free(neighbour_node, tree):
                    return False

                # for nn_node in PathUtil.neighbours(neighbour_node):
                #     if not PathUtil.node_is_free(nn_node, tree):
                #         return False

        return True

    @staticmethod
    def node_to_vpos(node: PathNode, tree: OcTree) -> np.ndarray:
        voxel_size: float = tree.get_resolution()
        half_voxel_size: float = voxel_size / 2.0
        return np.array([node[i] * voxel_size + half_voxel_size for i in range(3)])

    @staticmethod
    def occupancy_status(node: PathNode, tree: OcTree) -> str:
        # FIXME: Use an enumeration for the return values.
        vpos: np.ndarray = PathUtil.node_to_vpos(node, tree)
        octree_node: Optional[OcTreeNode] = tree.search(PathUtil.from_numpy(vpos))
        if octree_node is None:
            return "Unknown"
        else:
            occupied: bool = tree.is_node_occupied(octree_node)
            return "Occupied" if occupied else "Free"

    @staticmethod
    def path_is_traversible(path: np.ndarray, source: int, dest: int, tree: OcTree, *, use_clearance: bool) -> bool:
        source_node: PathNode = PathUtil.pos_to_node(path[source, :], tree)
        dest_node: PathNode = PathUtil.pos_to_node(path[dest, :], tree)

        source_vpos: np.ndarray = PathUtil.node_to_vpos(source_node, tree)
        dest_vpos: np.ndarray = PathUtil.node_to_vpos(dest_node, tree)

        # TODO: Fix and optimise this.
        prev_node: Optional[PathNode] = None
        for t in np.linspace(0.0, 1.0, 101):
            pos: np.ndarray = source_vpos * (1 - t) + dest_vpos * t
            node: PathNode = PathUtil.pos_to_node(pos, tree)
            if prev_node is None or node != prev_node:
                prev_node = node
                # noinspection PyCallByClass
                if not PathUtil.node_is_traversible(node, tree, use_clearance=use_clearance):
                    return False

        return True

    @staticmethod
    def pos_to_node(pos: np.ndarray, tree: OcTree) -> PathNode:
        voxel_size: float = tree.get_resolution()
        return tuple(np.round(pos // voxel_size).astype(int))

    @staticmethod
    def pull_strings(path: np.ndarray, tree: OcTree, *, use_clearance: bool) -> np.ndarray:
        pulled_path: List[np.ndarray] = []

        i: int = 0
        while i < len(path):
            pulled_path.append(path[i, :])

            j: int = i + 2
            while j < len(path) and PathUtil.path_is_traversible(path, i, j, tree, use_clearance=use_clearance):
                j += 1

            i = j - 1

        return np.vstack(pulled_path)
