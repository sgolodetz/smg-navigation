import numpy as np

from smg.pyoctomap import OcTree, OcTreeNode, Vector3

from scipy.interpolate import CubicSpline
from typing import List, Optional, Tuple


# HELPER TYPES

PathNode = Tuple[int, int, int]


# MAIN CLASS

class PathUtil:
    """Utility functions related to paths."""

    # PUBLIC STATIC METHODS

    @staticmethod
    def from_numpy(v: np.ndarray) -> Vector3:
        return Vector3(v[0], v[1], v[2])

    @staticmethod
    def interpolate(path: np.ndarray, *, smoothed_length: int = 100) -> np.ndarray:
        x: List[int] = np.arange(len(path))
        cs: CubicSpline = CubicSpline(x, path, bc_type='clamped')
        return cs(np.linspace(0, len(path) - 1, smoothed_length))

    @staticmethod
    def is_traversible(path: np.ndarray, source: int, dest: int, tree: OcTree) -> bool:
        source_node: PathNode = PathUtil.pos_to_node(PathUtil.from_numpy(path[source, :]), tree)
        dest_node: PathNode = PathUtil.pos_to_node(PathUtil.from_numpy(path[dest, :]), tree)

        print(source_node)
        print(dest_node)

        # TODO

        return True

    @staticmethod
    def l2_distance(v1: Vector3, v2: Vector3) -> float:
        return np.linalg.norm(PathUtil.to_numpy(v1) - PathUtil.to_numpy(v2))

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
    def node_to_vpos(node: PathNode, tree: OcTree) -> Vector3:
        voxel_size: float = tree.get_resolution()
        half_voxel_size: float = voxel_size / 2.0
        return Vector3(
            node[0] * voxel_size + half_voxel_size,
            node[1] * voxel_size + half_voxel_size,
            node[2] * voxel_size + half_voxel_size
        )

    @staticmethod
    def node_to_vpos_np(node: PathNode, tree: OcTree) -> np.ndarray:
        return PathUtil.to_numpy(PathUtil.node_to_vpos(node, tree))

    @staticmethod
    def occupancy_status(node: PathNode, tree: OcTree) -> str:
        # FIXME: Use an enumeration for the return values.
        vpos: Vector3 = PathUtil.node_to_vpos(node, tree)
        octree_node: Optional[OcTreeNode] = tree.search(vpos)
        if octree_node is None:
            return "Unknown"
        else:
            occupied: bool = tree.is_node_occupied(octree_node)
            return "Occupied" if occupied else "Free"

    @staticmethod
    def pos_to_node(pos: Vector3, tree: OcTree) -> PathNode:
        voxel_size: float = tree.get_resolution()
        return \
            int(np.round(pos.x // voxel_size)), \
            int(np.round(pos.y // voxel_size)), \
            int(np.round(pos.z // voxel_size))

    @staticmethod
    def pull_strings(path: np.ndarray, tree: OcTree) -> np.ndarray:
        pulled_path: List[np.ndarray] = []

        i: int = 0
        while i < len(path):
            pulled_path.append(path[i, :])

            j: int = i + 2
            while j < len(path) and PathUtil.is_traversible(path, i, j, tree):
                j += 1

            i = j - 1

        return np.vstack(pulled_path)

    @staticmethod
    def to_numpy(v: Vector3) -> np.ndarray:
        return np.array([v.x, v.y, v.z])
