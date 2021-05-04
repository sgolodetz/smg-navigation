import numpy as np

from typing import List, Optional

from smg.pyoctomap import OcTree, OcTreeNode, Vector3


class AStarPathPlanner:
    """A path planner for Octomaps based on A*."""

    # CONSTRUCTOR

    def __init__(self, tree: OcTree):
        self.__tree: OcTree = tree

    # PUBLIC METHODS

    def plan_path(self, *, start, goal) -> List[np.ndarray]:
        start_coords: Vector3 = self.__pos_to_coords(Vector3(*start))
        goal_coords: Vector3 = self.__pos_to_coords(Vector3(*goal))
        print(start_coords, self.__occupancy_status(start_coords))
        print(goal_coords, self.__occupancy_status(goal_coords))
        return []

    # PRIVATE METHODS

    def __coords_to_vpos(self, voxel_coords: Vector3) -> Vector3:
        voxel_size: float = self.__tree.get_resolution()
        return voxel_coords * voxel_size

    def __occupancy_status(self, coords: Vector3) -> str:
        vpos: Vector3 = self.__coords_to_vpos(coords)
        node: Optional[OcTreeNode] = self.__tree.search(vpos)
        if node is None:
            return "Unknown"
        else:
            occupied: bool = self.__tree.is_node_occupied(node)
            return "Occupied" if occupied else "Free"

    def __pos_to_coords(self, pos: Vector3) -> Vector3:
        voxel_size: float = self.__tree.get_resolution()
        return Vector3(pos.x // voxel_size, pos.y // voxel_size, pos.z // voxel_size)

    def __pos_to_vpos(self, pos: Vector3) -> Vector3:
        voxel_size: float = self.__tree.get_resolution()
        half_voxel_size: float = voxel_size / 2.0
        x: float = (pos.x // voxel_size) * voxel_size + half_voxel_size
        y: float = (pos.y // voxel_size) * voxel_size + half_voxel_size
        z: float = (pos.z // voxel_size) * voxel_size + half_voxel_size
        return Vector3(x, y, z)
