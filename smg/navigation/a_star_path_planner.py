import numpy as np

from typing import Dict, List, Optional, Tuple

from smg.pyoctomap import OcTree, OcTreeNode, Vector3


class AStarPathPlanner:
    """A path planner for Octomaps based on A*."""

    # CONSTRUCTOR

    def __init__(self, tree: OcTree):
        self.__tree: OcTree = tree

    # PUBLIC METHODS

    def plan_path(self, *, start, goal) -> List[np.ndarray]:
        # See: https://en.wikipedia.org/wiki/A*_search_algorithm
        start_coords: Tuple[int, int, int] = self.__pos_to_coords(Vector3(*start))
        goal_coords: Tuple[int, int, int] = self.__pos_to_coords(Vector3(*goal))

        print(start_coords, self.__occupancy_status(start_coords))
        print(goal_coords, self.__occupancy_status(goal_coords))

        # TODO

        return []

    # PRIVATE METHODS

    def __coords_to_vpos(self, coords: Tuple[int, int, int]) -> Vector3:
        voxel_size: float = self.__tree.get_resolution()
        half_voxel_size: float = voxel_size / 2.0
        return Vector3(
            coords[0] * voxel_size + half_voxel_size,
            coords[1] * voxel_size + half_voxel_size,
            coords[2] * voxel_size + half_voxel_size
        )

    def __occupancy_status(self, coords: Tuple[int, int, int]) -> str:
        vpos: Vector3 = self.__coords_to_vpos(coords)
        node: Optional[OcTreeNode] = self.__tree.search(vpos)
        if node is None:
            return "Unknown"
        else:
            occupied: bool = self.__tree.is_node_occupied(node)
            return "Occupied" if occupied else "Free"

    def __pos_to_coords(self, pos: Vector3) -> Tuple[int, int, int]:
        voxel_size: float = self.__tree.get_resolution()
        return \
            int(np.round(pos.x // voxel_size)), \
            int(np.round(pos.y // voxel_size)), \
            int(np.round(pos.z // voxel_size))
