import numpy as np

from collections import defaultdict, deque
from typing import Callable, Deque, Dict, List, Optional, Tuple

from smg.pyoctomap import OcTree, OcTreeNode, Vector3
from smg.utility import PriorityQueue


class AStarPathPlanner:
    """A path planner for Octomaps based on A*."""

    # NESTED TYPES

    Node = Tuple[int, int, int]

    # CONSTRUCTOR

    def __init__(self, tree: OcTree, neighbours: Callable[[Node], List[Node]]):
        self.__neighbours: Callable[[AStarPathPlanner.Node], List[AStarPathPlanner.Node]] = neighbours
        self.__tree: OcTree = tree

    # PUBLIC STATIC METHODS

    @staticmethod
    def neighbours4(node: Node) -> List[Node]:
        x, y, z = node
        return [
            (x, y, z - 1),
            (x - 1, y, z),
            (x + 1, y, z),
            (x, y, z + 1)
        ]

    @staticmethod
    def neighbours8(node: Node) -> List[Node]:
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

    # PUBLIC METHODS

    def plan_path(self, *, source, goal, d: Optional[Callable[[Vector3, Vector3], float]] = None,
                  h: Optional[Callable[[Vector3, Vector3], float]] = None) -> Optional[Deque[np.ndarray]]:
        # Based on an amalgam of:
        #
        # 1) https://en.wikipedia.org/wiki/A*_search_algorithm
        # 2) https://www.redblobgames.com/pathfinding/a-star/implementation.html

        if d is None:
            d = self.__l2_distance
        if h is None:
            h = self.__l2_distance

        source_node: AStarPathPlanner.Node = self.__pos_to_node(Vector3(*source))
        goal_node: AStarPathPlanner.Node = self.__pos_to_node(Vector3(*goal))

        source_vpos: Vector3 = self.__node_to_vpos(source_node)
        goal_vpos: Vector3 = self.__node_to_vpos(goal_node)

        print(source, source_node, self.__occupancy_status(source_node))
        print(goal, goal_node, self.__occupancy_status(goal_node))

        g_scores: Dict[AStarPathPlanner.Node, float] = defaultdict(lambda: np.infty)
        g_scores[source_node] = 0.0
        came_from: Dict[AStarPathPlanner.Node, Optional[AStarPathPlanner.Node]] = {source_node: None}

        frontier: PriorityQueue[AStarPathPlanner.Node, float, type(None)] \
            = PriorityQueue[AStarPathPlanner.Node, float, type(None)]()
        frontier.insert(source_node, h(source_vpos, goal_vpos), None)

        while not frontier.empty():
            current_node: AStarPathPlanner.Node = frontier.top().ident
            if current_node == goal_node:
                path: Deque[np.ndarray] = self.__reconstruct_path(goal_node, came_from)
                path.appendleft(source)
                path.append(goal)
                return path

            frontier.pop()
            current_vpos: Vector3 = self.__node_to_vpos(current_node)

            for neighbour_node in self.__neighbours(current_node):
                if self.__occupancy_status(neighbour_node) != "Free":
                    continue

                neighbour_vpos: Vector3 = self.__node_to_vpos(neighbour_node)
                tentative_cost: float = g_scores[current_node] + d(current_vpos, neighbour_vpos)
                if tentative_cost < g_scores[neighbour_node]:
                    g_scores[neighbour_node] = tentative_cost
                    came_from[neighbour_node] = current_node
                    f_score: float = tentative_cost + h(neighbour_vpos, goal_vpos)
                    if frontier.contains(neighbour_node):
                        frontier.update_key(neighbour_node, f_score)
                    else:
                        frontier.insert(neighbour_node, f_score, None)

        return None

    # PRIVATE METHODS

    def __node_to_vpos(self, node: Node) -> Vector3:
        voxel_size: float = self.__tree.get_resolution()
        half_voxel_size: float = voxel_size / 2.0
        return Vector3(
            node[0] * voxel_size + half_voxel_size,
            node[1] * voxel_size + half_voxel_size,
            node[2] * voxel_size + half_voxel_size
        )

    def __node_to_vpos_np(self, node: Node) -> np.ndarray:
        return AStarPathPlanner.__to_numpy(self.__node_to_vpos(node))

    def __occupancy_status(self, node: Node) -> str:
        vpos: Vector3 = self.__node_to_vpos(node)
        octree_node: Optional[OcTreeNode] = self.__tree.search(vpos)
        if octree_node is None:
            return "Unknown"
        else:
            occupied: bool = self.__tree.is_node_occupied(octree_node)
            return "Occupied" if occupied else "Free"

    def __pos_to_node(self, pos: Vector3) -> Node:
        voxel_size: float = self.__tree.get_resolution()
        return \
            int(np.round(pos.x // voxel_size)), \
            int(np.round(pos.y // voxel_size)), \
            int(np.round(pos.z // voxel_size))

    def __reconstruct_path(self, goal_node: Node, came_from: Dict[Node, Optional[Node]]) -> Deque[np.ndarray]:
        path: Deque[np.ndarray] = deque()
        current_node: Optional[AStarPathPlanner.Node] = goal_node
        while current_node is not None:
            path.appendleft(self.__node_to_vpos_np(current_node))
            current_node = came_from.get(current_node)
        return path

    # PRIVATE STATIC METHODS

    @staticmethod
    def __from_numpy(v: np.ndarray) -> Vector3:
        return Vector3(v[0], v[1], v[2])

    @staticmethod
    def __l2_distance(pos: Vector3, goal_pos: Vector3) -> float:
        return np.linalg.norm(AStarPathPlanner.__to_numpy(pos) - AStarPathPlanner.__to_numpy(goal_pos))

    @staticmethod
    def __to_numpy(v: Vector3) -> np.ndarray:
        return np.array([v.x, v.y, v.z])
