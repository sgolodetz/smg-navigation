import numpy as np

from collections import defaultdict, deque
from typing import Callable, Deque, Dict, List, Optional

from smg.pyoctomap import OcTree, Vector3
from smg.utility import PriorityQueue

from .path_util import PathNode, PathUtil


class AStarPathPlanner:
    """A path planner for Octomaps based on A*."""

    # CONSTRUCTOR

    def __init__(self, tree: OcTree, neighbours: Callable[[PathNode], List[PathNode]]):
        self.__neighbours: Callable[[PathNode], List[PathNode]] = neighbours
        self.__tree: OcTree = tree

    # PUBLIC METHODS

    def plan_path(self, *, source, goal, d: Optional[Callable[[Vector3, Vector3], float]] = None,
                  h: Optional[Callable[[Vector3, Vector3], float]] = None, use_clearance: bool = False) \
            -> Optional[np.ndarray]:
        # Based on an amalgam of:
        #
        # 1) https://en.wikipedia.org/wiki/A*_search_algorithm
        # 2) https://www.redblobgames.com/pathfinding/a-star/implementation.html

        if d is None:
            d = PathUtil.l2_distance
        if h is None:
            h = PathUtil.l2_distance

        source_node: PathNode = PathUtil.pos_to_node(Vector3(*source), self.__tree)
        goal_node: PathNode = PathUtil.pos_to_node(Vector3(*goal), self.__tree)

        source_vpos: Vector3 = PathUtil.node_to_vpos(source_node, self.__tree)
        goal_vpos: Vector3 = PathUtil.node_to_vpos(goal_node, self.__tree)

        source_occupancy: str = PathUtil.occupancy_status(source_node, self.__tree)
        goal_occupancy: str = PathUtil.occupancy_status(goal_node, self.__tree)
        print(source, source_node, source_occupancy)
        print(goal, goal_node, goal_occupancy)
        if not (self.__node_is_traversible(source_node, use_clearance) and
                self.__node_is_traversible(goal_node, use_clearance)):
            return None

        g_scores: Dict[PathNode, float] = defaultdict(lambda: np.infty)
        g_scores[source_node] = 0.0
        came_from: Dict[PathNode, Optional[PathNode]] = {source_node: None}

        frontier: PriorityQueue[PathNode, float, type(None)] = PriorityQueue[PathNode, float, type(None)]()
        frontier.insert(source_node, h(source_vpos, goal_vpos), None)

        while not frontier.empty():
            current_node: PathNode = frontier.top().ident
            if current_node == goal_node:
                path: Deque[np.ndarray] = self.__reconstruct_path(goal_node, came_from)
                path.appendleft(source)
                path.append(goal)
                return np.vstack(path)

            frontier.pop()
            current_vpos: Vector3 = PathUtil.node_to_vpos(current_node, self.__tree)

            for neighbour_node in self.__neighbours(current_node):
                if not self.__node_is_traversible(neighbour_node, use_clearance):
                    continue

                neighbour_vpos: Vector3 = PathUtil.node_to_vpos(neighbour_node, self.__tree)
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

    def __node_is_traversible(self, node: PathNode, use_clearance: bool) -> bool:
        if not PathUtil.node_is_free(node, self.__tree):
            return False

        if use_clearance:
            for neighbour_node in self.__neighbours(node):
                if not PathUtil.node_is_free(neighbour_node, self.__tree):
                    return False

        return True

    def __reconstruct_path(self, goal_node: PathNode, came_from: Dict[PathNode, Optional[PathNode]]) \
            -> Deque[np.ndarray]:
        path: Deque[np.ndarray] = deque()
        current_node: Optional[PathNode] = goal_node
        while current_node is not None:
            path.appendleft(PathUtil.node_to_vpos_np(current_node, self.__tree))
            current_node = came_from.get(current_node)
        return path
