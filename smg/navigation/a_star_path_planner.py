import numpy as np

from collections import defaultdict, deque
from typing import Callable, Deque, Dict, Optional

from smg.pyoctomap import OcTree
from smg.utility import PriorityQueue

from .path_util import PathNode, PathUtil


class AStarPathPlanner:
    """A path planner for Octomaps based on A*."""

    # CONSTRUCTOR

    def __init__(self, tree: OcTree):
        self.__tree: OcTree = tree

    # PUBLIC METHODS

    def plan_path(self, *, source, goal, d: Optional[Callable[[np.ndarray, np.ndarray], float]] = None,
                  h: Optional[Callable[[np.ndarray, np.ndarray], float]] = None, use_clearance: bool = False) \
            -> Optional[np.ndarray]:
        # Based on an amalgam of:
        #
        # 1) https://en.wikipedia.org/wiki/A*_search_algorithm
        # 2) https://www.redblobgames.com/pathfinding/a-star/implementation.html

        if d is None:
            d = PathUtil.l2_distance
        if h is None:
            h = PathUtil.l2_distance

        source_node: PathNode = PathUtil.pos_to_node(source, self.__tree)
        goal_node: PathNode = PathUtil.pos_to_node(goal, self.__tree)

        source_vpos: np.ndarray = PathUtil.node_to_vpos(source_node, self.__tree)
        goal_vpos: np.ndarray = PathUtil.node_to_vpos(goal_node, self.__tree)

        source_occupancy: str = PathUtil.occupancy_status(source_node, self.__tree)
        goal_occupancy: str = PathUtil.occupancy_status(goal_node, self.__tree)
        print(source, source_node, source_occupancy)
        print(goal, goal_node, goal_occupancy)
        if not (PathUtil.node_is_traversible(source_node, self.__tree, use_clearance=use_clearance) and
                PathUtil.node_is_traversible(goal_node, self.__tree, use_clearance=use_clearance)):
            return None

        g_scores: Dict[PathNode, float] = defaultdict(lambda: np.infty)
        g_scores[source_node] = 0.0
        came_from: Dict[PathNode, Optional[PathNode]] = {source_node: None}

        frontier: PriorityQueue[PathNode, float, type(None)] = PriorityQueue[PathNode, float, type(None)]()
        frontier.insert(source_node, h(source_vpos, goal_vpos), None)

        while not frontier.empty():
            current_node: PathNode = frontier.top().ident
            current_vpos: np.ndarray = PathUtil.node_to_vpos(current_node, self.__tree)

            # if PathUtil.path_is_traversible(np.vstack([PathUtil.to_numpy(current_vpos), PathUtil.to_numpy(goal_vpos)]), 0, 1, self.__tree):
            #     path: Deque[np.ndarray] = self.__reconstruct_path(current_node, came_from)
            #     path.appendleft(source)
            #     path.append(PathUtil.node_to_vpos_np(current_node, self.__tree))
            #     path.append(goal)
            #     return np.vstack(path)

            if current_node == goal_node:
                path: Deque[np.ndarray] = self.__reconstruct_path(goal_node, came_from)
                path.appendleft(source)
                path.append(goal)
                return np.vstack(path)

            frontier.pop()

            for neighbour_node in PathUtil.neighbours(current_node):
                if not PathUtil.node_is_traversible(neighbour_node, self.__tree, use_clearance=use_clearance):
                    continue

                neighbour_vpos: np.ndarray = PathUtil.node_to_vpos(neighbour_node, self.__tree)
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

    def __reconstruct_path(self, goal_node: PathNode, came_from: Dict[PathNode, Optional[PathNode]]) \
            -> Deque[np.ndarray]:
        path: Deque[np.ndarray] = deque()
        current_node: Optional[PathNode] = goal_node
        while current_node is not None:
            path.appendleft(PathUtil.node_to_vpos(current_node, self.__tree))
            current_node = came_from.get(current_node)
        return path
