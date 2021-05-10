import numpy as np

from collections import defaultdict, deque
from typing import Callable, Deque, Dict, List, Optional

from smg.utility import PriorityQueue

from .planning_toolkit import PathNode, PlanningToolkit


class AStarPathPlanner:
    """A path planner for Octomaps based on A*."""

    # CONSTRUCTOR

    def __init__(self, toolkit: PlanningToolkit):
        self.__toolkit: PlanningToolkit = toolkit

    # PUBLIC METHODS

    def plan_multipath(self, waypoints: List[np.ndarray],
                       d: Optional[Callable[[np.ndarray, np.ndarray], float]] = None,
                       h: Optional[Callable[[np.ndarray, np.ndarray], float]] = None,
                       pull_strings: bool = True, use_clearance: bool = False) -> Optional[np.ndarray]:
        multipath: List[np.ndarray] = []

        for i in range(len(waypoints) - 1):
            j: int = i + 1
            path: Optional[np.ndarray] = self.plan_path(
                waypoints[i], waypoints[j], d=d, h=h, pull_strings=pull_strings, use_clearance=use_clearance
            )
            if path is None:
                return None
            else:
                if j == len(waypoints) - 1:
                    multipath.append(path)
                else:
                    multipath.append(path[:-1])

        return np.vstack(multipath)

    def plan_path(self, source: np.ndarray, goal: np.ndarray, *,
                  d: Optional[Callable[[np.ndarray, np.ndarray], float]] = None,
                  h: Optional[Callable[[np.ndarray, np.ndarray], float]] = None,
                  pull_strings: bool = True, use_clearance: bool = False) -> Optional[np.ndarray]:
        # Based on an amalgam of:
        #
        # 1) https://en.wikipedia.org/wiki/A*_search_algorithm
        # 2) https://www.redblobgames.com/pathfinding/a-star/implementation.html

        if d is None:
            d = PlanningToolkit.l1_distance()
        if h is None:
            h = PlanningToolkit.l1_distance()

        source_node: PathNode = self.__toolkit.pos_to_node(source)
        goal_node: PathNode = self.__toolkit.pos_to_node(goal)

        source_vpos: np.ndarray = self.__toolkit.node_to_vpos(source_node)
        goal_vpos: np.ndarray = self.__toolkit.node_to_vpos(goal_node)

        source_occupancy: str = self.__toolkit.occupancy_status(source_node)
        goal_occupancy: str = self.__toolkit.occupancy_status(goal_node)
        print(source, source_node, source_occupancy)
        print(goal, goal_node, goal_occupancy)
        if not (self.__toolkit.node_is_traversible(source_node, use_clearance=use_clearance) and
                self.__toolkit.node_is_traversible(goal_node, use_clearance=use_clearance)):
            return None

        g_scores: Dict[PathNode, float] = defaultdict(lambda: np.infty)
        g_scores[source_node] = 0.0
        came_from: Dict[PathNode, Optional[PathNode]] = {source_node: None}

        frontier: PriorityQueue[PathNode, float, type(None)] = PriorityQueue[PathNode, float, type(None)]()
        frontier.insert(source_node, h(source_vpos, goal_vpos), None)

        while not frontier.empty():
            current_node: PathNode = frontier.top().ident
            current_vpos: np.ndarray = self.__toolkit.node_to_vpos(current_node)

            # if self.__toolkit.path_is_traversible(
            #     np.vstack([current_vpos, goal_vpos]), 0, 1, use_clearance=use_clearance
            # ):
            #     path: Deque[np.ndarray] = self.__reconstruct_path(current_node, came_from)
            #     path.appendleft(source)
            #     path.append(self.__toolkit.node_to_vpos(current_node))
            #     path.append(goal)
            #     return np.vstack(path)

            if current_node == goal_node:
                path: Deque[np.ndarray] = self.__reconstruct_path(goal_node, came_from)
                path.appendleft(source)
                path.append(goal)
                if pull_strings:
                    return self.__toolkit.pull_strings(np.vstack(path), use_clearance=use_clearance)
                else:
                    return np.vstack(path)

            frontier.pop()

            for neighbour_node in self.__toolkit.neighbours(current_node):
                if not self.__toolkit.node_is_traversible(neighbour_node, use_clearance=use_clearance):
                    continue

                neighbour_vpos: np.ndarray = self.__toolkit.node_to_vpos(neighbour_node)
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
            path.appendleft(self.__toolkit.node_to_vpos(current_node))
            current_node = came_from.get(current_node)
        return path
