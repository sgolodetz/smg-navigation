import numpy as np

from collections import defaultdict, deque
from typing import Callable, Deque, Dict, List, Optional

from smg.utility import PriorityQueue

from .planning_toolkit import EOccupancyStatus, PathNode, PlanningToolkit


class AStarPathPlanner:
    """A path planner for Octomaps based on A*."""

    # CONSTRUCTOR

    def __init__(self, toolkit: PlanningToolkit, *, debug: bool = False):
        """
        Construct a path planner for Octomaps based on A*.

        :param toolkit: A planning toolkit that provides useful functions for path planners.
        :param debug:   Whether to print out debugging messages.
        """
        self.__debug: bool = debug
        self.__toolkit: PlanningToolkit = toolkit

    # PUBLIC METHODS

    def plan_multipath(self, waypoints: List[np.ndarray],
                       d: Optional[Callable[[np.ndarray, np.ndarray], float]] = None,
                       h: Optional[Callable[[np.ndarray, np.ndarray], float]] = None,
                       allow_shortcuts: bool = True, pull_strings: bool = True, use_clearance: bool = True) \
            -> Optional[np.ndarray]:
        """
        Try to plan a path that visits the specified set of waypoints.

        .. note::
            See PlanningToolkit.node_is_traversable for the definition of what "clearance" means in this case.

        :param waypoints:       The waypoints to visit.
        :param d:               An optional distance function (if None, L1 distance will be used).
        :param h:               An optional heuristic function (if None, L1 distance will be used).
        :param allow_shortcuts: Whether or not to allow shortcutting when the goal is in sight.
        :param pull_strings:    Whether to perform string pulling on the path prior to returning it.
        :param use_clearance:   Whether or not to plan a path that has sufficient "clearance" around it.
        :return:                The path, if one was successfully found, or None otherwise.
        """
        multipath: List[np.ndarray] = []

        # For each successive pair of waypoints:
        for i in range(len(waypoints) - 1):
            j: int = i + 1

            # Try to plan an individual path between them.
            path: Optional[np.ndarray] = self.plan_path(
                waypoints[i], waypoints[j], d=d, h=h,
                allow_shortcuts=allow_shortcuts,
                pull_strings=pull_strings,
                use_clearance=use_clearance
            )

            # If no path can be found between this pair of waypoints, early out. Otherwise, add this path to the
            # overall multi-path. Note that we remove the final point (the target waypoint) for all but the last
            # constituent path to avoid the multi-path containing duplicate points.
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
                  allow_shortcuts: bool = True, pull_strings: bool = True, use_clearance: bool = True) \
            -> Optional[np.ndarray]:
        """
        Try to plan a path from the specified source to the specified goal.

        .. note::
            See PlanningToolkit.node_is_traversable for the definition of what "clearance" means in this case.

        :param source:          The source (a 3D point in space).
        :param goal:            The goal (a 3D point in space).
        :param d:               An optional distance function (if None, L1 distance will be used).
        :param h:               An optional heuristic function (if None, L1 distance will be used).
        :param allow_shortcuts: Whether or not to allow shortcutting when the goal is in sight.
        :param pull_strings:    Whether to perform string pulling on the path prior to returning it.
        :param use_clearance:   Whether or not to plan a path that has sufficient "clearance" around it.
        :return:                The path, if one was successfully found, or None otherwise.
        """
        # Loosely based on an amalgam of:
        #
        # 1) https://en.wikipedia.org/wiki/A*_search_algorithm
        # 2) https://www.redblobgames.com/pathfinding/a-star/implementation.html

        # Use the default distance and heuristic functions if custom ones weren't specified.
        if d is None:
            d = PlanningToolkit.l1_distance()
        if h is None:
            h = PlanningToolkit.l1_distance()

        # Determine the path nodes corresponding to the source and goal.
        source_node: PathNode = self.__toolkit.pos_to_node(source)
        goal_node: PathNode = self.__toolkit.pos_to_node(goal)

        # Determine the centres of the voxels corresponding to the source and goal.
        source_vpos: np.ndarray = self.__toolkit.node_to_vpos(source_node)
        goal_vpos: np.ndarray = self.__toolkit.node_to_vpos(goal_node)

        # If requested, print out some debugging information about the source and goal.
        if self.__debug:
            source_occupancy: EOccupancyStatus = self.__toolkit.occupancy_status(source_node)
            goal_occupancy: EOccupancyStatus = self.__toolkit.occupancy_status(goal_node)
            print(f"Source: {source}, {source_node}, {source_occupancy}")
            print(f"Goal: {goal}, {goal_node}, {goal_occupancy}")

        # Check that the source and goal are traversable to avoid costly searching for a path that can't exist.
        # Early out if either is not traversable.
        if not self.__toolkit.node_is_traversable(source_node, use_clearance=use_clearance):
            if self.__debug:
                print("Source is not traversable")
            return None

        if not self.__toolkit.node_is_traversable(goal_node, use_clearance=use_clearance):
            if self.__debug:
                print("Goal is not traversable")
            return None

        # Search for a path using A*.
        g_scores: Dict[PathNode, float] = defaultdict(lambda: np.infty)
        g_scores[source_node] = 0.0
        came_from: Dict[PathNode, Optional[PathNode]] = {source_node: None}

        frontier: PriorityQueue[PathNode, float, type(None)] = PriorityQueue[PathNode, float, type(None)]()
        frontier.insert(source_node, h(source_vpos, goal_vpos), None)

        # While the search still has a chance of succeeding:
        while not frontier.empty():
            # Get the current node to explore from the frontier.
            current_node: PathNode = frontier.top().ident
            current_vpos: np.ndarray = self.__toolkit.node_to_vpos(current_node)

            # If we've reached the goal:
            if current_node == goal_node:
                # Construct and return the path.
                path: Deque[np.ndarray] = self.__reconstruct_path(goal_node, came_from)
                return self.__finalise_path(path, source, goal, pull_strings=pull_strings, use_clearance=use_clearance)

            # Otherwise, if we're allowing shortcuts and the goal's in sight:
            elif allow_shortcuts and self.__toolkit.line_segment_is_traversable(
                current_vpos, goal_vpos, use_clearance=use_clearance
            ):
                # Cut the path planning short, and construct and return a path that heads directly for it.
                path: Deque[np.ndarray] = self.__reconstruct_path(current_node, came_from)
                path.append(goal_vpos)
                return self.__finalise_path(path, source, goal, pull_strings=pull_strings, use_clearance=use_clearance)

            # Remove the current node from the frontier before proceeding.
            frontier.pop()

            # For each traversable neighbour of the current node:
            for neighbour_node in self.__toolkit.neighbours(current_node):
                if not self.__toolkit.node_is_traversable(neighbour_node, use_clearance=use_clearance):
                    continue

                # Compute the tentative cost to the neighbour via the current node.
                neighbour_vpos: np.ndarray = self.__toolkit.node_to_vpos(neighbour_node)
                tentative_cost: float = g_scores[current_node] + d(current_vpos, neighbour_vpos)

                # If it's less than any existing cost to the neighbour that we know about:
                if tentative_cost < g_scores[neighbour_node]:
                    # Update the best known path to the neighbour, and make sure the neighbour is on the frontier.
                    g_scores[neighbour_node] = tentative_cost
                    came_from[neighbour_node] = current_node
                    f_score: float = tentative_cost + h(neighbour_vpos, goal_vpos)
                    if frontier.contains(neighbour_node):
                        frontier.update_key(neighbour_node, f_score)
                    else:
                        frontier.insert(neighbour_node, f_score, None)

        # If the search has failed, return None.
        return None

    def update_path(self, current_pos: np.ndarray, path: np.ndarray) -> Optional[np.ndarray]:
        # TODO
        next_waypoint_pos: np.ndarray = path[1, :]
        next_waypoint_vpos: np.ndarray = self.__toolkit.node_to_vpos(self.__toolkit.pos_to_node(next_waypoint_pos))
        distance: float = np.linalg.norm(next_waypoint_pos - current_pos)
        print(f"Distance to next waypoint: {distance}")
        current_vpos: np.ndarray = self.__toolkit.node_to_vpos(self.__toolkit.pos_to_node(current_pos))
        if distance <= 0.2 and self.__toolkit.line_segment_is_traversable(current_vpos, next_waypoint_vpos, use_clearance=True):
            path = np.vstack([path[0, :], path[2:]])

        if len(path) == 1:
            return None
        else:
            ay: float = 10
            mini_path: Optional[np.ndarray] = self.plan_path(
                current_pos,
                path[1, :],
                d=PlanningToolkit.l1_distance(ay=ay),
                h=PlanningToolkit.l1_distance(ay=ay),
                allow_shortcuts=True,
                pull_strings=True,
                use_clearance=True
            )

            if mini_path is not None:
                return np.vstack([mini_path[:-1], path[1:, :]])
            else:
                return None

    # PRIVATE METHODS

    def __finalise_path(self, path: Deque[np.ndarray], source: np.ndarray, goal: np.ndarray, *,
                        pull_strings: bool, use_clearance: bool) -> np.ndarray:
        """
        Finalise a reconstructed path by adding the source and goal to it, and optionally performing string pulling.

        :param path:            The path to finalise.
        :param source:          The source (a 3D point in space).
        :param goal:            The goal (a 3D point in space).
        :param pull_strings:    Whether to perform string pulling on the path prior to returning it.
        :param use_clearance:   Whether or not to require sufficient "clearance" during string pulling.
        :return:                The finalised path.
        """
        # Respectively prepend and append the true source and goal to the path.
        path.appendleft(source)
        path.append(goal)

        # Return the path, performing string pulling in the process if requested.
        if pull_strings:
            return self.__toolkit.pull_strings(np.vstack(path), use_clearance=use_clearance)
        else:
            return np.vstack(path)

    def __reconstruct_path(self, goal_node: PathNode, came_from: Dict[PathNode, Optional[PathNode]]) \
            -> Deque[np.ndarray]:
        """
        Reconstruct a path from the centre of the source voxel to the centre of the goal voxel by following the
        'came from' links backwards from the goal node.

        .. note::
            The actual source and goal may not have been at the centres of their respective voxels, but they
            will be respectively prepended and appended to the reconstructed path later.

        :param goal_node:   The goal node.
        :param came_from:   A table specifying what the preceding node on the path would have been for each node.
        :return:            The reconstructed path, as a sequence of 3D points in space.
        """
        path: Deque[np.ndarray] = deque()
        current_node: Optional[PathNode] = goal_node
        while current_node is not None:
            path.appendleft(self.__toolkit.node_to_vpos(current_node))
            current_node = came_from.get(current_node)
        return path
