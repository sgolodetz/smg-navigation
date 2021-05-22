import numpy as np

from collections import defaultdict, deque
from typing import Callable, Deque, Dict, List, Optional, Tuple

from smg.utility import PriorityQueue

from .path import Path
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

    def plan_multi_step_path(self, waypoints: List[np.ndarray],
                             d: Optional[Callable[[np.ndarray, np.ndarray], float]] = None,
                             h: Optional[Callable[[np.ndarray, np.ndarray], float]] = None,
                             allow_shortcuts: bool = True, pull_strings: bool = True,
                             use_clearance: bool = True) -> Optional[Path]:
        """
        Try to plan a path that visits the specified list of waypoints.

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
        multipath_positions: List[np.ndarray] = []
        multipath_essential_flags: List[np.ndarray] = []

        # For each successive pair of waypoints:
        for i in range(len(waypoints) - 1):
            j: int = i + 1

            # Try to plan a path between them.
            path: Optional[Path] = self.plan_single_step_path(
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
                    multipath_positions.append(path.positions)
                    multipath_essential_flags.append(path.essential_flags)
                else:
                    multipath_positions.append(path.positions[:-1])
                    multipath_essential_flags.append(path.essential_flags[:-1])

        return Path(np.vstack(multipath_positions), np.vstack(multipath_essential_flags))

    def plan_single_step_path(self, source: np.ndarray, goal: np.ndarray, *,
                              d: Optional[Callable[[np.ndarray, np.ndarray], float]] = None,
                              h: Optional[Callable[[np.ndarray, np.ndarray], float]] = None,
                              allow_shortcuts: bool = True, pull_strings: bool = True,
                              use_clearance: bool = True) -> Optional[Path]:
        """
        Try to plan a path from the specified source to the specified goal.

        .. note::
            See PlanningToolkit.node_is_traversable for the definition of what "clearance" means in this case.
        .. note::
            The auxiliary data currently used consists of an array of flags, one for each path node. The flag
            for a node indicates whether or not it is an essential part of the path (True = yes, False = no).

        :param source:          The source (a 3D point in space).
        :param goal:            The goal (a 3D point in space).
        :param d:               An optional distance function (if None, L1 distance will be used).
        :param h:               An optional heuristic function (if None, L1 distance will be used).
        :param allow_shortcuts: Whether or not to allow shortcutting when the goal is in sight.
        :param pull_strings:    Whether to perform string pulling on the path prior to returning it.
        :param use_clearance:   Whether or not to plan a path that has sufficient "clearance" around it.
        :return:                A tuple consisting of the path and any auxiliary data about its nodes,
                                if a path was successfully found, or None otherwise.
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

        # Signal that the search has failed.
        return None

    def update_path(self, current_pos: np.ndarray, path: Path, *,
                    d: Optional[Callable[[np.ndarray, np.ndarray], float]] = None,
                    h: Optional[Callable[[np.ndarray, np.ndarray], float]] = None,
                    allow_shortcuts: bool, pull_strings: bool, use_clearance: bool) -> Optional[Path]:
        """
        Try to update the specified path based on the agent's current position.

        :param current_pos:     The current position of the agent.
        :param path:            The path to update.
        :param d:               An optional distance function (if None, L1 distance will be used).
        :param h:               An optional heuristic function (if None, L1 distance will be used).
        :param allow_shortcuts: Whether or not to allow shortcutting when the goal is in sight.
        :param pull_strings:    Whether to perform string pulling on the path prior to returning it.
        :param use_clearance:   Whether or not to take "clearance" around the path into account when updating it.
        :return:                The updated path, if successful, or None otherwise.
        """
        # Find the nearest waypoint of those on the path up to and including the next essential one.
        nearest_waypoint_idx: int = -1
        nearest_waypoint_dist: float = np.inf

        # For each waypoint beyond the agent's current position:
        for i in range(1, len(path)):
            # Find the distance between the agent and the waypoint.
            waypoint_dist: float = np.linalg.norm(path[i].position - current_pos)

            # If the waypoint is the best we've seen so far, record it.
            if waypoint_dist < nearest_waypoint_dist:
                nearest_waypoint_idx = i
                nearest_waypoint_dist = waypoint_dist

            # Stop if we reach an essential waypoint.
            if path[i].is_essential:
                break

        print(f"Distance to nearest waypoint: {nearest_waypoint_dist}")

        # TODO
        if nearest_waypoint_dist <= 0.2:
            # If we're within striking distance of the nearest waypoint, straighten the path up to and including
            # its successor.
            path = path.straighten_before(nearest_waypoint_idx + 1)
        else:
            # If there's a direct line of sight to the nearest waypoint, straight the path up to and including it.
            current_vpos: np.ndarray = self.__toolkit.pos_to_vpos(current_pos)
            nearest_waypoint_vpos: np.ndarray = self.__toolkit.pos_to_vpos(path[nearest_waypoint_idx].position)
            if self.__toolkit.line_segment_is_traversable(current_vpos, nearest_waypoint_vpos, use_clearance=True):
                path = path.straighten_before(nearest_waypoint_idx)

        if len(path) == 1:
            return None
        else:
            minipath: Optional[Path] = self.plan_single_step_path(
                current_pos, path[1].position, d=d, h=h,
                allow_shortcuts=allow_shortcuts,
                pull_strings=pull_strings,
                use_clearance=use_clearance
            )

            if minipath is not None:
                # TODO
                updated_path: Path = path.replace_before(1, minipath)

                # TODO
                if pull_strings:
                    return self.__toolkit.pull_strings(updated_path, use_clearance=use_clearance)
                else:
                    return updated_path
            else:
                return None

    # PRIVATE METHODS

    def __finalise_path(self, positions: Deque[np.ndarray], source: np.ndarray, goal: np.ndarray, *,
                        pull_strings: bool, use_clearance: bool) -> Path:
        """
        Finalise a reconstructed path by adding the source and goal to it, and optionally performing string pulling.

        :param positions:       The 3D positions of the nodes on the path.
        :param source:          The source (a 3D point in space).
        :param goal:            The goal (a 3D point in space).
        :param pull_strings:    Whether to perform string pulling on the path prior to returning it.
        :param use_clearance:   Whether or not to require sufficient "clearance" during string pulling.
        :return:                The finalised path.
        """
        # Respectively prepend and append the true source and goal to the path.
        positions.appendleft(source)
        positions.append(goal)

        # Make the "essential flags" for the path, namely a flag for each node indicating whether or not the node
        # is an essential part of the path (or can be smoothed away). For a simple path, the source and goal are
        # essential, but the waypoints in between aren't.
        essential_flags: np.ndarray = np.zeros((len(positions), 1), dtype=bool)
        essential_flags[0] = essential_flags[-1] = True

        # Construct and return the finalised path, performing string pulling in the process if requested.
        path: Path = Path(np.vstack(positions), essential_flags)
        if pull_strings:
            return self.__toolkit.pull_strings(path, use_clearance=use_clearance)
        else:
            return path

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
