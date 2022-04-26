import numpy as np

from typing import Callable, List, Optional, Tuple

from smg.pyoctomap import OcTree, OcTreeNode, Vector3

from .path import Path


# HELPER ENUMERATIONS

class EOccupancyStatus(int):
    """The occupancy status of an Octomap voxel."""

    # SPECIAL METHODS

    def __str__(self) -> str:
        """
        Get a string representation of the occupancy status of an Octomap voxel.

        :return:    A string representation of the occupancy status of an Octomap voxel.
        """
        if self == OCS_FREE:
            return "OCS_FREE"
        elif self == OCS_OCCUPIED:
            return "OCS_OCCUPIED"
        else:
            return "OCS_UNKNOWN"


# The voxel is known to be unoccupied.
OCS_FREE = EOccupancyStatus(0)

# The voxel is known to be occupied.
OCS_OCCUPIED = EOccupancyStatus(1)

# The occupancy status of the voxel is unknown.
OCS_UNKNOWN = EOccupancyStatus(2)


# HELPER TYPES

# A path planning node, corresponding to a voxel in an Octomap.
PathNode = Tuple[int, int, int]


# MAIN CLASS

class PlanningToolkit:
    """A toolkit of useful functions for use by path planners that work with Octomap scenes."""

    # CONSTRUCTOR

    def __init__(self, tree: OcTree, *, neighbours: Optional[Callable[[PathNode], List[PathNode]]] = None,
                 node_is_free: Optional[Callable[[PathNode], bool]] = None):
        """
        Construct a planning toolkit.

        .. note::
            If the neighbours function isn't explicitly specified, 6-connected neighbours will be computed.

        :param tree:            The Octomap octree over which paths are to be planned.
        :param neighbours:      An optional function specifying how the neighbours of a path node are to be computed.
        :param node_is_free:    An optional function specifying what counts as a "free" path node.
        """
        self.__tree: OcTree = tree

        if neighbours is None:
            neighbours = PlanningToolkit.neighbours6
        if node_is_free is None:
            node_is_free = lambda n: self.occupancy_status(n) == OCS_FREE

        self.neighbours: Callable[[PathNode], List[PathNode]] = neighbours
        self.node_is_free: Callable[[PathNode], bool] = node_is_free

    # PUBLIC STATIC METHODS

    @staticmethod
    def l1_distance(*, ax: float = 1.0, ay: float = 1.0, az: float = 1.0) -> Callable[[np.ndarray, np.ndarray], float]:
        """
        Construct a function that computes a scaled L1 distance between two vectors.

        .. note::
            Specifically, the constructed function will compute ax * |x2-x1| + ay * |y2-y1| + az * |z2-z1|.

        :param ax:  The x scaling factor.
        :param ay:  The y scaling factor.
        :param az:  The z scaling factor.
        :return:    A function that will compute a scaled L1 distance between two vectors.
        """
        def inner(v1: np.ndarray, v2: np.ndarray) -> float:
            """
            Compute a scaled L1 distance between two vectors.

            .. note::
                The scaling factors are baked into the function when it is constructed.

            :param v1:  The first vector.
            :param v2:  The second vector.
            :return:    The scaled L1 distance between the vectors.
            """
            dx: float = abs(v2[0] - v1[0])
            dy: float = abs(v2[1] - v1[1])
            dz: float = abs(v2[2] - v1[2])
            return ax * dx + ay * dy + az * dz

        return inner

    @staticmethod
    def l2_distance(v1: np.ndarray, v2: np.ndarray) -> float:
        """
        Compute the L2 distance between two vectors.

        :param v1:  The first vector.
        :param v2:  The second vector.
        :return:    The L2 distance between the vectors.
        """
        return np.linalg.norm(v1 - v2)

    @staticmethod
    def neighbours4(node: PathNode) -> List[PathNode]:
        """
        Get the 4-connected neighbours of a path node.

        :param node:    The node.
        :return:        The 4-connected neighbours of the node.
        """
        x, y, z = node
        return [
            (x, y, z - 1),
            (x - 1, y, z),
            (x + 1, y, z),
            (x, y, z + 1)
        ]

    @staticmethod
    def neighbours6(node: PathNode) -> List[PathNode]:
        """
        Get the 6-connected neighbours of a path node.

        :param node:    The node.
        :return:        The 6-connected neighbours of the node.
        """
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
        """
        Get the 8-connected neighbours of a path node.

        :param node:    The node.
        :return:        The 8-connected neighbours of the node.
        """
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
        """
        Get the 26-connected neighbours of a path node.

        :param node:    The node.
        :return:        The 26-connected neighbours of the node.
        """
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

    # PUBLIC METHODS

    def chord_is_traversable(self, path: Path, source: int, dest: int, *, use_clearance: bool) -> bool:
        """
        Check whether the specified (straight line) chord between two points on a path can be directly traversed.

        :param path:            The path.
        :param source:          The index of the source point on the path.
        :param dest:            The index of the destination point on the path.
        :param use_clearance:   Whether to require there to be sufficient "clearance" around the chord.
        :return:                True, if the chord is traversable, or False otherwise.
        """
        source_node: PathNode = self.pos_to_node(path[source].position)
        source_vpos: np.ndarray = self.node_to_vpos(source_node)
        dest_node: PathNode = self.pos_to_node(path[dest].position)
        dest_vpos: np.ndarray = self.node_to_vpos(dest_node)
        return self.line_segment_is_traversable(source_vpos, dest_vpos, use_clearance=use_clearance)

    def get_tree(self) -> OcTree:
        """
        Get the Octomap octree associated with the toolkit.

        :return:    The Octomap octree associated with the toolkit.
        """
        return self.__tree

    def line_segment_is_traversable(self, source_vpos: np.ndarray, dest_vpos: np.ndarray, *,
                                    use_clearance: bool) -> bool:
        """
        Check whether the specified line segment between two voxel centres can be directly traversed.

        :param source_vpos:     The centre of the source voxel.
        :param dest_vpos:       The centre of the destination voxel.
        :param use_clearance:   Whether to require there to be sufficient "clearance" around the line segment.
        :return:                True, if the line segment is traversable, or False otherwise.
        """
        # Test voxels along the segment for their traversability. If any of them is non-traversable, so is the segment.
        # TODO: Fix and optimise this. It can fail if the segment's very long (through not testing enough points),
        #       and it's needlessly slow. A midpoint line algorithm should be used instead.
        prev_node: Optional[PathNode] = None
        for t in np.linspace(0.0, 1.0, 101):
            pos: np.ndarray = source_vpos * (1 - t) + dest_vpos * t
            node: PathNode = self.pos_to_node(pos)
            if prev_node is None or node != prev_node:
                prev_node = node
                if not self.node_is_traversable(node, use_clearance=use_clearance):
                    return False

        return True

    def node_is_traversable(self, node: PathNode, *, use_clearance: bool) -> bool:
        """
        Check whether the specified path node can be traversed.

        .. note::
            The definition of a node's traversability depends on whether we're requiring there to be sufficient
            "clearance" around the node or not. If not, a node is traversable if it's "free", which is defined
            in terms of its occupancy status. If so, a node is traversable if not only is it "free", but so are
            its immediate neighbours.

        :param node:            The node whose traversability we want to check.
        :param use_clearance:   Whether to require there to be sufficient "clearance" around the node.
        :return:                True, if the node is traversable, or False otherwise.
        """
        if not self.node_is_free(node):
            return False

        if use_clearance:
            for neighbour_node in self.neighbours(node):
                if not self.node_is_free(neighbour_node):
                    return False

        return True

    def node_to_vpos(self, node: PathNode) -> np.ndarray:
        """
        Compute the position of the centre of the voxel corresponding to the specified path node.

        :param node:    The path node.
        :return:        The position of the centre of the corresponding voxel.
        """
        voxel_size: float = self.__tree.get_resolution()
        half_voxel_size: float = voxel_size / 2.0
        return np.array([node[i] * voxel_size + half_voxel_size for i in range(3)])

    def occupancy_colourer(self) -> Callable[[np.ndarray], np.ndarray]:
        """
        Construct a function that can be used to colour waypoints on a path based on their occupancy status.

        .. note::
            Specifically, waypoints in free space will be coloured green, and all the others will be coloured red.

        :return:    A function that can be used to colour waypoints on a path based on their occupancy status.
        """
        def inner(pos: np.ndarray) -> np.ndarray:
            """
            A function that can be used to colour waypoints on a path based on their occupancy status.

            :param pos: The position of a waypoint.
            :return:    The colour to assign to the waypoint.
            """
            occupancy: EOccupancyStatus = self.occupancy_status(self.pos_to_node(pos))
            if occupancy == OCS_FREE:
                return np.array([0, 1, 0])
            else:  # in practice, occupancy == OCS_UNKNOWN
                return np.array([1, 0, 0])

        return inner

    def occupancy_status(self, node: PathNode) -> EOccupancyStatus:
        """
        Get the occupancy status of the Octomap voxel corresponding to the specified path node.

        :param node:    A path node.
        :return:        The occupancy status of the Octomap voxel corresponding to the path node.
        """
        vpos: np.ndarray = self.node_to_vpos(node)
        octree_node: Optional[OcTreeNode] = self.__tree.search(Vector3(*vpos))
        if octree_node is None:
            return OCS_UNKNOWN
        else:
            occupied: bool = self.__tree.is_node_occupied(octree_node)
            return OCS_OCCUPIED if occupied else OCS_FREE

    def point_is_in_bounds(self, pos: np.ndarray) -> bool:
        """
        Check whether the specified position is within the bounds of the Octomap octree.

        :param pos: The position.
        :return:    True, if the position is within the Octomap octree bounds, or False otherwise.
        """
        return self.__tree.is_point_in_bounds(Vector3(*pos))

    def pos_to_node(self, pos: np.ndarray) -> PathNode:
        """
        Compute the path node corresponding to the voxel that contains the specified position.

        :param pos: The position.
        :return:    The path node corresponding to the voxel that contains the position.
        """
        voxel_size: float = self.__tree.get_resolution()
        return tuple(np.round(pos // voxel_size).astype(int))

    def pos_to_vpos(self, pos: np.ndarray) -> np.ndarray:
        """
        Compute the position of the centre of the voxel that contains the specified position.

        :param pos: The position.
        :return:    The position of the centre of the voxel that contains it.
        """
        return self.node_to_vpos(self.pos_to_node(pos))

    def pull_strings(self, path: Path, *, use_clearance: bool) -> Path:
        """
        Perform "string pulling" on the specified path.

        :param path:            The path on which to perform string pulling.
        :param use_clearance:   Whether to take "clearance" into account during the string pulling.
        :return:                The result of performing string pulling on the path.
        """
        pulled_positions: List[np.ndarray] = []
        pulled_essential_flags: List[bool] = []

        # Start at the beginning of the input path.
        i: int = 0

        # For each segment start point:
        while i < len(path):
            # Add the segment start point to the output path.
            pulled_positions.append(path[i].position)
            pulled_essential_flags.append(path[i].is_essential)

            # If the next point along the path is essential, use it as the start of the next segment.
            if i + 1 < len(path) and path[i + 1].is_essential:
                i = i + 1
                continue

            # Find the furthest point along the path to which we can directly traverse from the segment start point.
            j: int = i + 2
            while j < len(path) and self.chord_is_traversable(path, i, j, use_clearance=use_clearance):
                j += 1

                # If we encounter an essential node, that's as far as we can go for this segment.
                if path[j - 1].is_essential:
                    break

            # Use the furthest point to which we were able to directly traverse as the start of the next segment.
            i = j - 1

        return Path(np.vstack(pulled_positions), np.vstack(pulled_essential_flags))
