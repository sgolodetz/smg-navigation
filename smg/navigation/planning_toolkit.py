import numpy as np

from smg.pyoctomap import OcTree, OcTreeNode, Vector3

from scipy.interpolate import Akima1DInterpolator
from typing import Callable, List, Optional, Tuple


# HELPER TYPES

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
            node_is_free = lambda n: self.occupancy_status(n) == "Free"

        self.neighbours: Callable[[PathNode], List[PathNode]] = neighbours
        self.node_is_free: Callable[[PathNode], bool] = node_is_free

    # PUBLIC STATIC METHODS

    @staticmethod
    def interpolate_path(path: np.ndarray, *, new_length: int = 100) -> np.ndarray:
        """
        Make a smoother version of a path by using curve fitting and interpolation.

        :param path:        The original path.
        :param new_length:  The number of points to take from the interpolating curve.
        :return:            The interpolated path.
        """
        x: np.ndarray = np.arange(len(path))
        cs: Akima1DInterpolator = Akima1DInterpolator(x, path)
        return cs(np.linspace(0, len(path) - 1, new_length))

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

    def chord_is_traversible(self, path: np.ndarray, source: int, dest: int, *, use_clearance: bool) -> bool:
        """
        Check whether the specified (straight line) chord between two points on a path can be directly traversed.

        :param path:            The path.
        :param source:          The index of the source point on the path.
        :param dest:            The index of the destination point on the path.
        :param use_clearance:   Whether to require there to be sufficient "clearance" around the chord.
        :return:                True, if the chord is traversible, or False otherwise.
        """
        # Look up the voxels containing the source and destination points.
        source_node: PathNode = self.pos_to_node(path[source, :])
        dest_node: PathNode = self.pos_to_node(path[dest, :])

        source_vpos: np.ndarray = self.node_to_vpos(source_node)
        dest_vpos: np.ndarray = self.node_to_vpos(dest_node)

        # Test voxels along the chord for their traversibility. If any of them is non-traversible, so is the chord.
        # TODO: Fix and optimise this. It can fail if the chord's very long (through not testing enough points),
        #       and it's needlessly slow. A midpoint line algorithm should be used instead.
        prev_node: Optional[PathNode] = None
        for t in np.linspace(0.0, 1.0, 101):
            pos: np.ndarray = source_vpos * (1 - t) + dest_vpos * t
            node: PathNode = self.pos_to_node(pos)
            if prev_node is None or node != prev_node:
                prev_node = node
                if not self.node_is_traversible(node, use_clearance=use_clearance):
                    return False

        return True

    def node_is_traversible(self, node: PathNode, *, use_clearance: bool) -> bool:
        """
        Check whether the specified path node can be traversed.

        .. note::
            The definition of a node's traversibility depends on whether we're requiring there to be sufficient
            "clearance" around the node or not. If not, a node is traversible if it's "free", which is defined
            in terms of its occupancy status. If so, a node is traversible if not only is it "free", but so are
            its immediate neighbours.

        :param node:            The node whose traversibility we want to check.
        :param use_clearance:   Whether to require there to be sufficient "clearance" around the node.
        :return:                True, if the node is traversible, or False otherwise.
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

    def occupancy_status(self, node: PathNode) -> str:
        """
        Get the occupancy status of the Octomap voxel corresponding to the specified path node.

        :param node:    A path node.
        :return:        The occupancy status of the Octomap voxel corresponding to the path node.
        """
        # FIXME: Use an enumeration for the return values.
        vpos: np.ndarray = self.node_to_vpos(node)
        octree_node: Optional[OcTreeNode] = self.__tree.search(Vector3(*vpos))
        if octree_node is None:
            return "Unknown"
        else:
            occupied: bool = self.__tree.is_node_occupied(octree_node)
            return "Occupied" if occupied else "Free"

    def pos_to_node(self, pos: np.ndarray) -> PathNode:
        """
        Compute the path node corresponding to the voxel that contains the specified position.

        :param pos: The position.
        :return:    The path node corresponding to the voxel that contains the position.
        """
        voxel_size: float = self.__tree.get_resolution()
        return tuple(np.round(pos // voxel_size).astype(int))

    def pull_strings(self, path: np.ndarray, *, use_clearance: bool) -> np.ndarray:
        """
        TODO

        :param path:            TODO
        :param use_clearance:   TODO
        :return:                TODO
        """
        pulled_path: List[np.ndarray] = []

        i: int = 0
        while i < len(path):
            pulled_path.append(path[i, :])

            j: int = i + 2
            while j < len(path) and self.chord_is_traversible(path, i, j, use_clearance=use_clearance):
                j += 1

            i = j - 1

        return np.vstack(pulled_path)
