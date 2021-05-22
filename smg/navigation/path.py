import numpy as np

from OpenGL.GL import *
from typing import Callable, Optional

from smg.opengl import OpenGLUtil


class Path:
    """A navigation path."""

    # NESTED TYPES

    class Waypoint:
        """A waypoint along a navigation path."""

        # CONSTRUCTOR

        def __init__(self, position: np.ndarray, is_essential: bool):
            """
            Construct a waypoint.

            :param position:        The 3D position of the waypoint.
            :param is_essential:    Whether or not the waypoint is essential to the path or can be smoothed away.
            """
            self.__position: np.ndarray = position
            self.__is_essential: bool = is_essential

        # PROPERTIES

        @property
        def is_essential(self) -> bool:
            """
            Get whether or not the waypoint is essential to the path or can be smoothed away.

            :return:    True, if it is essential, or False otherwise.
            """
            return self.__is_essential

        @property
        def position(self) -> np.ndarray:
            """
            Get the 3D position of the waypoint.

            :return:    The 3D position of the waypoint.
            """
            return self.__position

    # CONSTRUCTOR

    def __init__(self, positions: np.ndarray, essential_flags: np.ndarray):
        """
        Construct a navigation path.

        :param positions:       The 3D positions of the path's waypoints.
        :param essential_flags: An array of flags indicating which waypoints are essential to the path.
        """
        self.__positions: np.ndarray = positions
        self.__essential_flags: np.ndarray = essential_flags

    # SPECIAL METHODS

    def __getitem__(self, i) -> Waypoint:
        """
        Get the i'th waypoint of the path.

        :param i:   The index of the waypoint to get.
        :return:    The i'th waypoint of the path.
        """
        return Path.Waypoint(self.__positions[i], self.__essential_flags[i].item())

    def __len__(self) -> int:
        """
        Get the length of the path (i.e. the number of waypoints it contains).

        .. note::
            Just to be absolutely clear, this doesn't return the arc length of the path!

        :return:    The length of the path (i.e. the number of waypoints it contains).
        """
        return len(self.__positions)

    # PROPERTIES

    @property
    def essential_flags(self) -> np.ndarray:
        """
        Get an array of flags indicating which waypoints are essential to the path.

        :return:    An array of flags indicating which waypoints are essential to the path.
        """
        return self.__essential_flags

    @property
    def positions(self) -> np.ndarray:
        """
        Get the 3D positions of the path's waypoints.

        :return:    The 3D positions of the path's waypoints.
        """
        return self.__positions

    # PUBLIC METHODS

    def render(self, *, start_colour, end_colour, width: int = 1,
               waypoint_colourer: Optional[Callable[[np.ndarray], np.ndarray]] = None) -> None:
        """
        Render the path using OpenGL.

        .. note::
            The colour will be linearly interpolated between start_colour and end_colour as we move along the path.

        :param start_colour:        The colour to use for the start of the path.
        :param end_colour:          The colour to use for the end of the path.
        :param waypoint_colourer:   An optional function that can be used to determine the colours with which to
                                    render waypoints on the path (when None, the waypoints will not be rendered).
        :param width:               The width to use for the path.
        """
        if len(self) < 2:
            return

        start_colour: np.ndarray = np.array(start_colour)
        end_colour: np.ndarray = np.array(end_colour)

        glLineWidth(width)
        glBegin(GL_LINE_STRIP)
        for i in range(len(self)):
            t: float = i / (len(self) - 1)
            colour: np.ndarray = (1 - t) * start_colour + t * end_colour
            pos: np.ndarray = self.positions[i, :]

            glColor3f(*colour)
            glVertex3f(*pos)
        glEnd()
        glLineWidth(1)

        if waypoint_colourer is not None:
            for pos in self.positions:
                glColor3f(*waypoint_colourer(pos))
                OpenGLUtil.render_sphere(pos, 0.01, slices=10, stacks=10)
