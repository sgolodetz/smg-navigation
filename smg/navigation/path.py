import numpy as np

from OpenGL.GL import *
from typing import Callable, Optional

from smg.opengl import OpenGLUtil


class Path:
    """A path."""

    # NESTED TYPES

    class Element:
        """A path element."""

        # CONSTRUCTOR

        def __init__(self, position: np.ndarray, is_essential: bool):
            self.__position: np.ndarray = position
            self.__is_essential: bool = is_essential

        # PROPERTIES

        @property
        def is_essential(self) -> bool:
            return self.__is_essential

        @property
        def position(self) -> np.ndarray:
            return self.__position

    # CONSTRUCTOR

    def __init__(self, positions: np.ndarray, essential_flags: np.ndarray):
        self.__positions: np.ndarray = positions
        self.__essential_flags: np.ndarray = essential_flags

    # SPECIAL METHODS

    def __getitem__(self, idx) -> Element:
        return Path.Element(self.__positions[idx, :], self.__essential_flags[idx].item())

    def __len__(self) -> int:
        return len(self.__positions)

    # PROPERTIES

    @property
    def essential_flags(self) -> np.ndarray:
        return self.__essential_flags

    @property
    def positions(self) -> np.ndarray:
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
