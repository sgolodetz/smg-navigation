from __future__ import annotations

import numpy as np

from OpenGL.GL import *
from scipy.interpolate import CubicHermiteSpline
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

    def arc_length(self) -> float:
        """
        Calculate the arc length of the path (in m).

        :return:    The arc length of the path (in m).
        """
        result: float = 0.0
        for i in range(len(self.__positions) - 1):
            result += np.linalg.norm(self.__positions[i+1] - self.__positions[i])

        return result

    def copy(self) -> Path:
        """
        Make a copy of the path.

        :return:    A copy of the path.
        """
        return Path(self.__positions.copy(), self.__essential_flags.copy())

    def interpolate(self, *, min_gap_mean: float = 0.05, new_length: int = 100) -> Path:
        """
        Make a smoother version of the path by using spline fitting and interpolation.

        .. note::
            The specified number of points to take will be reduced if necessary to try to achieve the specified
            minimum average gap.

        :param min_gap_mean:    The minimum average gap wanted between adjacent points on the interpolated path.
        :param new_length:      The initial number of points to take from the interpolating spline.
        :return:                The interpolated path.
        """
        # Fit a Hermite spline to the current path, setting the gradients needed to something sensible.
        # noinspection PyShadowingNames
        x: np.ndarray = np.arange(len(self))
        gradients: np.ndarray = np.zeros_like(self.__positions)
        for i in range(1, len(self.__positions) - 1):
            gradients[i] = (self.__positions[i+1] - self.__positions[i-1]) / 2

        cs: CubicHermiteSpline = CubicHermiteSpline(x, self.__positions, gradients)

        # Until we get an interpolated path that we're happy with:
        while True:
            # Interpolate the path by sampling a fixed number of points along the spline.
            interpolated_path: Path = self.__make_interpolated_path(cs, new_length)

            # Calculate the average gap between adjacent points on the interpolated path.
            gap_sum: float = 0.0
            for i in range(0, new_length - 1):
                gap_sum += np.linalg.norm(interpolated_path.positions[i+1] - interpolated_path.positions[i])

            gap_mean: float = gap_sum / len(interpolated_path)

            # If either (i) the average gap is greater than the minimum we specified, or (ii) the number of points
            # to take from the interpolating spline would be too small if we reduced it any further, accept the
            # interpolated path and exit. (The "4" was chosen somewhat arbitrarily here, but seems to work ok.)
            if gap_mean >= min_gap_mean or new_length <= 4:
                break

            # Otherwise, halve the number of points to take from the interpolating spline and try again.
            else:
                new_length //= 2

        return interpolated_path

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
            pos: np.ndarray = self.positions[i]

            glColor3f(*colour)
            glVertex3f(*pos)
        glEnd()
        glLineWidth(1)

        if waypoint_colourer is not None:
            for pos in self.positions:
                glColor3f(*waypoint_colourer(pos))
                OpenGLUtil.render_sphere(pos, 0.01, slices=10, stacks=10)

    def replace_before(self, waypoint_idx: int, new_subpath: Path, *, keep_last: bool) -> Path:
        """
        Make a copy of the path in which the sub-path prior to the specified waypoint has been replaced
        with a new sub-path.

        :param waypoint_idx:    The index of the waypoint at the start of the remainder of the path.
        :param new_subpath:     The new sub-path to substitute for the sub-path before the specified waypoint.
        :param keep_last:       Whether or not to include the last waypoint of the new sub-path.
        :return:                A copy of the path in which the sub-path prior to the specified waypoint has
                                been replaced with the new sub-path.
        """
        if keep_last:
            return Path(
                np.vstack([new_subpath.positions, self.__positions[waypoint_idx:]]),
                np.vstack([new_subpath.essential_flags, self.__essential_flags[waypoint_idx:]])
            )
        else:
            return Path(
                np.vstack([new_subpath.positions[:-1], self.__positions[waypoint_idx:]]),
                np.vstack([new_subpath.essential_flags[:-1], self.__essential_flags[waypoint_idx:]])
            )

    def straighten_before(self, waypoint_idx: int) -> Path:
        """
        Make a copy of the path in which the source is the only waypoint retained prior to the specified waypoint.

        :param waypoint_idx:    The index of the waypoint at the start of the remainder of the path.
        :return:                A copy of the path in which the source is the only waypoint retained prior to
                                the specified waypoint.
        """
        return Path(
            np.vstack([self.__positions[0], self.__positions[waypoint_idx:]]),
            np.vstack([self.__essential_flags[0], self.__essential_flags[waypoint_idx:]])
        )

    # PRIVATE METHODS

    def __make_interpolated_path(self, cs: CubicHermiteSpline, new_length: int) -> Path:
        """
        Make an interpolated version of the current path by sampling the specified number of points from a spline
        that has been fitted to it.

        :param cs:          The spline.
        :param new_length:  The number of points to sample.
        :return:            The interpolated version of the current path.
        """
        essential_flags: np.ndarray = np.full((new_length, 1), True, dtype=bool)
        return Path(cs(np.linspace(0, len(self) - 1, new_length)), essential_flags)
