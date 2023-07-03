"Work with manifolds."

# This file is currently in a broken state

import numpy as np
from numpy.typing import NDArray

from typing import Tuple, Callable, List, Union, Optional, Any

Int = np.integer[Any]
Bool = np.bool_

Connector = Callable[
    [NDArray[Int], NDArray[Int], NDArray[Int]],
    Tuple[NDArray[Bool], NDArray[Int], NDArray[Int], NDArray[Int]]
]
Limits = NDArray[Int]


# TODO account for LHS/RHS heterogeneous atlases
# TODO bilinear interp impl
# TODO bicubic interp triangular algo
# TODO line singularities
# TODO edges
# TODO multiply-charted regions


def healpix_atlas(nside):
    import numpy as np

    lim = np.arange(2)[:, None].repeat(nside, 1) * (nside - 1)
    return Atlas(
        healpix_base_atlas(),
        lim,
        lim,
    )


class Atlas:
    current_faces: NDArray[Int]
    faces: NDArray[Int]
    current_sides: NDArray[Int]
    sides: NDArray[Int]
    corners: List[NDArray]
    x_limits: List[Limits]
    y_limits: List[Limits]
    connectors: List[Tuple[Connector, Connector, Connector, Connector]]

    def __init__(self, faces, x_limits, y_limits, connectors):
        import numpy as np

        self.faces = faces
        self.sides = next_side(faces)
        self.current_faces, self.current_sides = np \
            .meshgrid(*map(np.arange, faces.shape), indexing='ij')
        self.corners = atlas_dual(faces)
        self.x_limits = x_limits
        self.y_limits = y_limits
        self.connectors = connectors

    def ortho_translate(
            self,
            x: NDArray[Int],
            y: NDArray[Int],
            chart: NDArray[Int],
            direction: Union[NDArray[Int], int],
            distance: Union[NDArray[Int], int],
            in_place: Optional[bool] = False,
    ) -> Tuple[NDArray[Bool], NDArray[Int], NDArray[Int], NDArray[Int]]:
        """
        Translate coordinates in a given direction within this `Atlas`. All
        inputs are expected to have the same shape. This translation can
        only be directly along coordinate axes (since these are well-defined
        for connected sides); use `translate` for diagonal movements.

        .. code-block:: text

                   1 ◄── DIRECTION
                   ▲
             y     │
             ▲ ┌───┬───┐
             │ │0,Y│X,Y│
           0◄──├───┼───┤──►2
             │ │0,0│X,0│
             │ └───┴───┘
             0─────│───►x ◄── COORD AXIS
                   ▼
                   3

        Parameters
        ----------
        x : NDArray[Int]
            x-coordinate of each point in its chart.
        y : NDArray[Int]
            y-coordinate of each point in its chart.
        chart : NDArray[Int]
            Chart number of each point.
        direction : NDArray[Int]
            Direction in which to move from each starting point (see diagram
            above).
        distance : NDArray[Int]
            Number of steps in coordinate grid by which to move.
        in_place : bool, optional
            If `true`, write results to input buffers (default: false).

        Returns
        -------
        mask : NDArray[Bool]
            A boolean mask with the same shape as the inputs indicating which
            coordinates were successfully translated.
        x_out, y_out, chart_out : NDArray[Int]
            The translated coordinates (flattened). Use `mask` as an index to
            recover original shape.

        See Also
        --------
        Atlas.translate :
            Perform diagonal translations and extract distance information
            on results.
        """
        import numpy as np

        dx = np.array([-1, 0, 1, 0])
        dy = np.array([0, 1, 0, -1])
        x_out = x if in_place else x.copy()
        y_out = y if in_place else y.copy()
        chart_out = chart if in_place else chart.copy()
        x_out += dx[direction] * distance
        y_out += dy[direction] * distance
        mask = self.valid_idx(x_out, y_out, chart)
        invalid = ~mask
        mask[invalid], x_out[invalid], y_out[invalid], chart_out[invalid] = \
            dispatch(
                chart_out[invalid],
                lambda c: dispatch(
                    lambda dirs, *a: dispatch(
                        dirs,
                        lambda d: self.connectors[c][d],
                        *a,
                    )
                ),
                _empty_translate,
                direction if np.isscalar(direction) else direction[invalid],
                x_out[invalid],
                y_out[invalid],
                chart_out[invalid],
            )
        return mask, x_out, y_out, chart_out

    def valid_idx(
            self,
            x: NDArray[Int],
            y: NDArray[Int],
            chart: Union[NDArray[Int], int],
    ) -> NDArray[Bool]:
        """
        Check whether each index is valid.

        Parameters
        ----------
        x, y : NDArray[Int]
            x, y coordinates of each point.
        chart : int or NDArray[Int]
            The chart to check (if scalar) or the chart corresponding to each
            x, y point to check.

        Returns
        -------
        mask : NDArray[Bool]
            Whether each point is contained within its chart.
        """
        import numpy as np

        return dispatch(
            chart,
            lambda s: lambda *a: idx_in_chart(*a, self.x_limits[s],
                                              self.y_limits[s]),
            lambda: np.full(chart.shape, False, dtype=np.bool),
            x,
            y,
        )
        # if np.isscalar(chart):
        #    return idx_in_chart(x, y, self.x_limits[chart],
        #                        self.y_limits[chart])
        # if len(chart) == 0:
        #    return np.full(chart.shape, False, dtype=np.bool)
        ## make the check logarithmic in chart count
        # charts, inv, counts = np.unique(chart, return_inverse=True,
        #                                return_counts=True)
        # if len(charts) == 1:
        #    return self.valid_idx(x, y, charts[0])
        # out = np.full(x.shape, False)
        # bisect = np.searchsorted(counts.cumsum()/len(inv), 0.5)
        # idx = inv <= bisect
        # sub_chart = chart[bisect] if bisect == 0 else chart[idx]
        # out[idx] = self.valid_idx(x[idx], y[idx], sub_chart)
        # idx = ~idx
        # sub_chart = chart[bisect] if bisect == len(charts) - 1 else chart[idx]
        # out[idx] = self.valid_idx(x[idx], y[idx], sub_chart)
        # return out

    # TODO impl
    def translate(self):
        raise NotImplementedError()


def healpix_base_atlas():
    import numpy as np

    return np.array([
        [4, 3, 1, 5],
        [5, 0, 2, 6],
        [6, 1, 3, 7],
        [7, 2, 0, 4],
        [11, 3, 0, 8],
        [8, 0, 1, 9],
        [9, 1, 2, 10],
        [10, 2, 3, 11],
        [11, 4, 5, 9],
        [8, 5, 6, 10],
        [9, 6, 7, 11],
        [10, 7, 4, 8],
    ], dtype=np.uint8)


def healpix_side_subpixel_luts():
    """
    Get look-up-tables (LUTs) mapping from HEALPix sub-pixels (as given by the
    HEALPix NEST convention) to corners' next-sides (as stored in the
    HEALPix dual-atlas given by ``atlas_dual(healpix_base_atlas)``).

    Viewed from earth (inside the HEALPix sphere, XY axes swapped):

    .. code-block:: text

       1    2◄───3
       │┌───┬───┐
       ▼│1,0│1,1│
       3├───┼───┤1
        │0,0│0,1│▲
        └───┴───┘│
       0───►0    2

    Returns
    -------
    subpixel2side : array
        Pass NEST subpixel index in to get corner next-side (see arrows in
        diagram above).
    side2subpixel : array
        Pass corner next-side index in to get NEST subpixel (opposite direction
        from arrows in diagram above).

    See Also
    --------
    healpix_base_atlas :
        Get the HEALPix atlas.
    atlas_dual :
        Calculate the dual of an atlas (corner nodes vs. face nodes) whose
        planar form is specified in the same manner as in `healpix_base_atlas`.
    """
    import numpy as np

    return np.array([0, 3, 1, 2]), np.array([0, 2, 3, 1])


def healpix_grid_limits(nside):
    """
    Get the chart grid limits for a HEALPix base pixel of size nside.

    Parameters
    ----------
    nside : int
        HEALPix nside.

    Returns
    -------

    """
    # TODO implement
    raise NotImplementedError()


def idx_in_chart(
        x: NDArray[Int],
        y: NDArray[Int],
        x_limits: NDArray[Int],
        y_limits: NDArray[Int],
) -> NDArray[Bool]:
    """
    See which indices are contained in a chart. This notion can be used to
    keep charts in an atlas evenly aligned at the borders to minimize overlap.

    Parameters
    ----------
    x : array
        x indices to check for membership.
    y : array
        y indices to check for membership.
    x_limits : array
        Min/max x values within the base y-values band (left/right walls)
    y_limits : array
        Min/max y values within the base x-values band (top/bottom walls)

    Returns
    -------
    inside : NDArray[Bool]
        Whether each pixel in ``x, y`` is in the boundaries, broadcast along
        the 3rd...Nth dimensions of ``x_limits`` and ``y_limits`` (in case
        multiple grids with the same x/y bands are being tested at once).

    Raises
    ------
    ValueError
        If the input limits don't intersect the x/y band corners or if input
        array shapes do not match.
    """
    import numpy as np

    dim = y_limits.shape[0], x_limits.shape[0]
    if not np.all((x_limits[0, [0, -1]] == 0) &
                  (x_limits[1, [0, -1]] == dim[0]) &
                  (y_limits[0, [0, -1]] == 0) &
                  (y_limits[1, [0, -1]] == dim[1])):
        raise ValueError("x/y limits must intersect corners.")

    idx = np.minimum(dim[1], np.maximum(0, y))
    res = x >= x_limits[0, idx]
    res &= x <= x_limits[1, idx]
    idx = np.minimum(dim[0], np.maximum(0, x))
    res &= y >= y_limits[0, idx]
    res &= y <= y_limits[1, idx]
    return res


def atlas_dual(faces):
    import numpy as np

    right, face = np.meshgrid(*map(np.arange, faces.shape[::-1]))
    loops = np.stack([*map(np.ravel, [np.roll(right, 1), face, right])]).T
    return close_loops(faces, loops)


def close_loops(faces, loops):
    import numpy as np

    next_sides = loops[:, -1]
    last_faces = loops[:, -2]
    next_faces = faces[last_faces, next_sides]
    next_last_sides = np.where(
        faces[next_faces] ==
        last_faces.reshape((-1, 1))
    )[1]
    is_done = (loops[:, 0] == next_last_sides) & (loops[:, 1] == next_faces)
    done = loops[is_done]

    # dedup and clean up
    k = done.shape[1]
    roll = 3 * done[:, np.arange(1, k, 3)].argsort()[:, 0]
    rows, col_idx = np.ogrid[:done.shape[0], :k]
    col_idx = (col_idx + roll[:, None]) % k
    next_res = done[rows, col_idx]
    srt_res = next_res[np.lexsort([next_res[:, i] for i in range(k - 2, 0, -3)])]
    for i in range(1, k // 3):
        assert (srt_res[::k // 3] == srt_res[i::k // 3]).all()
    final = srt_res[::k // 3].reshape((-1, k // 3, 3))

    remove = (
            loops[is_done, :2].reshape((1, -1, 2)) ==
            loops[:, :2].reshape((-1, 1, 2))
    ).all(axis=2).any(axis=1)
    if remove.all():
        return [final]
    next_next_sides = ((
                               next_last_sides[~remove].repeat(2).reshape((-1, 2)) + (1 + 2 * np.arange(2))
                       ) % 4).ravel()
    next_loops = np.concatenate(
        [
            loops[~remove].repeat(2, 0),
            np.stack([
                next_last_sides[~remove].repeat(2),
                next_faces[~remove].repeat(2),
                next_next_sides
            ]).T
        ],
        1,
    )
    return [final, *close_loops(faces, next_loops)]


def next_side(faces):
    import numpy as np

    return np.argwhere(
        faces[faces] == np.arange(12)[:, None, None]
    )[:, 2].reshape(faces.shape)


def dispatch(
        selector: Union[NDArray[Int], int],
        func_getter: Callable[[int], Callable],
        empty: Callable,
        *args: Any,
) -> Any:
    """
    Dispatch a `func` based on some `selector`.

    Parameters
    ----------
    selector : Union[NDArray[Int], int]
        Indices uniquely specifying the function to be dispatched. Can be a
        scalar if a single dispatched function is intended to be broadcast.
    func_getter : Callable
        A scalar getter function which selects, using a scalar int from
        `selector`, a vector function taking `*args` as its arguments.
    empty : Callable
        A function returning a type-correct default value in case empty
        arguments are passed.
    *args : Any
        Arguments to the return value of `func_getter`.

    Returns
    -------
    result : Any
        The results of calling the selected versions of `func` on the input
        `args`. Has the same return type as `func`.
    """
    import numpy as np

    if np.isscalar(selector):
        return func_getter(selector)(*args)
    if len(selector) == 0:
        return empty()
    # make the check logarithmic in selector count
    uniq, inv, counts = np.unique(selector, return_inverse=True,
                                  return_counts=True)
    if len(uniq) == 1:
        return func_getter(uniq[0])(*args)
    out = np.full(selector.shape, False)
    bisect = np.searchsorted(counts.cumsum() / len(inv), 0.5)
    idx = inv <= bisect
    sub = selector[bisect] if bisect == 0 else selector[idx]
    out[idx] = dispatch(sub, func_getter, empty,
                        *(x if np.isscalar(x) else x[idx] for x in args))
    idx = ~idx
    sub = selector[bisect] if bisect == len(uniq) - 1 else selector[idx]
    out[idx] = dispatch(sub, func_getter, empty,
                        *(x if np.isscalar(x) else x[idx] for x in args))
    return out


def _empty_translate():
    import numpy as np

    return (
        np.array([], dtype=np.bool_),
        np.array([], dtype=np.int64),
        np.array([], dtype=np.int64),
        np.array([], dtype=np.int64),
    )


def square_matrix_connectors() -> NDArray[np.int64]:
    """
    Matrices for connecting identically-sized square charts to each other.

    Returns
    -------
    connectors : NDArray[Shape["4, 4, 2, 3"], Int64]
        Coordinate connection matrices. First 2 indices correspond to input
        side followed by output side (relative to output chart). Last 2 indices
        define augmented transformation matrices, with
        `connectors[i, j, :, :2]` defining the xy rotation matrix A and
        `connectors[i, j, :, 2]` defining the constant offset B⃗, so that the
        coordinate transformation leaving through side `i` and entering the
        next chart through side `j` is given by `Ax⃗  + B⃗`. Handedness of each
        chart is assumed to be the same. B⃗ is initially set to a factor which
        must be multiplied by the grid size.
    """
    import numpy as np

    return np.array([-1, 0, 0, 0, -1, 1, 0, -1, 1, 1, 0, 1, 1, 0,
                     1, 0, 1, 0, 0, 1, 0, -1, 0, 0, 0, 1, -1, -1,
                     0, 1, -1, 0, 1, 0, -1, 2, 0, -1, 2, 1, 0, 0,
                     1, 0, 0, 0, 1, -1, 1, 0, -1, 0, 1, 0, 0, 1,
                     0, -1, 0, 2, -1, 0, 2, 0, -1, 1, 0, -1, 1, 1,
                     0, -1, 0, -1, 0, 1, 0, 0, 1, 0, 0, 0, 1, 1,
                     0, 1, 1, -1, 0, 1, -1, 0, 1, 0, -1, 0]) \
        .reshape((4, 4, 2, 3))
