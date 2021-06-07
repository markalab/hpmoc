"Work with manifolds."


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


def atlas_dual(atlas):
    import numpy as np

    right, face = np.meshgrid(*map(np.arange, atlas.shape[::-1]))
    loops = np.stack([*map(np.ravel, [np.roll(right, 1), face, right])]).T
    return close_loops(atlas, loops)


def close_loops(atlas, loops):
    import numpy as np

    next_sides = loops[:,-1]
    last_faces = loops[:,-2]
    next_faces = atlas[last_faces, next_sides]
    next_last_sides = np.where(
        atlas[next_faces] ==
        last_faces.reshape((-1, 1))
    )[1]
    is_done = (loops[:,0]==next_last_sides) & (loops[:,1]==next_faces)
    done = loops[is_done]

    # dedup and clean up
    k = done.shape[1]
    roll = 3*done[:,np.arange(1, k, 3)].argsort()[:,0]
    rows, col_idx = np.ogrid[:done.shape[0], :k]
    col_idx = (col_idx + roll[:, None])%k
    next_res = done[rows, col_idx]
    srt_res = next_res[np.lexsort([next_res[:,i] for i in range(k-2, 0, -3)])]
    for i in range(1, k//3):
        assert (srt_res[::k//3]==srt_res[i::k//3]).all()
    final = srt_res[::k//3].reshape((-1, k//3, 3))

    remove = (
        loops[is_done,:2].reshape((1, -1, 2)) ==
        loops[:,:2].reshape((-1, 1, 2))
    ).all(axis=2).any(axis=1)
    if remove.all():
        return [final]
    next_next_sides = ((
        next_last_sides[~remove].repeat(2).reshape((-1, 2))+(1+2*np.arange(2))
    )%4).ravel()
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
    return [final, *close_loops(atlas, next_loops)]
