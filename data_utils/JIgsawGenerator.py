import numpy as np
from itertools import product


def generate_permutation_list(classes=10):
    import itertools
    from scipy.spatial.distance import cdist
    from tqdm import trange
    outname = f'permutations_{classes}.npy'

    P_hat = np.array(list(itertools.permutations(list(range(9)), 9)))
    n = P_hat.shape[0]

    for i in trange(classes):
        if i == 0:
            j = np.random.randint(n)
            P = np.array(P_hat[j]).reshape([1, -1])
        else:
            P = np.concatenate([P, P_hat[j].reshape([1, -1])], axis=0)

        P_hat = np.delete(P_hat, j, axis=0)
        D = cdist(P, P_hat, metric='hamming').mean(axis=0).flatten()
        j = D.argmax()

        if i % 100 == 0:
            np.save(outname, P)

    np.save(outname, P)
    print('file created --> ' + outname)


def cut_jigsaw(
        in_image,  # type: np.ndarray
        x_wid,  # type: int
        y_wid,  # type: int
        gap=False,
        jitter=False,
        jitter_dim=None,  # type: Optional[int]
):
    # type: (...) -> List[np.ndarray]
    """Cuts the image into little pieces
    :param in_image: the image to cut-apart
    :param x_wid: the size of the piece in x
    :param y_wid: the size of the piece in y
    :param gap: if there is a gap between tiles
    :param jitter: if the positions should be moved around
    :param jitter_dim: amount to jitter (default is x_wid or y_wid/2)
    :return : a 4D array with tiles x x_wid x y_wid * d
    """
    if len(in_image.shape) == 2:
        in_image = np.expand_dims(in_image, -1)
        expand = True
    else:
        expand = False
    x_size, y_size, d_size = in_image.shape
    x_chunks = x_size // x_wid
    y_chunks = y_size // y_wid
    out_tiles = np.zeros((x_chunks * y_chunks, x_wid, y_wid, d_size), dtype=in_image.dtype)
    if gap:
        # we calculate the maximum gap and
        x_gap = x_size - x_chunks * x_wid
        y_gap = y_size - y_chunks * y_wid
    else:
        x_gap, y_gap = 0, 0
    x_jitter = x_wid // 2 if jitter_dim is None else jitter_dim
    y_jitter = y_wid // 2 if jitter_dim is None else jitter_dim
    for idx, (i, j) in enumerate(product(range(x_chunks), range(y_chunks))):
        x_start = i * x_wid + min(x_gap, i)
        y_start = j * y_wid + min(y_gap, j)
        if jitter:
            x_range = max(x_start - x_jitter, 0), min(x_start + x_jitter + 1, x_size - x_wid)
            y_range = max(y_start - y_jitter, 0), min(y_start + y_jitter + 1, y_size - y_wid)

            x_start = np.random.choice(range(*x_range)) if x_range[1] > x_range[0] else x_start
            y_start = np.random.choice(range(*y_range)) if y_range[1] > y_range[0] else y_start

        out_tiles[idx, :, :, :] = in_image[x_start:x_start + x_wid, y_start:y_start + y_wid, :]

    return out_tiles[:, :, :, 0] if expand else out_tiles


def jigsaw_to_image(
        in_tiles,  # type: np.ndarray
        out_x,  # type: int
        out_y,  # type: int
        gap=False
):
    # type: (...) -> np.ndarray
    """Reassembles little pieces into an image
    :param in_tiles: the tiles to reassemble
    :param out_x: the size of the image in x (default is calculated automatically)
    :param out_y: the size of the image in y
    :param gap: if there is a gap between tiles
    :return : an image from the tiles

    """
    if len(in_tiles.shape) == 3:
        in_tiles = np.expand_dims(in_tiles, -1)
        expand = True
    else:
        expand = False
    tile_count, x_wid, y_wid, d_size = in_tiles.shape
    x_chunks = out_x // x_wid
    y_chunks = out_y // y_wid
    out_image = np.zeros((out_x, out_y, d_size), dtype=in_tiles.dtype)

    if gap:
        x_gap = out_x - x_chunks * x_wid
        y_gap = out_y - y_chunks * y_wid
    else:
        x_gap, y_gap = 0, 0

    for idx, (i, j) in enumerate(product(range(x_chunks), range(y_chunks))):
        x_start = i * x_wid + min(x_gap, i)
        y_start = j * y_wid + min(y_gap, j)
        out_image[x_start:x_start + x_wid, y_start:y_start + y_wid] = in_tiles[idx, :, :]

    return out_image[:, :, 0] if expand else out_image


def jigsaw_transform(image, permutations=100, one_hot=False):
    """
    Generates jigsaw puzzle patches.
    :param image: input image
    :param permutations: number of permutations
    :param one_hot: apply one-hot encoding to labels
    :return: jigsaw images and labels
    """
    permutations_list = np.load(f'./data_utils/permutations_{permutations}.npy')
    label = np.random.randint(len(permutations_list))
    selected_perm = permutations_list[label]
    if image.ndim == 4:
        image = image.squeeze()
    size = image.shape[1]
    patches_size = size // 3
    out_tiles = cut_jigsaw(image, patches_size, patches_size, gap=False)
    out_tiles = out_tiles[selected_perm]
    if one_hot:
        label = np.eye(permutations)[label]
    recon_img = jigsaw_to_image(out_tiles, size, size)
    return recon_img, label


def jigsaw_transform_tile(image, permutations=10, one_hot=False):
    permutations_list = np.load(f'./data_utils/permutations_{permutations}.npy')
    # label = np.random.randint(len(permutations_list))
    # selected_perm = permutations_list[label]
    if image.ndim == 4:
        image = image.squeeze()
    size = image.shape[1]
    patches_size = size // 3
    out_tiles = cut_jigsaw(image, patches_size, patches_size, gap=False)
    out_all_tiles = [out_tiles[c_perm] for c_perm in permutations_list]
    out_all_tiles = np.array(out_all_tiles)
    label = np.arange(len(permutations_list))
    # label = label.squeeze()
    # print(label.shape)
    # out_tiles = out_tiles[selected_perm]
    # if one_hot:
    #     label = np.eye(permutations)[label]
    #     print(label.shape)

    return out_all_tiles, label


def combined_jigsaw_generator(super_iter, jigsaw_iter, batch_size, one_hot=True, n_classes=100):
    """
    Utility function to load data into required Keras model format for training on supervised batch and self-supervised
    batch.
    :param super_iter: supervised data generator based on labelled training images
    :param jigsaw_iter: self-supervised data generator based on unlabelled training images
    :param batch_size: size of mini-batch
    :param one_hot: use one-hot encoding
    :param n_classes:
    """
    super_batch = batch_size * 1
    self_batch = batch_size
    while True:
        x_super, y_super = zip(*[next(super_iter) for _ in range(super_batch)])
        x_self, y_self = zip(*[jigsaw_transform(next(jigsaw_iter), one_hot=one_hot, permutations=n_classes)
                               for _ in range(self_batch)])

        x_super = np.vstack(x_super)
        y_super = np.vstack(y_super)
        x_self = np.array(x_self)
        y_self = np.vstack(y_self)
        # print("print before yield   ", x_self.shape, y_self.shape, x_super.shape, y_super.shape)
        yield [x_self, x_super], [y_self, y_super]


def jigsaw_data_generator(self_iter, batch_size, one_hot=True, n_classes=10):
    """
    Utility function to load data into required Keras model format.
    :param self_iter: self-supervised data generator based on unlabelled training images
    :param batch_size: size of mini-batch
    :param one_hot: use one-hot encoding
    :param n_classes:
    """
    while True:
        x_self, y_self = zip(*[jigsaw_transform(next(self_iter), one_hot=one_hot, permutations=n_classes)
                               for _ in range(batch_size)])
        x_self = np.array(x_self)
        y_self = np.array(y_self)
        # x_self = np.vstack(x_self)
        # y_self = np.vstack(y_self)
        # print("print before yield   ", x_self.shape, y_self.shape)
        yield x_self, y_self


def jigsaw_tile_data_generator(self_iter, batch_size, one_hot=False, n_classes=100):
    """
    Utility function to load data into required Keras model format.
    :param self_iter: self-supervised data generator based on unlabelled training images
    :param batch_size: size of mini-batch
    :param one_hot: use one-hot encoding
    :param n_classes:
    """
    while True:
        x_self, y_self = zip(*[jigsaw_transform_tile(next(self_iter), one_hot=one_hot, permutations=n_classes)
                               for _ in range(batch_size)])

        x_self = np.array(x_self)
        # print("before vstack ", x_self.shape)
        x_self = np.vstack(x_self)
        y_self = np.tile(np.arange(n_classes), batch_size)
        # print("before vstack ", y_self.shape)

        # y_self = np.vstack(y_self)
        # print("print before yield   ", x_self.shape, y_self.shape, one_hot)
        yield x_self, y_self


def combined_jigsaw_tile_generator(super_iter, jigsaw_iter, batch_size, one_hot=False, n_classes=100):
    """
    Utility function to load data into required Keras model format for training on supervised batch and self-supervised
    batch.
    :param super_iter: supervised data generator based on labelled training images
    :param jigsaw_iter: self-supervised data generator based on unlabelled training images
    :param batch_size: size of mini-batch
    :param one_hot: use one-hot encoding
    :param n_classes:
    """
    super_batch = batch_size * 1
    self_batch = batch_size
    while True:
        x_super, y_super = zip(*[next(super_iter) for _ in range(super_batch)])
        x_self, y_self = zip(*[jigsaw_transform_tile(next(jigsaw_iter), one_hot=one_hot, permutations=n_classes)
                               for _ in range(self_batch)])

        x_super = np.vstack(x_super)
        y_super = np.vstack(y_super)
        x_self = np.array(x_self)
        x_self = np.vstack(x_self)
        y_self = np.tile(np.arange(n_classes), batch_size)
        # print("print before yield   ", x_self.shape, y_self.shape, x_super.shape, y_super.shape)
        yield [x_self, x_super], [y_self, y_super]

