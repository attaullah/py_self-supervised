import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda")


def triplet_loss(x_np, y_np, dev=device):
    x_np = F.normalize(x_np, p=2, dim=1)
    pdist_matrix = torch.cdist(x_np, x_np)
    # Build pairwise binary adjacency matrix.
    batch_size = y_np.size(0)
    adjacent = torch.zeros([batch_size, batch_size]).type(torch.int32).to(dev)  # create adjacent matrix

    for i in range(batch_size):
        for j in range(batch_size):
            if y_np[i] == y_np[j]:
                adjacent[i, j] = 1
    # # Invert so we can select negatives only.
    adjacency_not = torch.logical_not(adjacent)

    # Compute the mask.
    pdist_matrix_tile = torch.tile(pdist_matrix, [batch_size, 1])

    mask = torch.logical_and(
        torch.tile(adjacency_not, [batch_size, 1]),
        torch.greater(pdist_matrix_tile,
                      torch.reshape(torch.transpose(pdist_matrix, 0, -1)
                                    , [-1, 1])
                      )
    )
    mask_final = torch.reshape(
        torch.greater(
            torch.sum(
                mask.type(torch.float32), 1, keepdims=True),
            0.0), [batch_size, batch_size])
    mask_final = torch.transpose(mask_final, 0, -1)

    adjacency_not = adjacency_not.type(torch.float32)
    mask = mask.type(torch.float32)

    def masked_minimum(data, mask, dim=1):
        axis_maximums = torch.max(data, dim, keepdims=True)  # dim,)#
        # print('axis ',axis_maximums[0])
        masked_minimums = torch.min(
            torch.multiply(data - axis_maximums[0], mask), dim,
            keepdims=True)[0] + axis_maximums[0]
        return masked_minimums
        # negatives_outside: smallest D_an where D_an > D_ap.

    negatives_outside = torch.reshape(
        masked_minimum(pdist_matrix_tile, mask), [batch_size, batch_size])
    negatives_outside = torch.transpose(negatives_outside, 0, -1)

    def masked_maximum(data, mask, dim=1):
        axis_minimums = torch.min(data, dim, keepdims=True)
        masked_maximums = torch.max(
            torch.multiply(data - axis_minimums[0], mask), dim,
            keepdims=True)[0] + axis_minimums[0]
        return masked_maximums

    # negatives_inside: largest D_an.
    negatives_inside = torch.tile(
        masked_maximum(pdist_matrix, adjacency_not), [1, batch_size])
    semi_hard_negatives = torch.where(  # array_ops
        mask_final, negatives_outside, negatives_inside)
    margin = 1.0
    loss_mat = torch.add(margin, pdist_matrix - semi_hard_negatives)

    mask_positives = adjacent.type(torch.float32) - torch.diag(torch.ones([batch_size]).to(dev))

    #   in semihard, they take all positive pairs except the diagonal.
    num_positives = torch.sum(mask_positives)

    semi_hard_triplet_loss_distance = torch.true_divide(
        torch.sum(
            torch.maximum(
                torch.multiply(loss_mat, mask_positives), torch.zeros(1).to(dev))),
        num_positives + 1e-16)

    return semi_hard_triplet_loss_distance