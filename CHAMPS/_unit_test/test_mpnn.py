import sys
import os.path as path
import pytest
import torch

path = path.dirname(path.dirname(__file__))
if path not in sys.path:
    sys.path.append(path)

from mpnn import MessagePassing, GruUpdate, GGNNReadOut, MPNN  # noqa 402


NUM_OF_NODES = 2
FEATURE_LENGTH = 5
EDGE_TYPES = 3
REPR_LENGTH = 8


def test_message_pass():

    model = MessagePassing(FEATURE_LENGTH, EDGE_TYPES)
    print("message passing weight: ", model.W.shape)

    e_feat = torch.randn(2, NUM_OF_NODES, FEATURE_LENGTH)
    print("feature state: ", e_feat.shape)

    e_type = torch.Tensor([
        [[[0, 0, 0], [0, 0, 1]], [[1, 0, 0], [0, 0, 0]]],
        [[[0, 0, 1], [0, 1, 0]], [[0, 1, 0], [1, 0, 0]]]
    ])
    print("edge type: ", e_type.shape)

    adjacent = torch.Tensor([
        [[0, 1], [1, 0]], [[1, 1], [1, 1]]
    ])
    print("adjacent: ", adjacent.shape)

    new_state = model(e_feat, e_type, adjacent)
    mp_mat = model._mp_mat
    update_mat = model._sum_mat
    print("mp_mat: ", mp_mat.shape)  # (num_of_nodes, f, f)
    print("update_tensor: ", update_mat.shape)
    print("new state: ", new_state)

    # batch 1
    assert torch.all(mp_mat[0, 0, 0, :, :] == 0)
    assert torch.all(mp_mat[0, 1, 1, :, :] == 0)

    assert torch.all(torch.eq(mp_mat[0, 0, 1, :, :], model.W[2, :, :]))
    assert torch.all(torch.eq(mp_mat[0, 1, 0, :, :], model.W[0, :, :]))

    assert torch.allclose(update_mat[0, 0, :, :], model.W[2, :, :])
    assert torch.allclose(update_mat[0, 1, :, :], model.W[0, :, :])

    updated0 = torch.matmul(model.W[2, :, :], e_feat[0, 0, :])
    updated1 = torch.matmul(model.W[0, :, :], e_feat[0, 1, :])
    assert torch.allclose(new_state[0, 0, :], updated0)
    assert torch.allclose(new_state[0, 1, :], updated1)

    # batch 2
    assert torch.all(torch.eq(mp_mat[1, 0, 1, :, :], model.W[1, :, :]))
    assert torch.all(torch.eq(mp_mat[1, 1, 0, :, :], model.W[1, :, :]))

    sum_mat0 = model.W[1, :, :] + model.W[2, :, :]
    sum_mat1 = model.W[1, :, :] + model.W[0, :, :]
    assert torch.allclose(update_mat[1, 0, :, :], sum_mat0)
    assert torch.allclose(update_mat[1, 1, :, :], sum_mat1)

    updated0 = torch.matmul(sum_mat0, e_feat[1, 0, :])
    updated1 = torch.matmul(sum_mat1, e_feat[1, 1, :])
    assert torch.allclose(new_state[1, 0, :], updated0)
    assert torch.allclose(new_state[1, 1, :], updated1)


def test_gru_update():
    model = GruUpdate(FEATURE_LENGTH, FEATURE_LENGTH)
    node_mat = torch.randn(2, NUM_OF_NODES, FEATURE_LENGTH)
    mp_mat = torch.randn(2, NUM_OF_NODES, FEATURE_LENGTH)

    new_state = model(mp_mat, node_mat)
    print(new_state.shape)


def test_GGNNReadOut():
    model = GGNNReadOut(FEATURE_LENGTH, REPR_LENGTH)

    node_state0 = torch.randn(2, 2, FEATURE_LENGTH)
    node_state1 = torch.randn(2, 2, FEATURE_LENGTH)

    state = model(node_state0, node_state1)
    print(state.shape)


def test_MPNN():
    model = MPNN(
        mp_steps=3,
        feat_len=FEATURE_LENGTH,
        edge_types=EDGE_TYPES,
        repr_len=REPR_LENGTH
    )

    feat_mat = torch.randn(2, NUM_OF_NODES, FEATURE_LENGTH)

    # BATCH = NUM_OF_NODES = 2
    etype_mat = torch.Tensor([
        [[[0, 0, 0], [0, 0, 1]], [[1, 0, 0], [0, 0, 0]]],
        [[[0, 0, 0], [0, 1, 0]], [[0, 1, 0], [0, 0, 0]]]
    ])
    adj_mat = torch.Tensor([
        [[0, 1], [1, 0]], [[0, 1], [1, 0]]
    ])

    repr_vector = model(feat_mat, etype_mat, adj_mat)
    print(repr_vector.shape)


if __name__ == "__main__":
    pytest.main(["-s", "-v", __file__])
