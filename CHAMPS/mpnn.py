"""
Pytorch implementation of Gated Graph Sequence Neural Networks (ICLR 2016)
https://arxiv.org/abs/1511.05493
"""

import torch
import torch.nn as nn


class MessagePassing(nn.Module):

    def __init__(self, feature_length: int, edge_types_num: int):
        """Init a message passing matrix

        Args:
            feature_length (int): the internal state of node
            edge_types_num (int): the number of edge type
        """
        super().__init__()
        self.W = nn.Parameter(
            torch.randn(edge_types_num, feature_length, feature_length)
        )

    def forward(
            self,
            node_state: torch.Tensor,
            edge_type_mat: torch.Tensor,
            adj_mat: torch.Tensor
            )-> torch.Tensor:
        """Update node states with message passing matrix

        Args:
            node_state (torch.Tensor):
                tensor of shape (batch, nodes, feature length)
            edge_type_mat (torch.Tensor):
                tensor of shape (batch, nodes, nodes, edge types),
                which stands for the edge type between each node pairs
            adj_mat:
                the connectivity between nodes,
                tensor of shape (batch, nodes, nodes),
                whose elements must be 0/1 valued
        Returns:
            message tensor of shape (batch, nodes, feature length)
        """
        # s,o => self,other
        self._mp_mat = torch.einsum("bsoe,eij->bsoij", edge_type_mat, self.W)
        self._sum_mat = torch.einsum("bso,bsoij->bsij", adj_mat, self._mp_mat)
        return torch.einsum("bsij,bsj->bsi", self._sum_mat, node_state)


class GruUpdate(nn.Module):

    def __init__(self, message_length: int, feature_length: int):
        """An GRU cell updates node state with message passing matrix

        Args:
            message_length: the length of the message passing feature
                for GGNN model, this is always equal to feature length
            feature_length: the length of the node feature
        """
        super().__init__()
        self._gru_cell = nn.GRUCell(
            input_size=message_length, hidden_size=feature_length, bias=True
        )

    def forward(
            self,
            message_tensor: torch.Tensor,
            node_state: torch.Tensor,
            ) -> torch.Tensor:
        """Forward GRU update

        Args:
            node_state (torch.Tensor):
                tensor of shape (batch, nodes, feature length)
            message_tensor (torch.Tensor):
                tensor of shape (batch, nodes, message_length)

        Returns:
            The new node state
        """
        return torch.stack(
            [self._gru_cell(m, e) for m, e in zip(message_tensor, node_state)]
        )


class GGNNReadOut(nn.Module):

    def __init__(self, feature_length: int, output_length: int):
        """A read-out function defined in GGNN paper

        Args:
            feature_length: the length of the node feature
        """
        super().__init__()
        self._attension = nn.Linear(2*feature_length, output_length)
        self._dense = nn.Linear(feature_length, output_length)

    def forward(self, node_state_0: torch.Tensor, node_state: torch.Tensor):
        """Get global representation of Graph

        Args:
            node_state_0 (torch.Tensor):
                the initial node state at t = 0,
                of shape (batch, nodes, feature length)

            node_state (torch.Tensor):
                the final node state at t = T
                of shape (batch, nodes, feature length)

        Returns:
            Global graph representation with length output_length
        """
        attension = torch.sigmoid(self._attension(
            torch.cat([node_state_0, node_state], dim=-1)
        ))
        dense = self._dense(node_state)
        return torch.sum(dense * attension, dim=1)


def _shape_equal(tensors, dim) -> bool:
    state = True
    for tensor in tensors:
        state = state and (dim < tensor.dim())
        if not state:
            return False

    shapes = [t.shape[dim] for t in tensors]
    return shapes.count(shapes[0]) == len(shapes)


def _tensor_01_valued(tensor: torch.Tensor) -> bool:
    return torch.all((tensor == 1) | (tensor == 0))


class MPNN(nn.Module):

    def __init__(
            self,
            mp_steps: int,
            feat_len: int,
            edge_types: int,
            repr_len: int
            ):
        """Init a MPNN model

        Args:
            mp_steps (int): the total steps for message passing
            feat_len (int): the length of node feature
            edge_types (int): the total number of types of edges
            repr_len (int): the length of representation vector
        """
        super().__init__()
        self._mp_steps = mp_steps
        self._feat_len = feat_len
        self._edge_types = edge_types

        self._mp = MessagePassing(
            feature_length=feat_len, edge_types_num=edge_types
        )
        self._gru = GruUpdate(
            feature_length=feat_len, message_length=feat_len
        )
        self._read_out = GGNNReadOut(
            feature_length=feat_len, output_length=repr_len
        )

    def forward(
            self,
            feat_mat: torch.Tensor,
            etype_mat: torch.Tensor,
            adj_mat: torch.Tensor
            ):
        """Get reper vector of MPNN model

        Args:
            feat_mat (torch.Tensor):
                initial feature state of nodes,
                tensor of shape (batch, nodes, feature length)
            etype_mat (torch.Tensor):
                0/1 valued matrix as node type of nodes,
                tensor of shape (batch, nodes, nodes, edge types)
            adj_mat (torch.Tensor):
                0/1 valued matrix as connectivity of graph,
                tensor of shape (batch, nodes, nodes)
        """
        batch = feat_mat.shape[0]
        node_num = feat_mat.shape[1]
        assert _shape_equal([feat_mat, etype_mat, adj_mat], dim=0)
        assert _shape_equal([feat_mat, etype_mat, adj_mat], dim=1)
        assert feat_mat.shape[2] == self._feat_len
        assert etype_mat.shape[2] == etype_mat.shape[1]
        assert etype_mat.shape[3] == self._edge_types
        assert adj_mat.shape[2] == node_num
        assert _tensor_01_valued(etype_mat)
        assert _tensor_01_valued(adj_mat)

        init_state = feat_mat.clone()
        state = feat_mat.clone()
        for step in range(self._mp_steps):
            mp = self._mp(
                node_state=state, edge_type_mat=etype_mat, adj_mat=adj_mat
            )
            state = self._gru(message_tensor=mp, node_state=state)

        return self._read_out(node_state_0=init_state, node_state=state)
