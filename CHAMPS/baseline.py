import torch
import torch.nn as nn

TOTAL_TIME_STEP = 3

feature_length = annotation(1) + edge_type(C52) + feature_length
node_vectors = Matrix((batch_size, num_of_edges -> CN2(num_of_atom), feature_length))

# message passing
message_passing_matrix = Matirx((feature_length, feature_length), edge_type=edge_type)

for t in range(TOTAL_TIME_STEP):
    new_node_vector = Matrix((batch_size, num_of_edges, feature_length))

    for b in range(batch_size):
        for edge_self in range(num_of_edges):
            message = empty -> array of shape (feature_length, )
            for edge_other in range(num_of_edges - 1):
                edge_state = node_vectors[b, edge_other, :]  -> array of shape (feature_length, )
                message += message_passing_matrix * edge_state

            new_feature = Gru_update(message, node_vecotr(b, edge_self, :))
            new_node_vector[b, edge_self, :] = new_feature

    node_vector = new_node_vector[:, :, :]

result = ReadOutFunction(node_vector)


if __name__ == "__main__":

    x = torch.randn((3, 3), requires_grad=True)
    y = x * x
    g = y + 5
    out = g.sum()
    out.backward(retain_graph=True)
    print(x)
    print(x.grad)
    print(g)
    out.backward(retain_graph=True)
    print(x)
    print(x.grad)
    print(g)
