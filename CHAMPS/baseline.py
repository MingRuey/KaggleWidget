import torch
import torch.nn as nn


TOTAL_TIME_STEP = 3
FEATURE_LENGTH = annotation(1) + edge_type(C52) + FEATURE_LENGTH

node_vectors = Matrix((batch_size, num_of_edges -> CN2(num_of_atom), FEATURE_LENGTH))  # input
edge_type_matrix = Matrix((edge_type, num_of_edges))  # input

# message passing (MP)
mp_matrix = Matrix((FEATURE_LENGTH, FEATURE_LENGTH, edge_type))  # learnable

for t in range(TOTAL_TIME_STEP):
    new_node_vector = Matrix((batch_size, num_of_edges, FEATURE_LENGTH))

    select_mp_matrix = mp_matrix * edge_type_matrix -> array of shape (FEATURE_LENGTH, FEATURE_LENGTH, num_of_edges)
    update_matrix = node_vectors
    update_matrix = node_vectors * select_mp_matrix -> array of shape (batch_size, num_of_edges, FEATURE_LENGTH)

        for edge_self in range(num_of_edges):

            message = empty -> array of shape (FEATURE_LENGTH, )
            for edge_other in range(num_of_edges - 1):
                edge_state = node_vectors[b, edge_other, :]  -> array of shape (FEATURE_LENGTH, )
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
    out.backward(retain_graph=True)CH
    print(x)
    print(x.grad)
    print(g)
