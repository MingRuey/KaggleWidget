import tensorflow as tf
from tensorflow.python.ops import math_ops

from MLBOX.Database.dataset import DataBase
from create_database import FeaturesV0
from variables import train_features_v0, train_csv

bond_table = tf.lookup.StaticHashTable(
    tf.lookup.KeyValueTensorInitializer(
        keys=['1JHC', '1JHN', '2JHH', '2JHC', '2JHN', '3JHH', '3JHC', '3JHN'],
        values=[0, 1, 2, 3, 4, 5, 6, 7]
    ),
    default_value=-1,
    name="BondTypeLookUp"
)

atom_table = tf.lookup.StaticHashTable(
    tf.lookup.KeyValueTensorInitializer(
        keys=['H', 'C', 'N', 'O', 'F'],
        values=[0, 1, 2, 3, 4]
    ),
    default_value=-1,
    name="AtomTypeLookUp"
)


def one_hot_bond_dist(bond_dist: tf.Tensor) -> tf.Tensor:
    """Categorize bond by bond dist with pre-defined bins"""
    bins = [float(num) for num in range(2, 29, 3)]
    bucket = math_ops._bucketize(bond_dist, boundaries=bins)
    return tf.one_hot(bucket, depth=10)


def one_hot_bond_types(bond: tf.Tensor) -> tf.Tensor:
    """Simple map-and-one-hot operation."""
    indices = bond_table.lookup(bond)
    return tf.one_hot(indices=indices, depth=8)


def one_hot_atom_types(atom: tf.Tensor) -> tf.Tensor:
    """Simple map-and-one-hot operation"""
    indices = atom_table.lookup(atom)
    return tf.one_hot(indices=indices, depth=5)


def pairwise_dist(pos: tf.Tensor) -> tf.Tensor:
    """Calculate parwise Euclidean(square) distance between nodes

    Args:
        pos (tf.Tensor):
            positions of each node
            tensor of shape (batch, nodes, D),
            where D is the dimension of position vector

    Returns:
        tensor of shape (batch, nodes, nodes),
        (batch, i, j) is the distance between node i and j in the batch
        note that the tensor is symmertic in nodes dimension,
        i.e. (batch, i, j) == (batch, j, i)
    """
    sq_i = tf.einsum("bnd,bnd->bn", pos, pos)
    sq_j = tf.transpose(sq_i)
    dist = sq_i - 2 * tf.einsum("bni,bmi->bnm", pos, pos) + sq_j
    return dist


def batch_bond(
        bond_atoms: tf.Tensor,
        atom_types: tf.Tensor,
        ) -> tf.Tensor:
    """Create atoms' features for each bond

    Args:
        bond_atoms (tf.Tensor):
            tensor of shape (bonds, 2),
            specify the two indices of atoms that bond connects
        atoms_types (tf.Tensor):
            tensor of shape (atoms, atom types),
            specify the types of atoms in one-hoted vectors
    Returns:
        tensor of shape (bond, atoms, feature length),
        the feature vectors of each bond
    """
    bond_count = tf.shape(bond_atoms)[0]
    atom_count = tf.shape(atom_types)[0]

    atom_anno = tf.one_hot(bond_atoms, depth=atom_count)  # shape (bonds, 2, atoms)
    atom_anno = tf.reduce_sum(atom_anno, axis=1)  # shape (bonds, atoms)
    atom_anno = tf.expand_dims(atom_anno, axis=-1)  # shape (bonds, atoms, 1)
    atom_types = tf.expand_dims(atom_types, axis=0)
    atom_types = tf.tile(atom_types, multiples=[bond_count, 1, 1])  # shape (bonds, atoms, atom_types)
    atom_features = tf.zeros(shape=(bond_count, atom_count, 58))

    atom_features = tf.concat([atom_anno, atom_types, atom_features], axis=-1)
    return atom_features


if __name__ == "__main__":
    dataformat = FeaturesV0()
    db = DataBase(formats=dataformat)
    db.load_path(train_features_v0)
    db.config_parser()

    for data, label in db.get_input_tensor(epoch=1, batchsize=1):
        bonds = data["data"]["bonds"][0]
        if bonds.shape[0] < 5:
            print("bonds: ", bonds)
            # bond_types = one_hot_bond_types(bonds)
            # print("bond_types: ", bond_types)

            index_1 = data["data"]["bonds/index1"][0]
            index_2 = data["data"]["bonds/index2"][0]
            bond_atoms = tf.stack([index_1, index_2], axis=-1)
            # print("index_1: ", index_1)
            # print("index_2: ", index_2)
            print("bond_atoms: ", bond_atoms)

            atoms = data["data"]["atoms"][0]
            atom_types = one_hot_atom_types(atoms)
            atom_features = batch_bond(
                bond_atoms=bond_atoms, atom_types=atom_types
            )

            print("atom_features: ", atom_features)
            # print("atoms: ", atoms)
            # print("atom_types: ", atom_types)

            bond_count = tf.shape(atom_features)[0]
            atom_count = tf.shape(atom_features)[1]

            x_pos = data["data"]["atoms/x"]
            y_pos = data["data"]["atoms/y"]
            z_pos = data["data"]["atoms/z"]
            dist = pairwise_dist(tf.stack([x_pos, y_pos, z_pos], axis=-1))[0]
            dist_mat = one_hot_bond_dist(dist)  # (atoms, atoms, 15)
            dist_mat = tf.expand_dims(dist_mat, axis=0)
            dist_mat = tf.tile(dist_mat, multiples=[bond_count, 1, 1, 1])  # shape (bonds, atoms, atoms, 15)

            # print("x_pos: ", x_pos)
            # print("y_pos: ", y_pos)
            # print("z_pos: ", z_pos)
            # print("dist: ", dist)
            print("dist: ", dist)
            print("one-hot dist: ", dist_mat)

            adjacent_mat = 1.0 - tf.eye(atom_count, batch_shape=[bond_count])
            print("adj_mat: ", adjacent_mat)
            break
