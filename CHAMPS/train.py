import os
import re
import pathlib

import tensorflow as tf

import torch
import torch.nn as nn
from torch.optim import Adam

cpus = tf.config.experimental.list_physical_devices('CPU')
tf.config.experimental.set_visible_devices(cpus)

from mpnn import MPNN
from mpnn_inputs import batch_bond, pairwise_dist, one_hot_bond_types
from mpnn_inputs import one_hot_bond_dist, one_hot_atom_types

from MLBOX.Database.dataset import DataBase
from MLBOX.Scenes.SimpleSplit import SimpleSplit
from create_database import FeaturesV0
from variables import train_features_v0, train_csv


MODEL_FILE = "MPNN_Ep{}_Train-{:.4f}_VAL-{:.4f}.pt"

WEIGHT = None
INIT_EPOCH = 0
_pattern = re.compile("EP\d+")
for _file in pathlib.Path(os.path.dirname(__file__)).glob("*.pt"):
    _epoch = re.match(_pattern)
    if not _epoch:
        msg = "Can't extract epoch count from {}"
        raise ValueError(msg.format(_file))
    else:
        _epoch = _epoch.group()[2:]
        if _epoch > INIT_EPOCH:
            WEIGHT = str(_file)
            INIT_EPOCH = _epoch

TOTAL_EPOCH = 10
MESSAGE_PASSING_STEPS = 10
FEATURE_LENGTH = 64
EDGE_TYPES = 10
REPR_LEN = 16


def process_input(data, label):
    index_1 = data["data"]["bonds/index1"][0]  # (bonds, )
    index_2 = data["data"]["bonds/index2"][0]  # (bonds, )
    bond_atoms = tf.stack([index_1, index_2], axis=-1)  # (bonds, 2)

    atoms = data["data"]["atoms"][0]  # (atoms, )
    atom_types = one_hot_atom_types(atoms)  # (atoms, atom_types)

    atom_features = batch_bond(
        bond_atoms=bond_atoms, atom_types=atom_types
    )  # (bond, atoms, feature length)

    bond_count = tf.shape(atom_features)[0]
    atom_count = tf.shape(atom_features)[1]

    x_pos = data["data"]["atoms/x"]  # (batch, atoms)
    y_pos = data["data"]["atoms/y"]  # (batch, atoms)
    z_pos = data["data"]["atoms/z"]  # (batch, atoms)
    dist = pairwise_dist(tf.stack([x_pos, y_pos, z_pos], axis=-1))[0]  # (atoms, atoms)
    dist_mat = one_hot_bond_dist(dist)  # (atoms, atoms, 15)
    dist_mat = tf.expand_dims(dist_mat, axis=0)
    dist_mat = tf.tile(dist_mat, multiples=[bond_count, 1, 1, 1])  # shape (bonds, atoms, atoms, 15)

    adjacent_mat = 1.0 - tf.eye(atom_count, batch_shape=[bond_count])
    target = label[0]  # -> ["data"]["bonds/index1"], shape (bonds, 1)

    return (
        atom_features.numpy(),
        dist_mat.numpy(),
        adjacent_mat.numpy(),
        target.numpy()
    )


if __name__ == "__main__":

    dataformat = FeaturesV0()
    db = DataBase(formats=dataformat)
    db.load_path(train_features_v0)
    scene = SimpleSplit(database=db, ratio_for_validation=0.2)

    train_db = scene.get_train_dataset()
    vali_db = scene.get_vali_dataset()
    train_db.config_parser()
    vali_db.config_parser()
    vali_count = vali_db.data_count
    train_count = train_db.data_count

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = MPNN(
        mp_steps=MESSAGE_PASSING_STEPS,
        feat_len=FEATURE_LENGTH,
        edge_types=EDGE_TYPES,
        repr_len=REPR_LEN
    )
    model = nn.DataParallel(model)
    model.to(device)

    if WEIGHT:
        model.load_state_dict(torch.load(WEIGHT))
        print("Model load weight from: ", WEIGHT)
    else:
        print("No .pt file found, train model from scratch")

    optim = Adam(
        params=model.parameters(), lr=0.001, betas=(0.9, 0.999),
        eps=1e-8, weight_decay=0, amsgrad=False
    )

    loss_func = nn.L1Loss(reduction="mean")

    def process_one_epoch(model: nn.Module, db: DataBase, optim=None):
        err_count = 0
        total_loss = 0
        for data, label in db.get_input_tensor(epoch=1, batchsize=1):
            atom_features, dist_mat, adjacent_mat, target = process_input(
                data=data, label=label
            )

            try:
                feat_mat = torch.from_numpy(atom_features).to(device)
                etype_mat = torch.from_numpy(dist_mat).to(device)
                adj_mat = torch.from_numpy(adjacent_mat).to(device)
                target = torch.from_numpy(target).to(device)

                if optim:
                    optim.zero_grad()
                    outputs = model(feat_mat=feat_mat, etype_mat=etype_mat, adj_mat=adj_mat)
                    prediction = torch.sum(outputs, -1)
                    loss = loss_func(prediction, target)
                    loss.backward()
                    optim.step()
                else:
                    with torch.no_grad():
                        outputs = model(feat_mat=feat_mat, etype_mat=etype_mat, adj_mat=adj_mat)
                        prediction = torch.sum(outputs, -1)
                        loss = loss_func(prediction, target)
            except RuntimeError:
                err_count += 1
                print(err_count)
                continue

            total_loss += loss.item() * target.shape[0]

        return total_loss, err_count

    for epoch in range(INIT_EPOCH, INIT_EPOCH + TOTAL_EPOCH):
        print("epoch: ", epoch)
        model.train(mode=True)
        train_loss, err = process_one_epoch(
            model=model, db=train_db, optim=optim
        )
        msg = "  -- train loss: {:.8f}; err count: {}"
        print(msg.format(train_loss/train_count, err))

        model.eval()
        vali_loss, err = process_one_epoch(
            model=model, db=vali_db, optim=None
        )
        msg = "  -- validation loss: {:.8f}; err count: {}"
        print(msg.format(vali_loss/vali_count, err))

        torch.save(
            model.state_dict(),
            MODEL_FILE.format(epoch, train_loss/train_count, vali_loss/vali_count)
        )
