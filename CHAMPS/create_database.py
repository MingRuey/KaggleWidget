import os
import sys
import pathlib
import logging
from collections import defaultdict
from typing import NamedTuple
from shutil import copyfile

file = os.path.basename(__file__)
file = pathlib.Path(file).stem
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s-%(name)s-%(message).1000s ',
    handlers=[
        logging.FileHandler("{}.log".format(file)),
        logging.StreamHandler(sys.stdout)
        ]
    )

import tensorflow as tf  # noqa: E402
from MLBOX.Database.formats import _tffeature_bytes, _tffeature_float, _tffeature_int64   # noqa: E402
from MLBOX.Database.formats import DataFormat  # noqa: E402
from MLBOX.Database.dataset import DataBase  # noqa: E402
from variables import train_csv, test_csv, train_features_v0, test_features_v0  # noqa: E402


Bond = NamedTuple("Bond", [
        ("type", str),
        ("index1", int),
        ("index2", int),
        ("coupling", float)
    ])

Atom = NamedTuple("Atom", [
        ("type", str),
        ("index", int),
        ("x", float),
        ("y", float),
        ("z", float)
    ])


def parse_bonds_csv(file):

    bonds = defaultdict(list)

    with open(file, "r") as f:
        f.readline()  # the header lines
        for line in f.readlines():
            _, molecule, index1, index2, name, *coupling = line.split(",")
            coupling = "nan" if not coupling else coupling[0]  # for test.csv
            bonds[molecule].append(Bond(
                type=name,
                index1=int(index1), index2=int(index2),
                coupling=float(coupling)
            ))

    return bonds


class FeaturesV0(DataFormat):

    features = {
        'molecule': tf.io.FixedLenFeature([], tf.string),
        'bonds':
            tf.io.FixedLenSequenceFeature([], tf.string, allow_missing=True),
        'bonds/index1':
            tf.io.FixedLenSequenceFeature([], tf.int64, allow_missing=True),
        'bonds/index2':
            tf.io.FixedLenSequenceFeature([], tf.int64, allow_missing=True),
        'bonds/coupling':
            tf.io.FixedLenSequenceFeature([], tf.float32, allow_missing=True),
        'atoms':
            tf.io.FixedLenSequenceFeature([], tf.string, allow_missing=True),
        'atoms/index':
            tf.io.FixedLenSequenceFeature([], tf.int64, allow_missing=True),
        'atoms/x':
            tf.io.FixedLenSequenceFeature([], tf.float32, allow_missing=True),
        'atoms/y':
            tf.io.FixedLenSequenceFeature([], tf.float32, allow_missing=True),
        'atoms/z':
            tf.io.FixedLenSequenceFeature([], tf.float32, allow_missing=True),
    }
    valid_extensions = {'.xyz'}

    def __init__(self, molecule_bonds_map=None):
        """
        Args:
            molecule_to_bonds_map:
                a dict that maps molecule id into a Bond or a list of Bond,
                which specify the bonds of the molecule
                If set to None,
                no coupling label is included in the bonds of tf.train.Example
        """
        self._molecule_bonds_map = \
            None if not molecule_bonds_map else molecule_bonds_map.copy()

    @staticmethod
    def _xyz_file_reader(xyz_file_path):
        """Read an xyz file and return its atoms

        Args:
            xyz_file_path: a string specify the path of xyz_file

        Returns:
            a list of Atom
        """
        atoms = []
        with open(str(xyz_file_path), "r") as f:
            count = int(f.readline())
            _ = f.readline()  # there is a blank line
            for index in range(count):
                name, x, y, z = f.readline().split()
                atom = Atom(
                    type=name, index=index,
                    x=float(x), y=float(y), z=float(z)
                    )
                atoms.append(atom)
        return atoms

    def to_tfexample(self, xyz_file_path):
        """Load an xyz file and return a tf.train.Example object

        Args:
            xyz_file_path: a string specify the path of xyz_file

        Return:
            a tf.train.Example object
        """

        path = pathlib.Path(xyz_file_path)
        if not path.is_file():
            raise OSError("Invalid file path")

        molecule_id = path.stem
        atoms = FeaturesV0._xyz_file_reader(xyz_file_path)
        bonds = self._molecule_bonds_map[molecule_id]

        fields = {
            'molecule': _tffeature_bytes(molecule_id),
            'bonds':
                _tffeature_bytes([bytes(bond.type, 'utf8') for bond in bonds]),
            'bonds/index1':
                _tffeature_int64([bond.index1 for bond in bonds]),
            'bonds/index2':
                _tffeature_int64([bond.index2 for bond in bonds]),
            'atoms':
                _tffeature_bytes([bytes(atom.type, 'utf8') for atom in atoms]),
            'atoms/index':
                _tffeature_int64([atom.index for atom in atoms]),
            'atoms/x':
                _tffeature_float([atom.x for atom in atoms]),
            'atoms/y':
                _tffeature_float([atom.y for atom in atoms]),
            'atoms/z':
                _tffeature_float([atom.z for atom in atoms])
        }

        if self._molecule_bonds_map:
            fields['bonds/coupling'] = \
                _tffeature_float([bond.coupling for bond in bonds])

        return tf.train.Example(features=tf.train.Features(feature=fields))

    @staticmethod
    def get_parser(**kwargs):

        def parse_tfexample(example):
            parsed_feature = tf.io.parse_single_example(
                example,
                features=FeaturesV0.features
            )

            dataid = parsed_feature["molecule"]
            label = parsed_feature['bonds/coupling']
            return parsed_feature, dataid, label

        return parse_tfexample


if __name__ == "__main__":

    train_bonds = parse_bonds_csv(train_csv)
    test_bonds = parse_bonds_csv(test_csv)

    print("train data count: ", len(train_bonds))
    print("test data count: ", len(test_bonds))

    # copy .xyz file
    source = pathlib.Path("/rawdata/CHAMPS_Molecular/structures")
    train_target = pathlib.Path("/archive/CHAMPS/xyzs_train")
    test_target = pathlib.Path("/archive/CHAMPS/xyzs_test")

    # for xyz_id in train_bonds:
    #     file = xyz_id + ".xyz"
    #     copyfile(
    #         str(source.joinpath(file)),
    #         str(train_target.joinpath(file))
    #     )

    # for xyz_id in test_bonds:
    #     file = xyz_id + ".xyz"
    #     copyfile(
    #         str(source.joinpath(file)),
    #         str(test_target.joinpath(file))
    #     )

    dataformat = FeaturesV0(test_bonds)
    dataset = DataBase(formats=dataformat)
    dataset.build_database(
        input_dir=test_target,
        output_dir=test_features_v0
    )

    dataformat = FeaturesV0(train_bonds)
    dataset = DataBase(formats=dataformat)
    dataset.build_database(
        input_dir=train_target,
        output_dir=train_features_v0
    )