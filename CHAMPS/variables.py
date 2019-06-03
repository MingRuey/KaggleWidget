from variables import train_csv, test_csv, train_features_v0, test_features_v0  # noqa: E402
from variables import train_features_v0, test_features_v0  # noqa: E402

_base = "/rawdata/CHAMPS_Molecular/"

train_csv = _base + "train.csv"
test_csv = _base + "test.csv"

train_features_v0 = "/archive/CHAMPS/databaseV00/train"
test_features_v0 = "/archive/CHAMPS/databaseV00/test"

dipole_moments = _base + "dipole_moments.csv"
magnetic_shielding_tensors = _base + "magnetic_shielding_tensors.csv"
mulliken_charges = _base + "mulliken_charges.csv"
potential_energy = _base + "potential_energy.csv"
sample_submission = _base + "sample_submission.csv"
scalar_coupling_contributions = _base + "scalar_coupling_contributions.csv"
