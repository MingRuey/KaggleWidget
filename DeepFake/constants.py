import pathlib
import json

_train_videos = "/rawdata2/DeepFake/DetectionCompetition/train_videos"
_test_videos = "/rawdata2/DeepFake/DetectionCompetition/test_videos"


def get_train_videos():
    """Get absolute file path of train videos

    Returns:
        list of path like object: the file path of videos
    """
    root = pathlib.Path(_train_videos)
    dir_prefix = "dfdc_train_part_{}"

    output = []
    for part in range(50):
        folder_name = dir_prefix.format(part)
        videos = list(root.joinpath(folder_name).glob("*.mp4"))
        output.extend(videos)
    return output


def get_test_videos():
    """Get absolute file path of test videos

    Returns:
        same as get_train_videos
    """
    root = pathlib.Path(_test_videos)
    output = list(root.glob("*.mp4"))
    return output


def get_metadata():
    """Get metadata of train videos

    Returns:
        a dictonry of {string: bool},
        where keys are video file name,
        and values are fake video or not {fake = True, real = False}
    """
    root = pathlib.Path(_train_videos)
    dir_prefix = "dfdc_train_part_{}"

    output = {}
    for part in range(50):
        folder_name = dir_prefix.format(part)
        metadata = root.joinpath(folder_name).joinpath("metadata.json")
        with open(str(metadata), "r") as f:
            metadata = json.load(f)
            assert all(
                value["label"] in ["REAL", "FAKE"]
                for value in metadata.values()
            )
            output.update(
                {k: v["label"] == "FAKE" for k, v in metadata.items()}
            )
    return output


train_videos = get_train_videos()
test_videos = get_test_videos()
metadata = get_metadata()
submission = "/rawdata2/DeepFake/DetectionCompetition/sample_submission.csv"


def _check_train_videos():
    """Sanity check"""
    # check non duplicate videos separately for each part
    assert len(train_videos) == len(set(train_videos))
    assert len(test_videos) == len(set(test_videos))
    assert len(metadata) == len(set(metadata.keys()))
    videos = set(path.name for path in train_videos)
    labels = set(metadata.keys())
    assert videos <= labels


if __name__ == "__main__":
    _check_train_videos()
