import argparse
import importlib


def run():
    parser = argparse.ArgumentParser(
        description="Automatic Music Transcription using deeplearning"
    )
    subparsers = parser.add_subparsers(dest="command", title="command")
    subparsers.required = True

    prepare_dataset_parser = subparsers.add_parser(
        "prepare-dataset",
        help="Prepare the dataset for the model training",
        description=(
            "Preparing the datasets using "
            '"Onsets and Frames: Dual-Objective Piano Transcription" '
            "from magenta datasets as source"
        ),
    )
    prepare_dataset_parser.add_argument(
        "--input",
        "-i",
        required=True,
        help="A glob of the source TFRecords",
        dest="input_pattern",
    )
    prepare_dataset_parser.add_argument(
        "--output",
        "-o",
        required=True,
        help="Path of the generated dataset",
        dest="output_path",
    )

    extract_dataset_parser = subparsers.add_parser(
        "extract-dataset",
        help="Extract a dataset",
        description="Extract a dataset as a group of wav and midi files",
    )
    extract_dataset_parser.add_argument(
        "--input",
        "--i",
        required=True,
        help="Path of the source dataset",
        dest="dataset_path",
    )
    extract_dataset_parser.add_argument(
        "--output",
        "-o",
        required=True,
        help="Path of directory to extract dataset to",
        dest="output_path",
    )

    train_acoustic_parser = subparsers.add_parser(
        "train-acoustic",
        help="Train the acoustic model",
        description="Train the acoustic model",
    )
    train_acoustic_parser.add_argument(
        "--train-dataset",
        "-t",
        required=True,
        help="Path of the training dataset",
        dest="train_path",
    )
    train_acoustic_parser.add_argument(
        "--validation-dataset",
        "-v",
        required=True,
        help="Path of the validation dataset",
        dest="validation_path",
    )

    args = parser.parse_args()
    command = args.command.replace("-", "_")
    module = importlib.import_module("perfectpitch.cmd." + command)
    func = getattr(module, command)
    kwargs = vars(args)
    del kwargs["command"]
    func(**kwargs)
