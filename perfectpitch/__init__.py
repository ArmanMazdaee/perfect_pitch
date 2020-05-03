import argparse

from perfectpitch.dataset.maps import convert_maps
from perfectpitch.dataset.maestro import convert_maestro
from perfectpitch.acoustic.train import train_acoustic


def main():
    parser = argparse.ArgumentParser(
        description="Automatic Music Transcription using deeplearning"
    )
    subparsers = parser.add_subparsers(title="command")
    subparsers.required = True

    convert_maps_parser = subparsers.add_parser(
        "convert-maps",
        help="Convert the MAPS dataset for the perfectpitch",
        description=(
            "Convert the MAPS datasets to a format which is useable by perfectpitch"
        ),
    )
    convert_maps_parser.set_defaults(func=convert_maps)
    convert_maps_parser.add_argument(
        "--input",
        "-i",
        required=True,
        help="Path of the extracted MAPS dataset",
        dest="input_path",
    )
    convert_maps_parser.add_argument(
        "--output",
        "-o",
        required=True,
        help="Path of the converted dataset",
        dest="output_path",
    )

    convert_maestro_parser = subparsers.add_parser(
        "convert-maestro",
        help="Convert the MAESTRO dataset for the perfectpitch",
        description=(
            "Convert the MAESTRO datasets to a format which is useable by perfectpitch"
        ),
    )
    convert_maestro_parser.set_defaults(func=convert_maestro)
    convert_maestro_parser.add_argument(
        "--input",
        "-i",
        required=True,
        help="Path of the extracted MAESTRO dataset",
        dest="input_path",
    )
    convert_maestro_parser.add_argument(
        "--output",
        "-o",
        required=True,
        help="Path of the converted dataset",
        dest="output_path",
    )

    acoustic_train_parser = subparsers.add_parser(
        "train-acoustic",
        help="Train the acoustic model",
        description="Train the acoustic model",
    )
    acoustic_train_parser.set_defaults(func=train_acoustic)
    acoustic_train_parser.add_argument(
        "--train-dataset",
        "-t",
        required=True,
        help="Path of the train dataset",
        dest="train_dataset_path",
    )
    acoustic_train_parser.add_argument(
        "--validation-dataset",
        "-v",
        required=True,
        help="Path of the validation dataset",
        dest="validation_dataset_path",
    )
    acoustic_train_parser.add_argument(
        "--model-dir",
        "-m",
        required=True,
        help="Directory to use save the logs and weights",
        dest="model_dir",
    )
    acoustic_train_parser.add_argument(
        "--device",
        "-d",
        default="cpu",
        help="device to use for training the model",
        dest="device",
    )

    args = parser.parse_args()
    func = args.func
    kwargs = vars(args)
    del kwargs["func"]
    func(**kwargs)


if __name__ == "__main__":
    main()
