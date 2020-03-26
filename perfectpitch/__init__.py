import argparse

from perfectpitch.dataset.maps import convert_maps
from perfectpitch.dataset.maestro import convert_maestro


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
        "acoustic-train",
        help="Train the acoustic model",
        description="Train the acoustic model",
    )
    acoustic_train_parser.set_defaults(
        func=lambda dataset_path, use_gpu: print(
            "acoustic-trian", dataset_path, use_gpu
        )
    )
    acoustic_train_parser.add_argument(
        "--dataset-path",
        "-d",
        required=True,
        help="Path of the dataset",
        dest="dataset_path",
    )
    acoustic_train_parser.add_argument(
        "--gpu",
        "-g",
        action="store_true",
        help="Use gpu for training the model",
        dest="use_gpu",
    )

    args = parser.parse_args()
    func = args.func
    kwargs = vars(args)
    del kwargs["func"]
    func(**kwargs)


if __name__ == "__main__":
    main()
