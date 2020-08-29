import argparse

import perfectpitch.convert_dataset.convert_maps
import perfectpitch.convert_dataset.convert_maestro
import perfectpitch.music_model.train


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
    convert_maps_parser.set_defaults(
        func=perfectpitch.convert_dataset.convert_maps.convert_maps
    )
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
    convert_maestro_parser.set_defaults(
        func=perfectpitch.convert_dataset.convert_maestro.convert_maestro
    )
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

    train_music_model_parser = subparsers.add_parser(
        "train-music-model",
        help="Train the music model",
        description="Train the music model",
    )
    train_music_model_parser.set_defaults(func=perfectpitch.music_model.train.train)
    train_music_model_parser.add_argument(
        "--train-dataset",
        "-t",
        required=True,
        help="Path of the train dataset",
        dest="train_dataset_path",
    )
    train_music_model_parser.add_argument(
        "--validation-dataset",
        "-v",
        required=True,
        help="Path of the validation dataset",
        dest="validation_dataset_path",
    )
    train_music_model_parser.add_argument(
        "--model-dir",
        "-m",
        required=True,
        help="Directory to use save the logs and weights",
        dest="model_dir",
    )

    args = parser.parse_args()
    func = args.func
    kwargs = vars(args)
    del kwargs["func"]
    func(**kwargs)


if __name__ == "__main__":
    main()
