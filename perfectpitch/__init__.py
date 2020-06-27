import argparse

from perfectpitch.dataset.maps import convert_maps
from perfectpitch.dataset.maestro import convert_maestro
from perfectpitch.onsetsdetector.train import train_onsets_detector
from perfectpitch.transcriber.evaluate import evaluate_transcriber


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

    train_onsets_detector_parser = subparsers.add_parser(
        "train-onsets-detector",
        help="Train the onsets detector",
        description="Train the onsets detector",
    )
    train_onsets_detector_parser.set_defaults(func=train_onsets_detector)
    train_onsets_detector_parser.add_argument(
        "--train-dataset",
        "-t",
        required=True,
        help="Path of the train dataset",
        dest="train_dataset_path",
    )
    train_onsets_detector_parser.add_argument(
        "--validation-dataset",
        "-v",
        required=True,
        help="Path of the validation dataset",
        dest="validation_dataset_path",
    )
    train_onsets_detector_parser.add_argument(
        "--model-dir",
        "-m",
        required=True,
        help="Directory to use save the logs and weights",
        dest="model_dir",
    )
    train_onsets_detector_parser.add_argument(
        "--device",
        "-D",
        default="cpu",
        help="device to use for training the model",
        dest="device",
    )

    evaluate_transcriber_parser = subparsers.add_parser(
        "evaluate-transcriber",
        help="Evaluate the transcriber system",
        description="Evaluate the transcriber system",
    )
    evaluate_transcriber_parser.set_defaults(func=evaluate_transcriber)
    evaluate_transcriber_parser.add_argument(
        "--dataset",
        "-d",
        required=True,
        help="Path of the dataset",
        dest="dataset_path",
    )
    evaluate_transcriber_parser.add_argument(
        "--onsets-detector",
        "-o",
        required=True,
        help="Path of the onsets deterctor weights",
        dest="onsets_detector_path",
    )
    evaluate_transcriber_parser.add_argument(
        "--device",
        "-D",
        default="cpu",
        help="device to use for the inference",
        dest="device",
    )

    args = parser.parse_args()
    func = args.func
    kwargs = vars(args)
    del kwargs["func"]
    func(**kwargs)


if __name__ == "__main__":
    main()
