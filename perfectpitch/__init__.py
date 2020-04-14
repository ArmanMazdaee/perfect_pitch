import argparse


def main():
    parser = argparse.ArgumentParser(
        description="Automatic Music Transcription using deeplearning"
    )
    subparsers = parser.add_subparsers(title="command")
    subparsers.required = True

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
