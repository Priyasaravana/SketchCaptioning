import argparse


def argument_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # ************************************************************
    # Datasets (general)
    # ************************************************************
    parser.add_argument(
        "--DatasetName", type=str, default="./data", help="root path to data directory"
    )

    parser.add_argument(
        "--JsonPath", type=str, default="./data", help="json path to data directory"
    )
    parser.add_argument(
        "--ImagePath", type=str, default="./data", help="image path to data directory"
    )
    parser.add_argument(
        "--ImagePath2", type=str, default="./data", help="image path 2 to data directory"
    )

    parser.add_argument(
        "--OutputPath", type=str, default="./data", help="output path to data directory"
    )
    parser.add_argument(
        "--CaptionPath", type=str, default="./data", help="caption root path to data directory"
    )
    parser.add_argument(
        "--cocoBasePath", type=str, default="./data", help="coco base path to data directory"
    )
    parser.add_argument(
        "--cocoAnnotationPath", type=str, default="./data", help="coco base path to data directory"
    )
    parser.add_argument(
        "--splitPercentage", type=str, default="70", help="percentage of split"
    )
    parser.add_argument(
        "--Experiment", type=str, default="exp2", help="percentage of split"
    )


    return parser

