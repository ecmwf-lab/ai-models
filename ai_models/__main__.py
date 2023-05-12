# (C) Copyright 2023 European Centre for Medium-Range Weather Forecasts.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import argparse
import logging
import os
import shlex
import subprocess
import sys

from .model import available_models, load_model

LOG = logging.getLogger(__name__)


def check_gpus():
    try:
        n = 0
        for line in subprocess.check_output(["nvidia-smi", "-L"], text=True).split(
            "\n"
        ):
            if line.startswith("GPU"):
                n += 1
        return n
    except subprocess.CalledProcessError:
        return 0


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--models",
        action="store_true",
        help="List models and exit",
    )

    parser.add_argument(
        "--debug",
        action="store_true",
        help="Turn on debug",
    )

    parser.add_argument(
        "--input",
        default="mars",
        help="Source to use",
    )

    parser.add_argument(
        "--file",
        help="Source to use if source=file",
    )

    parser.add_argument(
        "--output",
        default="file",
        help="Source to use",
    )

    parser.add_argument(
        "--date",
        default="-1",
        help="For which analysis date to start the inference (default: yesterday)",
    )

    parser.add_argument(
        "--time",
        type=int,
        default=12,
        help="For which analysis time to start the inference (default: 12)",
    )

    parser.add_argument(
        "--assets",
        default=os.environ.get("AI_MODELS_ASSETS", "."),
        help="Path to directory containing the weights and other assets",
    )

    parser.add_argument(
        "--assets-sub-directory",
        help="Load assets from a subdirectory of --assets based on the name of the model.",
        action="store_true",
    )

    parser.add_argument(
        "--download-assets",
        help="Download assets if they do not exists.",
        action="store_true",
    )

    parser.add_argument(
        "--path",
        default="out.grib",
        help="Path where to write the output of the model",
    )

    parser.add_argument(
        "--num-threads",
        type=int,
        default=1,
        help="Number of threads. Only relevant for some models.",
    )

    parser.add_argument(
        "--lead-time",
        type=int,
        default=240,
        help="Length of forecast in hours.",
    )

    parser.add_argument(
        "--model-version",
        default="latest",
        help="Model version",
    )

    if "--models" not in sys.argv:
        parser.add_argument(
            "model",
            metavar="MODEL",
            choices=available_models(),
            help="The model to run",
        )

    args = parser.parse_args()

    if args.models:
        for p in sorted(available_models()):
            print(p)
        sys.exit(0)

    if args.assets_sub_directory:
        args.assets = os.path.join(args.assets, args.model)

    logging.basicConfig(
        level="DEBUG" if args.debug else "INFO",
        format="%(asctime)s %(levelname)s %(message)s",
    )

    # if not check_gpus():
    #     LOG.error("ai-models must be run on a node with GPUs")
    #     sys.exit(1)

    try:
        model = load_model(args.model, **vars(args))
        model.run()
    except FileNotFoundError as e:
        LOG.exception(e)
        LOG.error(
            "It is possible that some files requited by %s are missing.", args.model
        )
        LOG.error("Rerun the command as:")
        LOG.error(
            "   %s", shlex.join([sys.argv[0], "--download-assets"] + sys.argv[1:])
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
