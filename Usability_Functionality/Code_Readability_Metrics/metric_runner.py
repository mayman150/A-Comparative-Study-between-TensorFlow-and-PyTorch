import os
import argparse
import subprocess
import sys
from pathlib import Path

list_of_models = [
    "CV/Inception.py",
    "CV/ResNet.py",
    "CV/VGG.py",
    "NLP/BERT/",
    "NLP/GNMT/",
    "SR/DeepSpeech2/",
    "SR/tacotron2/"
]

metric_file_translate = {
    "tc": "textual_cohesion_calculator.py",
    "itid": "itid_calculator.py"
}


def get_script_path():
    return Path(os.path.dirname(os.path.realpath(sys.argv[0])))


def get_output_file_name(output_file_prefix: str, model_path: Path, output_dir: Path):
    model_name = model_path.stem

    return output_dir / Path(output_file_prefix + model_name + ".csv")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("-m", "--metric_type", choices=["tc", "itid"], required=True, help="Choose the type of metric to run. Can be either 'tc' for Textual Coherance, or 'itid' for Identifier Terms in Dictionary.")
    parser.add_argument("-l", "--library_type", choices=["torch", "tf"], required=True, help="Choose the library. Can be either 'tf' for TensorFlow, or 'torch' for PyTorch.")
    parser.add_argument("-o", "--output_dir", type=Path, help="output directory", default=Path("./output/"))
    parser.add_argument("model_root_path", type=Path, help="root path containing the files of all the models")

    args = parser.parse_args()

    if not args.model_root_path.exists():
        parser.error("Unable to locate folder: " + str(args.model_root_path))

    output_file_prefix = args.metric_type + '_' + args.library_type + '_'

    metric_calculator_path = get_script_path() / metric_file_translate[args.metric_type]
    
    assert metric_calculator_path.exists()

    for model_path in list_of_models:
        source_model_path = args.model_root_path / Path(model_path)
        output_csv_name = get_output_file_name(output_file_prefix, source_model_path, args.output_dir)
        print("Currently processing model: " + Path(model_path).stem)
        subprocess.call(["/usr/bin/env", "python3", metric_calculator_path, source_model_path, output_csv_name], stdout=sys.stdout)
        print("Finished Model")
        print("==============")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        exit(0)
