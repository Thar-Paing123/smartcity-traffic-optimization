"""Promotes a trained SavedModel into the versioned layout TensorFlow Serving expects.

TF Serving auto-discovers the highest-numbered version subdirectory under
`serving_dir/<model_name>/`, so each export bumps the version rather than
overwriting the previous one. That's what lets a retrain (see
monitor/retrain.py) roll out to serving without downtime.
"""

import argparse
import os
import shutil
import time

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_MODEL_PATH = os.path.join(ROOT_DIR, "artifacts", "model", "saved_model")
DEFAULT_SERVING_DIR = os.path.join(ROOT_DIR, "artifacts", "serving")


def export(model_path: str, serving_dir: str, model_name: str = "traffic_forecaster") -> str:
    version = int(time.time())
    export_path = os.path.join(serving_dir, model_name, str(version))
    shutil.copytree(model_path, export_path)
    print(f"Exported model for serving at {export_path}")
    return export_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--serving-dir", default=DEFAULT_SERVING_DIR)
    parser.add_argument("--model-name", default="traffic_forecaster")
    args = parser.parse_args()
    export(args.model_path, args.serving_dir, args.model_name)
