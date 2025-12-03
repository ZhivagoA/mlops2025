import json
import sys

import yaml


def load_params():
    with open("params.yaml", "r") as f:
        return yaml.safe_load(f)


def validate_model():
    params = load_params()
    accuracy_min = params["accuracy_min"]

    try:
        with open("metrics/metrics.json", "r") as f:
            metrics = json.load(f)
    except FileNotFoundError:
        print("Error: metrics/metrics.json not found")
        sys.exit(1)

    accuracy = metrics.get("accuracy")

    if accuracy is None:
        print("Error: accuracy not found in metrics")
        sys.exit(1)

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Accuracy threshold: {accuracy_min:.4f}")

    if accuracy < accuracy_min:
        print(
            f"Validation failed: accuracy {accuracy:.4f} is below threshold {accuracy_min:.4f}"
        )
        sys.exit(1)
    else:
        print("Validation passed!")
        sys.exit(0)


if __name__ == "__main__":
    validate_model()
