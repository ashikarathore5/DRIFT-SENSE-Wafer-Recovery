import json
import os
from localize import evaluate


def run():
    os.makedirs("results", exist_ok=True)
    evaluate(data_dir="dataset", split="val", out_dir="results")


if __name__ == "__main__":
    run()