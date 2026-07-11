import argparse
import subprocess

def run(cmd):
    print("\nRunning:", " ".join(cmd))
    subprocess.run(cmd, check=True)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", default="0012203205")
    args = parser.parse_args()

    case_id = args.case
    print(f"Start pipeline for case: {case_id}")

    # Step 1: train
    run(["python", "train_puma.py"])

    # Step 2: predict / evaluate
    run(["python", "eval_puma.py"])

    print(f"Pipeline finished for case: {case_id}")

if __name__ == "__main__":
    main()