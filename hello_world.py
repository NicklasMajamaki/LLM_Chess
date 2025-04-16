import os
import argparse
import subprocess

def main():
    parser = argparse.ArgumentParser(description="Run vLLM evaluation.")
    parser.add_argument("--data_dir", type=str, default="./data", help="Path to the data directory")
    args = parser.parse_args()

    print(f"Data directory: {args.data_dir}")

    os.makedirs(f"{args.data_dir}/saved_data", exist_ok=True)
    with open(f"{args.data_dir}/saved_data/hello.txt", "w") as f:
        f.write("hello, world")

    cmd = f"aws s3 cp {args.data_dir}/saved_data s3://llm_chess/saved_data --recursive"
    subprocess.run(cmd.split())


if __name__ == "__main__":
    main()