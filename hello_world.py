import os
import argparse
import subprocess
import requests

def main():
    parser = argparse.ArgumentParser(description="Run vLLM evaluation.")
    parser.add_argument("--data_dir", type=str, default="./data", help="Path to the data directory")
    args = parser.parse_args()

    print(f"Data directory: {args.data_dir}")

    os.makedirs(f"{args.data_dir}/saved_data", exist_ok=True)
    file_path = f"{args.data_dir}/saved_data/hello.txt"

    with open(file_path, "w") as f:
        f.write("hello, world\n")

    # OpenAI API call to local vLLM server
    response = requests.post(
        "http://localhost:8000/v1/completions",
        headers={"Content-Type": "application/json"},
        json={
            "model": "meta-llama/Llama-3.2-3B",
            "prompt": "Tell me a whimsical bedtime story.",
            "max_tokens": 2048,
            "temperature": 0.7,
        },
    )

    if response.ok:
        result = response.json()["choices"][0]["text"]
        with open(file_path, "a") as f:
            f.write(f"\nOpenAI Response:\n{result.strip()}\n")
    else:
        print("OpenAI API call failed:", response.text)

    cmd = f"aws s3 cp {args.data_dir}/saved_data s3://llm-chess/saved_data --recursive"
    subprocess.run(cmd.split())

if __name__ == "__main__":
    main()
