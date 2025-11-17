import os
import subprocess

os.makedirs("./huggingface_data", exist_ok=True)
os.environ["HF_HOME"] = "./huggingface_data"

#
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
#

model = "Qwen/Qwen3-Embedding-8B"
# model = "Qwen/Qwen3-Embedding-4B"

def main():
    try:
        subprocess.run([
            "vllm", "serve", model,
            "--gpu-memory-utilization", "0.4",
            # "--cpu-offload-gb", "64", 
            "--swap-space", "64",
            "--task", "embed",
            "--max-model-len", "2048",
            "--port", "9804"
        ])
    except KeyboardInterrupt:
        print("\n종료되었습니다.")

if __name__ == "__main__":
    main()