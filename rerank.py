import os
import subprocess
import json

os.makedirs("./huggingface_data", exist_ok=True)
os.environ["HF_HOME"] = "./huggingface_data"

#
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
#

model = "Qwen/Qwen3-Reranker-8B"
# model = "Qwen/Qwen3-Reranker-4B"
# model = "Qwen/Qwen3-Reranker-0.6B"

def main():
    try:
        hf_overrides = {
            "architectures": ["Qwen3ForSequenceClassification"],
            "classifier_from_token": ["no", "yes"],
            "is_original_qwen3_reranker": True
        }
        hf_overrides_str = json.dumps(hf_overrides)

        subprocess.run([
            "vllm", "serve", model,
            "--hf_overrides", hf_overrides_str,
            "--gpu-memory-utilization", "0.5",
            # "--cpu-offload-gb", "64", 
            "--swap-space", "64",
            "--max-model-len", "2048",
            "--port", "9806"
        ])
    except KeyboardInterrupt:
        print("\n종료되었습니다.")

if __name__ == "__main__":
    main()