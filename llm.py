import os
import subprocess

os.makedirs("./huggingface_data", exist_ok=True)
os.environ["HF_HOME"] = "./huggingface_data"

#
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
#

# ---
model = "Qwen/Qwen3-30B-A3B-Thinking-2507-FP8"

def main():
    try:
        subprocess.run([
            "vllm", "serve", model,
            "--reasoning-parser", "deepseek_r1",
            "--gpu-memory-utilization", "0.9",
            "--swap-space", "64",
            "--max-model-len", "8192",
            "--port", "9805"
        ])
    except KeyboardInterrupt:
        print("\n종료되었습니다.")

if __name__ == "__main__":
    main()

# ---
# model = "Qwen/Qwen3-8B"

# def main():
#     try:
#         subprocess.run([
#             "vllm", "serve", model,
#             "--reasoning-parser", "deepseek_r1",
#             "--gpu-memory-utilization", "0.4",
#             # "--cpu-offload-gb", "64",
#             "--swap-space", "64",
#             "--max-model-len", "8192",
#             "--port", "9805"
#         ])
#     except KeyboardInterrupt:
#         print("\n종료되었습니다.")

# if __name__ == "__main__":
#     main()

# ---
# model = "Qwen/Qwen3-30B-A3B-Instruct-2507"

# def main():
#     try:
#         subprocess.run([
#             "vllm", "serve", model,
#             "--gpu-memory-utilization", "0.9",
#             "--cpu-offload-gb", "128", 
#             "--swap-space", "64",
#             "--max-model-len", "8192",
#             "--port", "9805"
#         ])
#     except KeyboardInterrupt:
#         print("\n종료되었습니다.")

# if __name__ == "__main__":
#     main()

# ---
# model = "Qwen/Qwen3-30B-A3B-Instruct-2507-FP8"

# def main():
#     try:
#         subprocess.run([
#             "vllm", "serve", model,
#             # "--gpu-memory-utilization", "0.9",
#             # "--cpu-offload-gb", "128", 
#             "--swap-space", "64",
#             "--max-model-len", "8192",
#             "--port", "9805"
#         ])
#     except KeyboardInterrupt:
#         print("\n종료되었습니다.")

# if __name__ == "__main__":
#     main()

# ---
# model = "Qwen/Qwen3-30B-A3B-GPTQ-Int4"

# def main():
#     try:
#         subprocess.run([
#             "vllm", "serve", model,
#             "--gpu-memory-utilization", "0.4",
#             # "--cpu-offload-gb", "128", 
#             "--swap-space", "64",
#             "--reasoning-parser", "deepseek_r1",
#             "--max-model-len", "8192",
#             "--port", "9805"
#         ])
#     except KeyboardInterrupt:
#         print("\n종료되었습니다.")

# if __name__ == "__main__":
#     main()

# ---
# model = "Qwen/Qwen3-30B-A3B-Instruct-2507-FP8"

# def main():
#     try:
#         subprocess.run([
#             "vllm", "serve", model,
#             "--gpu-memory-utilization", "0.9",
#             "--swap-space", "64",
#             "--max-model-len", "8192",
#             "--port", "9805"
#         ])
#     except KeyboardInterrupt:
#         print("\n종료되었습니다.")

# if __name__ == "__main__":
#     main()