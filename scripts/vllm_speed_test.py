"""A/B speed test: llm.chat() vs llm.generate() at 1024x1024."""
import base64
import glob
import os
import time
from io import BytesIO
from PIL import Image


MODEL = "Qwen/Qwen2.5-VL-7B-Instruct"
BASE = "/mnt/c/Users/frede/Desktop/kandidat/speciale/anomverse_extension/datasets/realiad_1024"
N_IMAGES = 20
BATCH_SIZE = 8


def pil_to_data_uri(img):
    buf = BytesIO()
    img.save(buf, format="JPEG", quality=85)
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{b64}"


def main():
    from vllm import LLM, SamplingParams

    jpg_files = glob.glob(os.path.join(BASE, "*/NG/**/*.jpg"), recursive=True)[:N_IMAGES]
    print(f"Using {len(jpg_files)} images")

    images = [Image.open(f).convert("RGB") for f in jpg_files]
    print(f"Image sizes: {[img.size for img in images[:3]]}")
    prompt = "Describe any defects visible in this image."

    print("Loading model...")
    llm = LLM(model=MODEL, max_model_len=4096, max_num_seqs=8,
              gpu_memory_utilization=0.85, trust_remote_code=True)
    sampling_params = SamplingParams(temperature=0.7, max_tokens=77)

    # --- Test A: llm.chat() with base64 JPEG ---
    print(f"\n=== Test A: llm.chat() with base64 JPEG, batch={BATCH_SIZE} ===")
    t0 = time.time()
    for batch_start in range(0, len(images), BATCH_SIZE):
        batch = images[batch_start:batch_start + BATCH_SIZE]
        conversations = []
        for img in batch:
            conversations.append([
                {"role": "user", "content": [
                    {"type": "image_url", "image_url": {"url": pil_to_data_uri(img)}},
                    {"type": "text", "text": prompt},
                ]},
            ])
        outputs = llm.chat(conversations, sampling_params=sampling_params)
        elapsed = time.time() - t0
        print(f"  batch {batch_start//BATCH_SIZE + 1}: {len(outputs)} outputs, "
              f"tokens: {[len(o.outputs[0].token_ids) for o in outputs]}, "
              f"elapsed: {elapsed:.1f}s")
    t_chat = time.time() - t0
    print(f"llm.chat() total: {t_chat:.1f}s = {len(images)/t_chat:.2f} img/s")

    # --- Test B: llm.generate() with direct PIL ---
    print(f"\n=== Test B: llm.generate() with direct PIL, batch={BATCH_SIZE} ===")
    t0 = time.time()
    for batch_start in range(0, len(images), BATCH_SIZE):
        batch = images[batch_start:batch_start + BATCH_SIZE]
        inputs = []
        for img in batch:
            text = (
                "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
                f"<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>{prompt}<|im_end|>\n"
                "<|im_start|>assistant\n"
            )
            inputs.append({"prompt": text, "multi_modal_data": {"image": img}})
        outputs = llm.generate(inputs, sampling_params=sampling_params)
        elapsed = time.time() - t0
        print(f"  batch {batch_start//BATCH_SIZE + 1}: {len(outputs)} outputs, "
              f"tokens: {[len(o.outputs[0].token_ids) for o in outputs]}, "
              f"elapsed: {elapsed:.1f}s")
    t_gen = time.time() - t0
    print(f"llm.generate() total: {t_gen:.1f}s = {len(images)/t_gen:.2f} img/s")

    print(f"\n=== Summary ===")
    print(f"chat():     {t_chat:.1f}s = {len(images)/t_chat:.2f} img/s")
    print(f"generate(): {t_gen:.1f}s = {len(images)/t_gen:.2f} img/s")
    print(f"Ratio: chat is {t_gen/t_chat:.1f}x {'faster' if t_chat < t_gen else 'slower'}")


if __name__ == "__main__":
    main()
