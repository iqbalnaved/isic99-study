import os
import re
import json
import argparse
from datetime import datetime

import torch
import pandas as pd
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, AutoModelForImageTextToText, BitsAndBytesConfig
from huggingface_hub import login


# ==========================================================
# Optional: authenticate with Hugging Face (use env var instead of hard-coding token)
# Set HF_TOKEN in your environment, e.g.:
#   export HF_TOKEN="your_token_here"
# or on Windows:
#   set HF_TOKEN=your_token_here
# If you don't need private models, you can comment this out.
hf_token = os.getenv("HF_TOKEN")
if hf_token:
    login(hf_token)


# ==========================================================
def load_model(model_id: str = "google/medgemma-4b-it"):
    """
    Load Med-Gemma with simple VRAM-aware config.
    """
    if torch.cuda.is_available():
        device = "cuda"
        gpu_props = torch.cuda.get_device_properties(0)
        vram_gb = gpu_props.total_memory / 1024 ** 3

        if vram_gb < 12:
            # 4-bit quantization for smaller GPUs
            bnb_cfg = BitsAndBytesConfig(load_in_4bit=True)
            model = AutoModelForImageTextToText.from_pretrained(
                model_id,
                device_map="auto",
                dtype=torch.bfloat16,
                offload_buffers=True,
                quantization_config=bnb_cfg,
            )
        else:
            model = AutoModelForImageTextToText.from_pretrained(
                model_id,
                torch_dtype=torch.bfloat16,
                device_map={"": "cuda"},
            )
    else:
        device = "cpu"
        model = AutoModelForImageTextToText.from_pretrained(
            model_id,
            torch_dtype=torch.float32,
            device_map=None,
        )

    processor = AutoProcessor.from_pretrained(model_id, use_fast=True)
    # Double-check device in case model landed on CPU
    if not any(p.device.type == "cuda" for p in model.parameters()):
        device = "cpu"
    return model, processor, device


# ==========================================================
def build_messages(image: Image.Image, question: str):
    """
    Build chat messages for Med-Gemma with a strong JSON-only instruction.
    """
    system_text = (
        "You are a dermatologist diagnosing dermoscopic images. "
        "You MUST respond with STRICT JSON only, with no extra text, "
        "in the following format:\n\n"
        "{\n"
        '  "diagnosis": "melanoma" or "benign",\n'
        '  "rationale": "your rationale for this diagnosis",\n'
        '  "confidence": number between 0 and 1\n'
        "}\n\n"
        "Do not include any additional keys, comments, markdown, or text."
    )

    messages = [
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": system_text,
                }
            ],
        },
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {
                    "type": "text",
                    "text": (
                        question
                        + " Remember: respond ONLY with valid JSON in the specified format."
                    ),
                },
            ],
        },
    ]
    return messages


# ==========================================================
def analyze_image(
    image_path: str,
    question: str,
    model,
    processor,
    device: str,
    max_new_tokens: int = 250,
) -> str:
    """
    Run Med-Gemma on a single image and return the raw decoded text.
    """
    image = Image.open(image_path).convert("RGB")

    messages = build_messages(image, question)

    prompt = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=False,
    )
    inputs = processor(images=image, text=prompt, return_tensors="pt")

    target_device = next(model.parameters()).device
    for k, v in inputs.items():
        if k == "input_ids":
            inputs[k] = v.to(target_device, dtype=torch.long)
        else:
            inputs[k] = v.to(target_device, dtype=model.dtype)

    input_length = inputs["input_ids"].shape[-1]

    output = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        return_dict_in_generate=True,
        output_scores=False,
        do_sample=True,
        temperature=0.7,  # adjust as needed
        top_p=0.9,        # nucleus sampling
    )

    sequences = output.sequences
    # Grab only generated tokens (beyond the prompt)
    if sequences.ndim == 2:
        generated_ids = sequences[0, input_length:]
    else:
        generated_ids = sequences[input_length:]

    decoded = processor.decode(generated_ids, skip_special_tokens=True).strip()
    return decoded


# ==========================================================
def _strip_code_fences(text: str) -> str:
    """
    Remove surrounding markdown code fences if present.
    """
    text = text.strip()
    if text.startswith("```"):
        # Remove leading ```[lang]? and trailing ```
        text = re.sub(r"^```[a-zA-Z]*\n?", "", text)
        text = re.sub(r"\n?```$", "", text).strip()
    return text


def _extract_json_block(text: str) -> str | None:
    """
    Try to extract the largest JSON-like {...} block from text.
    """
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return text[start : end + 1]
    return None


def _parse_confidence_from_text(text: str) -> float | None:
    """
    Heuristic extraction of a confidence value from free-form text.
    Looks for numbers between 0 and 1, or percentages.
    """
    # Look for numbers like 0.87, 0.5, 1, etc.
    num_match = re.findall(r"\b\d+(?:\.\d+)?\b", text)
    if not num_match:
        return None

    # Try to infer if they are percentages or direct probs
    # Use the first number that makes sense as probability
    for raw in num_match:
        try:
            val = float(raw)
        except ValueError:
            continue

        # If it's > 1 and <= 100, treat as percentage
        if 1 < val <= 100:
            return max(0.0, min(1.0, val / 100.0))
        # If it's 0–1, accept directly
        if 0.0 <= val <= 1.0:
            return val

    return None


def parse_model_response(text: str):
    """
    Ultra-robust JSON parser for Med-Gemma outputs.
    Handles:
        - Code fences (```json ... ```)
        - Escaped quotes inside rationale
        - Partial JSON embedded inside other text
        - Fixing malformed JSON (missing commas, bad escapes)
    """

    raw = text

    # --- 1) Remove code fences ------------------------------------
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```[a-zA-Z]*\n?", "", cleaned)
        cleaned = re.sub(r"\n?```$", "", cleaned).strip()

    # --- 2) Extract largest JSON block -----------------------------
    block = _extract_json_block(cleaned)
    if not block:
        block = _extract_json_block(raw)  # fallback

    if not block:
        # Fallback to keyword heuristics
        lower = raw.lower()
        if "melanoma" in lower and "benign" not in lower:
            return "melanoma", raw, _parse_confidence_from_text(raw)
        if "benign" in lower and "melanoma" not in lower:
            return "benign", raw, _parse_confidence_from_text(raw)
        return "error", raw, None

    # --- 3) Attempt strict JSON load ------------------------------
    try:
        obj = json.loads(block)
    except json.JSONDecodeError:
        # --- 4) Attempt repair -------------------------------------
        repaired = block

        # Fix common issues:
        repaired = repaired.replace('\\"', '"')      # remove unnecessary escapes
        repaired = repaired.replace("“", '"').replace("”", '"')
        repaired = repaired.replace("’", "'")

        # Remove trailing commas inside JSON
        repaired = re.sub(r",(\s*[}\]])", r"\1", repaired)

        try:
            obj = json.loads(repaired)
        except Exception:
            # Final fallback: heuristics only
            lower = raw.lower()
            if "melanoma" in lower and "benign" not in lower:
                return "melanoma", raw, _parse_confidence_from_text(raw)
            if "benign" in lower:
                return "benign", raw, _parse_confidence_from_text(raw)
            return "error", raw, None

    # --- 5) Extract fields safely --------------------------------
    diagnosis = obj.get("diagnosis") or obj.get("Diagnosis")
    rationale = obj.get("rationale") or obj.get("Rationale") or raw
    confidence = obj.get("confidence") or obj.get("Confidence")

    # Normalize diagnosis
    if isinstance(diagnosis, str):
        d = diagnosis.lower().strip()
        if "melanoma" in d:
            diagnosis = "melanoma"
        elif "benign" in d:
            diagnosis = "benign"
        else:
            diagnosis = "error"
    else:
        diagnosis = "error"

    # Normalize confidence
    if isinstance(confidence, (float, int, str)):
        try:
            val = float(confidence)
            if 1 < val <= 100:
                val /= 100.0
            confidence = max(0, min(1, val))
        except:
            confidence = _parse_confidence_from_text(raw)
    else:
        confidence = _parse_confidence_from_text(raw)

    return diagnosis, rationale, confidence


# ==========================================================
def compute_metrics(df: pd.DataFrame):
    """
    Compute accuracy, sensitivity (TPR), specificity (TNR).
    Assumes columns: true_label, prediction
    """
    TP = ((df.true_label == "melanoma") & (df.prediction == "melanoma")).sum()
    TN = ((df.true_label == "benign") & (df.prediction == "benign")).sum()
    FP = ((df.true_label == "benign") & (df.prediction == "melanoma")).sum()
    FN = ((df.true_label == "melanoma") & (df.prediction == "benign")).sum()

    total = len(df)
    accuracy = (TP + TN) / total if total else 0.0
    sensitivity = TP / (TP + FN) if (TP + FN) else 0.0
    specificity = TN / (TN + FP) if (TN + FP) else 0.0

    return accuracy, sensitivity, specificity


# ==========================================================
def process_dataset(root_dir: str, output_dir: str = ".", run_number: int | None = None):
    """
    Walk over mm_class (melanoma) and bn_class (benign) subfolders,
    run Med-Gemma, save per-image JSON + summary metrics CSV.
    """
    os.makedirs(output_dir, exist_ok=True)

    model, processor, device = load_model()
    print(f"Using device: {device}")

    question = "Is this lesion melanoma or benign?"

    rows = []
    classes = {
        "mm_class": "melanoma",
        "bn_class": "benign",
    }

    for folder, mapped_label in classes.items():
        subdir = os.path.join(root_dir, folder)
        if not os.path.isdir(subdir):
            print(f"⚠️ Missing directory: {subdir}")
            continue

        files = [
            f
            for f in os.listdir(subdir)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        ]
        print(f"📂 Class '{folder}' → {mapped_label}: {len(files)} images")

        for f in tqdm(files, desc=f"Processing {folder}"):
            path = os.path.join(subdir, f)
            try:
                raw = analyze_image(path, question, model, processor, device)
                pred, rationale, conf = parse_model_response(raw)
            except Exception as e:
                raw = f"Exception during inference: {e}"
                pred, rationale, conf = "error", raw, None

            print(f"\n🖼️ Image: {f}")
            print(f"📘 Raw / rationale: {rationale}")
            print(f"🔎 Prediction: {pred}, true label: {mapped_label}")
            print(f"📊 Confidence: {conf}")

            rows.append(
                {
                    "image": f,
                    "true_label": mapped_label,
                    "prediction": pred,
                    "rationale": rationale,
                    "confidence": conf,
                    "raw_response": raw,
                }
            )

    df = pd.DataFrame(rows)

    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_tag = f"run{run_number}" if run_number is not None else ""

    json_path = os.path.join(
        output_dir, f"medgemma-4b-it_results_{run_tag}_{ts}.json"
    )
    df.to_json(json_path, orient="records", indent=2)
    print(f"✅ Saved per-image JSON: {json_path}")

    acc, sens, spec = compute_metrics(df)

    # ---- PRINT FINAL METRICS ----
    print("\n================ FINAL METRICS ================")
    print(f"{run_tag}")
    print(f"Accuracy:    {acc:.4f}")
    print(f"Sensitivity: {sens:.4f}")
    print(f"Specificity: {spec:.4f}")
    print("================================================")

    df_metrics = pd.DataFrame(
        [
            {
                "accuracy": acc,
                "sensitivity": sens,
                "specificity": spec,
                "num_samples": len(df),
            }
        ]
    )
    csv_path = os.path.join(
        output_dir, f"medgemma-4b-it_metrics_{run_tag}_{ts}.csv"
    )
    df_metrics.to_csv(csv_path, index=False)
    print(f"✅ Saved metrics CSV: {csv_path}")


# ==========================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MedGemma dermoscopy classifier")
    parser.add_argument(
        "root_dir",
        type=str,
        help="Directory containing mm_class and bn_class subfolders",
    )
    parser.add_argument(
        "output_dir",
        type=str,
        help="Directory to save results (JSON + CSV)",
    )
    parser.add_argument(
        "run_number",
        type=int,
        nargs="?",
        default=None,
        help="Run number (1,2,3,...) for tagging outputs",
    )

    args = parser.parse_args()
    process_dataset(args.root_dir, args.output_dir, args.run_number)


""""

python medgemma_zeroshot_isic.py /mnt/d/Naved/Data/ISIC99/images/originals/ /mnt/d/Naved/Outputs/isic99_orig/ 1 


"""