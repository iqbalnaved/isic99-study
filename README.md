# k-Shot Dermoscopy Classification with Multimodal LLMs

Benchmarks zero-shot and few-shot classification performance of multimodal LLMs on a balanced ISIC dermoscopy image subset.

## Task

Binary classification: **Melanoma** vs. **Benign**  
Independent variable: **shot count** (k = 0, 1, 3, 5, 7)

## Dataset

- ~99 dermoscopy images, balanced across classes
- Source: [ISIC Archive](https://www.isic-archive.com/)

## Models

| Model | Inference | Hardware |
|---|---|---|
| `google/medgemma-4b-it` | Local | RTX 3070 (4-bit quant) / A5000 (bfloat16) |
| `gpt-4.1-2025-04-14` | OpenAI API | — |
| `gpt-4.1-mini` / `gpt-5-mini` | OpenAI API | — |

## Metrics

Accuracy · Sensitivity (TPR) · Specificity (TNR)

---

## Repository Structure

| File | Description |
|---|---|
| `medgemma_zeroshot_isic.py` | MedGemma zero-shot inference; VRAM-aware quantization; JSON output |
| `isic100_fewshot.py` | GPT k-shot inference (k = 0/1/3/5/7); chain-of-thought; JSON output |
| `isic100_fewshot_v2.py` | Updated few-shot variant |
| `isic100hr_crop.py` | High-resolution crop pipeline |
| `isic100hr_simple_crop.py` | Simplified crop pipeline |
| `topK_finding.py` | Top-K image similarity for few-shot candidate selection |
| `isic_metadata_download.py` | ISIC metadata download via API |
| `isic_json_to_csv.py` | JSON → CSV metadata conversion |
| `data_resizing.py` | Image resizing for model input |
| `copy_unique.py` | Image deduplication utility |
| `isic100.txt` | Image ID list |

---

## Setup

```bash
pip install torch transformers accelerate bitsandbytes \
            huggingface_hub pillow pandas tqdm openai scikit-learn
```

```bash
export HF_TOKEN="your_huggingface_token"   # MedGemma is a gated model
export OPENAI_API_KEY="your_openai_key"
```

---

## Usage

### MedGemma (zero-shot, local)

```bash
python medgemma_zeroshot_isic.py \
    /path/to/ISIC99/images/originals/ \
    /path/to/outputs/ \
    1    # run number
```

### GPT k-Shot

```bash
python isic100_fewshot.py \
    --shot 3 \
    --run 1 \
    --model gpt-4.1-2025-04-14
```

---

## Output

**MedGemma** — JSON per image:
```json
{
  "image": "ISIC_0000001.jpg",
  "true_label": "melanoma",
  "prediction": "melanoma",
  "rationale": "...",
  "confidence": 0.87
}
```

**GPT few-shot** — JSON per image:
```json
{
  "thoughts": "...",
  "answer": "Melanoma"
}
```

---

## Affiliation

West Virginia University — Department of Microbiology, Immunology & Cell Biology  
PI: Gangqing Hu · Collaborator: Donald Adjeroh

## License

Research use only. ISIC images subject to [ISIC Archive terms](https://www.isic-archive.com/).
