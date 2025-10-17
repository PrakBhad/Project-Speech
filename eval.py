import os
import asyncio
import pandas as pd
import csv
from deep_translator import GoogleTranslator
import torch
import whisper
from jiwer import wer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import matplotlib.pyplot as plt
import time
import tempfile
import soundfile as sf
from datasets import load_dataset
from evaluate import load as load_metric
import subprocess

# Model Setup
print("Loading Whisper model (small)...")
device = "cuda" if torch.cuda.is_available() else "cpu"
model = whisper.load_model("small", device=device)

# Configuration
LANG_CODES = ["en_us", "hi_in", "de_de", "fr_fr"]
TARGET_LANG = {
    "en_us": "spanish",
    "hi_in": "english",
    "de_de": "english",
    "fr_fr": "english"
}
SAMPLES_PER_LANG = 100
OUTPUT_CSV = "fleurs_whisper_small_results.csv"

# Helper Functions


def run_async(coro):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


def save_audio_temp(audio_array, sampling_rate):
    tmp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    sf.write(tmp_file.name, audio_array, sampling_rate)
    return tmp_file.name


async def translate_text(text, target_lang):
    if not text.strip() or target_lang.lower() in ["none", "original"]:
        return text
    try:
        return GoogleTranslator(source='auto', target=target_lang.lower()).translate(text)
    except Exception:
        return ""


async def transcribe_and_translate(audio_file, target_lang):
    wav_file = ensure_wav_format(audio_file)
    start_time = time.time()
    result = await asyncio.to_thread(model.transcribe, wav_file)
    end_time = time.time()
    transcript = result["text"].strip()
    translation = await translate_text(transcript, target_lang)
    os.remove(wav_file)
    processing_time = end_time - start_time
    return transcript, translation, processing_time


def ensure_wav_format(input_path):
    """Convert audio to mono 16kHz WAV for Whisper."""
    tmp_wav = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name
    cmd = ["ffmpeg", "-y", "-i", input_path,
           "-ac", "1", "-ar", "16000", tmp_wav]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return tmp_wav


def load_fleurs_dataset(language_code, sample_count):
    dataset = load_dataset("google/fleurs", language_code,
                           split="test", trust_remote_code=True)
    if sample_count > len(dataset):
        sample_count = len(dataset)
    dataset = dataset.shuffle(seed=42).select(range(sample_count))
    return dataset

# Evaluation


async def evaluate_dataset(language_code, target_lang, sample_count):
    dataset = load_fleurs_dataset(language_code, sample_count)
    results = []
    smooth = SmoothingFunction().method1
    bertscore = load_metric("bertscore")

    for idx, example in enumerate(dataset):
        # Save audio temporarily
        audio_file = save_audio_temp(
            example["audio"]["array"], example["audio"]["sampling_rate"])
        ref_transcript = example["transcription"]

        # Some FLEURS languages may not have a translation column
        if "translation" in example:
            ref_translation = example["translation"]
        else:
            ref_translation = await translate_text(ref_transcript, target_lang)

        # Transcribe + translate
        pred_transcript, pred_translation, proc_time = await transcribe_and_translate(audio_file, target_lang)
        os.remove(audio_file)

        # Metrics
        wer_value = wer(ref_transcript.lower(), pred_transcript.lower())
        bleu_value = sentence_bleu(
            [ref_translation.split()], pred_translation.split(), smoothing_function=smooth)
        bert_res = bertscore.compute(predictions=[pred_translation], references=[
                                     ref_translation], lang="en")
        bert_f1 = bert_res["f1"][0]

        print(f"\n--- {language_code.upper()} Clip {idx+1} ---")
        print(f"Ref Transcript: {ref_transcript}")
        print(f"Pred Transcript: {pred_transcript}")
        print(f"Ref Translation (EN): {ref_translation}")
        print(f"Pred Translation: {pred_translation}")
        print(
            f"WER: {wer_value:.3f}, BLEU: {bleu_value:.3f}, BERTScore: {bert_f1:.3f}, Time: {proc_time:.2f}s")

        results.append({
            "Language": language_code,
            "Audio Path": audio_file,
            "WER": round(wer_value, 3),
            "BLEU": round(bleu_value, 3),
            "BERTScore": round(bert_f1, 3),
            "Processing Time (s)": round(proc_time, 2)
        })

    return results

# Main
if __name__ == "__main__":
    all_results = []

    for lang_code in LANG_CODES:
        print(f"\nEvaluating {lang_code.upper()} dataset...")
        results = run_async(evaluate_dataset(
            lang_code, TARGET_LANG[lang_code], SAMPLES_PER_LANG))
        all_results.extend(results)

    # Save results
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["Language", "Audio Path", "WER",
                           "BLEU", "BERTScore", "Processing Time (s)"]
        )
        writer.writeheader()
        writer.writerows(all_results)

    print(f"\n✅ Evaluation complete! Results saved to {OUTPUT_CSV}")

    # Visualization
    df_results = pd.DataFrame(all_results)
    summary = df_results.groupby("Language")[
        ["WER", "BLEU", "BERTScore", "Processing Time (s)"]].mean().reset_index()

    def plot_bar(metric, color, title, filename):
        plt.figure(figsize=(8, 5))
        plt.bar(summary["Language"], summary[metric], color=color, alpha=0.7)
        plt.xlabel("Language")
        plt.ylabel(f"Average {metric}")
        plt.title(f"Average {metric} per Language (FLEURS)")
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.tight_layout()
        plt.savefig(filename, dpi=300)
        plt.show()

    plot_bar("WER", "orange", "WER", "fleurs_average_wer.png")
    plot_bar("BLEU", "blue", "BLEU", "fleurs_average_bleu.png")
    plot_bar("BERTScore", "purple", "BERTScore",
             "fleurs_average_bertscore.png")
    plot_bar("Processing Time (s)", "green",
             "Processing Time", "fleurs_processing_time.png")
