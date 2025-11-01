import whisper
import gradio as gr
import time
import tempfile
import os
import subprocess
from deep_translator import GoogleTranslator
import torch

# Initialize Models
device = "cuda" if torch.cuda.is_available() else "cpu"
model = whisper.load_model("small",device=device)

# Helper Functions
def ensure_wav_format(input_path):
    """Ensure audio is converted to mono 16kHz WAV for Whisper."""
    tmp_wav = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name
    cmd = ["ffmpeg", "-y", "-i", input_path,
           "-ac", "1", "-ar", "16000", tmp_wav]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return tmp_wav


async def translate_text_async(text, target_lang):
    """Translate text asynchronously using deep-translator."""
    if not text.strip() or target_lang.lower() in ["none", "original"]:
        return text

    lang_codes = {
        "english": "en",
        "hindi": "hi",
        "french": "fr",
        "german": "de",
    }
    dest = lang_codes.get(target_lang.lower(), "en")

    try:
        translated = GoogleTranslator(
            source='auto', target=dest).translate(text)
        return translated
    except Exception as e:
        return f"[Translation error: {e}]"

# Core Transcription


async def transcribe(audio_file, target_language="English"):
    """Transcribe and optionally translate the given audio."""
    if audio_file is None:
        return "No audio provided", None

    audio_path = ensure_wav_format(audio_file)

    audio = whisper.load_audio(audio_path)
    audio = whisper.pad_or_trim(audio)
    mel = whisper.log_mel_spectrogram(audio).to(model.device)

    # Detect language
    _, probs = model.detect_language(mel)
    detected_lang = max(probs, key=probs.get)

    # Transcribe
    st_time = time.time()
    result = model.transcribe(audio_path)
    fn_time = time.time()

    transcript_text = "\n".join([seg["text"].strip()
                                for seg in result["segments"]])
    processing_time = f"Processing time: {fn_time - st_time:.2f} sec"

    # Optional translation
    translated_text = None
    if target_language.lower() not in ["none", "original", detected_lang.lower()]:
        translated_text = await translate_text_async(transcript_text, target_language)

    if translated_text:
        transcript_text_full = (
            f"Translation ({target_language})\n{translated_text}\n\n"
            f"Original ({detected_lang})\n{transcript_text}"
        )
    else:
        transcript_text_full = transcript_text

    display_text = (
        f"Detected language: {detected_lang}\n\n"
        f"{processing_time}\n\n"
        f"{transcript_text_full}"
    )

    # Save transcript
    tmp_dir = tempfile.gettempdir()
    txt_path = os.path.join(tmp_dir, "whisper_transcript.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(transcript_text_full)

    os.remove(audio_path)
    return display_text, txt_path

# Gradio UI
with gr.Blocks(title="Whisper Speech-to-Text Translator") as demo:
    gr.Markdown(
        """
        # 🎙 Whisper Speech-to-Text + Translation  
        Record or upload speech → get transcript + translation instantly.  
        *(Powered by OpenAI Whisper & Deep Translator)*  
        """
    )

    with gr.Row(equal_height=True):
        with gr.Column(scale=1):
            mic_input = gr.Audio(
                type="filepath",
                label="Record Audio",
                source="microphone",  # This works in Gradio < 3.39
            )
            upload_input = gr.Audio(
                type="filepath",
                label="Upload Audio File",
            )
            target_lang = gr.Dropdown(
                ["Original", "English", "Hindi", "French", "German"],
                value="English",
                label="Translate To",
            )
            submit_btn = gr.Button("Transcribe", variant="primary")

        with gr.Column(scale=2):
            transcript_output = gr.Textbox(
                label="Transcript / Translation Output",
                lines=20,
                interactive=False,
                show_copy_button=True
            )
            download_file = gr.File(label="Download Transcript")

    async def handle_inputs(mic_file, upload_file, target_language):
        audio_file = mic_file or upload_file
        return await transcribe(audio_file, target_language)

    submit_btn.click(
        fn=handle_inputs,
        inputs=[mic_input, upload_input, target_lang],
        outputs=[transcript_output, download_file],
    )

# Run App
if __name__ == "__main__":
    demo.launch(share=True)
