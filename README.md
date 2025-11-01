# Speech Processing using NLP

This project focuses on **Speech Processing** using **Natural Language Processing (NLP)** techniques. It involves converting spoken language into text, analyzing linguistic patterns, and optionally synthesizing speech back from text. The environment leverages **Python 3.9**, **PyTorch 1.13.1**, and **CUDA 11.6** for GPU-accelerated model training and inference.

---

## Features

- **Speech-to-Text (STT)** — Convert audio input into text  
- **NLP Analysis** — Perform sentiment analysis, keyword extraction, or intent detection  
- **Text-to-Speech (TTS)** — Use `gTTS` for high-quality audio synthesis  
- **GPU Acceleration** — Utilize CUDA 11.6 for faster computations  
- **Enhanced Logging** — Use `rich` for visually appealing CLI output  

---

## Installation

### 1. Clone the Repository
```bash
git clone https://github.com/<your-username>/<your-repo-name>.git
cd <your-repo-name>
```

### 2. Create the Conda environment
``` bash
conda create -n speech-nlp python=3.9
conda activate speech-nlp
```

### 3. Install the pacages
``` bash
pip install -r requirements.txt
```

