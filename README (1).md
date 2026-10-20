# Arabic Egyptian Automatic Speech Recognition (ASR)

A Google Colab-based project for transcribing spoken Egyptian Arabic into Arabic text. The system uses a custom **CNN + Residual Block + BiLSTM** acoustic model trained with **Connectionist Temporal Classification (CTC)**, with an optional **KenLM 5-gram language model** to improve decoding. A simple **Gradio** interface lets users upload or record speech and view the transcription.

## Features

- Processes Egyptian Arabic speech from audio files.
- Extracts 40 MFCCs, plus delta and delta-delta features (120 features per time step).
- Trains on up to 10,000 samples from the `MAdel121/arabic-egy-cleaned` dataset.
- Uses a CNN, residual block, and bidirectional LSTM for acoustic modeling.
- Trains with CTC loss and evaluates using Word Error Rate (WER) and Character Error Rate (CER).
- Saves model checkpoints, resume checkpoints, cached features, and training logs to Google Drive.
- Uses KenLM 5-gram language-model beam search alongside greedy CTC decoding.
- Provides a Gradio UI for uploading or recording audio and displaying the Arabic transcription.

## How It Works

1. **Dataset preparation:** Loads the dataset, resamples audio to 16 kHz, filters samples by duration, and converts transcript characters into vocabulary indices.
2. **Feature extraction:** Normalizes audio and extracts MFCC, delta, and delta-delta features using `librosa`. Each sample is padded or truncated to 400 time steps.
3. **Model training:** Trains a CNN-BiLSTM model with CTC loss, SpecAugment, AdamW, mixed-precision support, and a learning-rate scheduler.
4. **Evaluation:** Calculates WER and CER on a validation subset and saves the best model according to CER.
5. **Language-model setup:** Builds a 5-gram KenLM model from the available transcript corpus if the binary model does not already exist.
6. **Inference:** Extracts features from uploaded audio, runs the acoustic model, decodes the output, and normalizes the resulting Arabic text.
7. **User interface:** Launches a Gradio interface for recording or uploading audio and viewing the transcription.

## Tech Stack

- Python
- PyTorch
- NumPy
- librosa
- Hugging Face Datasets
- scikit-learn
- jiwer
- KenLM
- pyctcdecode
- Gradio
- Google Colab / Google Drive

## Requirements

The notebook installs most dependencies as part of its setup. It expects a Google Colab environment with Google Drive access.

Main Python packages used:

- `numpy==1.26.4`
- `torch`
- `librosa`
- `datasets`
- `scikit-learn`
- `jiwer`
- `gradio`
- `torchaudio`
- `transformers`
- `soundfile`
- `pyctcdecode`
- `kenlm` (installed from its source repository)

The KenLM build also requires system build tools and libraries, including CMake, a C++ compiler, Boost, zlib, bzip2, and lzma development libraries.

## Getting Started

### 1. Open the notebook in Google Colab

Copy the project code into a Colab notebook, preserving the cell order. The code uses shell commands such as `!pip` and `!apt-get`, so it is intended for a notebook environment rather than running unchanged as a standard Python script.

### 2. Mount Google Drive

Run the setup cell and authorize access when prompted. The project stores files under:

```text
/content/drive/MyDrive/Arabic_ASR/
├── checkpoints/
├── data/
└── logs/
```

### 3. Install dependencies

Run the notebook's installation cells. These install Python dependencies and the system libraries needed to build KenLM.

### 4. Prepare the dataset and features

Run the data preparation section. It downloads the training split of:

`MAdel121/arabic-egy-cleaned`

The notebook keeps up to 10,000 valid samples with durations between 0.5 and 7 seconds. Processed features are cached so subsequent runs can reuse them.

### 5. Train or resume the model

Run the training section. The configured maximum is 80 epochs, with early stopping based on CER. If a resume checkpoint exists, training attempts to continue from it.

### 6. Build or load the language model

Run the language-model section to build the KenLM 5-gram model if its binary file is not already present. The corpus is created from the processed training data.

### 7. Launch the transcription interface

Run the inference and Gradio section after the best model checkpoint and language-model files are available. Upload an audio file or record speech, then read or copy the generated transcription.

## Project Files

The notebook stores generated artifacts in Google Drive:

| Path | Purpose |
|---|---|
| `checkpoints/best_model.pt` | Best acoustic-model checkpoint according to validation CER |
| `checkpoints/resume_checkpoint.pt` | Model, optimizer, and scaler state for resuming training |
| `logs/training_log.txt` | Epoch-by-epoch training and validation metrics |
| `data/processed_10k_delta.pkl` | Cached processed audio features and labels |
| `data/lm_corpus.txt` | Text corpus used to train the language model |
| `data/arabic_lm.arpa` | ARPA-format KenLM language model |
| `data/arabic_lm.binary` | Binary KenLM language model used by the decoder |

## Configuration

Important settings defined in the notebook:

| Setting | Value | Description |
|---|---:|---|
| Maximum samples | 10,000 | Maximum dataset samples processed |
| Minimum audio duration | 0.5 seconds | Shorter samples are skipped |
| Maximum audio duration | 7 seconds | Longer samples are skipped |
| Sample rate | 16,000 Hz | Audio sampling rate |
| MFCC coefficients | 40 | Base acoustic features |
| Total feature dimension | 120 | MFCC + delta + delta-delta |
| Maximum feature length | 400 frames | Features are padded or truncated |
| Batch size | 32 | Training batch size |
| Maximum epochs | 80 | Maximum training epochs |
| Optimizer | AdamW | Optimizer with weight decay |
| Initial learning rate | `3e-4` | Learning rate before scheduling |
| Hidden size | 320 | BiLSTM hidden size per direction |
| BiLSTM layers | 3 | Number of recurrent layers |
| Dropout | 0.2 | Model dropout setting |
| Language model | 5-gram KenLM | Used for beam-search decoding |

## Evaluation

The notebook reports:

- **WER (Word Error Rate):** Measures word-level transcription errors.
- **CER (Character Error Rate):** Measures character-level transcription errors.

The best checkpoint is selected using CER. Actual accuracy depends on the trained checkpoint, audio quality, speaker, vocabulary, and dialect. The notebook includes a printed training result, but that result should be treated as specific to that run rather than a guaranteed performance level.

## Limitations and Notes

- The project is intended for **Egyptian Arabic**, not guaranteed to work equally well for Modern Standard Arabic or other dialects.
- The configured character vocabulary limits which characters can be represented; unsupported transcript characters are omitted during label conversion.
- Inputs are padded or truncated to a maximum of 400 feature frames. The interface recommends keeping recordings under 7 seconds.
- The language-model build depends on the processed dataset and on KenLM compiling successfully in the Colab environment.
- GPU use is automatic when CUDA is available; otherwise, inference and training run on CPU and may be slower.
- The notebook contains repeated package installation steps. Dependency versions may change over time, so a clean Colab runtime may be needed if imports conflict.
- `share=False` means the Gradio interface is not configured to create a public share link.
- This README describes the code's intended workflow; it does not guarantee that every notebook cell runs without modification in every environment.

## Troubleshooting

- **Google Drive errors:** Confirm that Drive was mounted and that the notebook has permission to access it.
- **Missing checkpoint:** Run the training section or place the expected `best_model.pt` file in the checkpoints directory.
- **Missing KenLM binary:** Run the language-model setup section and check that KenLM compiled successfully.
- **Dependency/import errors:** Restart the Colab runtime and rerun setup cells in order. Avoid mixing incompatible NumPy and compiled-package versions.
- **Poor transcription:** Use a clear recording, reduce background noise, speak at a normal pace, and keep the recording within the configured duration range.
- **Empty transcription:** Try a louder, clearer recording with less background noise.

## Acknowledgements

- Training data: [`MAdel121/arabic-egy-cleaned`](https://huggingface.co/datasets/MAdel121/arabic-egy-cleaned)
- Language-model toolkit: [KenLM](https://github.com/kpu/kenlm)
- CTC beam-search decoder: [pyctcdecode](https://github.com/kensho-technologies/pyctcdecode)
- Interface: [Gradio](https://www.gradio.app/)
- Audio features: [librosa](https://librosa.org/)
- Deep learning framework: [PyTorch](https://pytorch.org/)

## License

No project license is specified in the provided code. Add a license file if you intend to distribute or reuse this project publicly, and verify the dataset and dependency licenses before redistribution.
