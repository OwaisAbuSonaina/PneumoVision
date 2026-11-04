# PneumoVision

[![Python](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/API-FastAPI-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Gradio](https://img.shields.io/badge/UI-Gradio-FF7C00)](https://www.gradio.app/)
[![Docker Compose](https://img.shields.io/badge/Run-Docker%20Compose-2496ED?logo=docker&logoColor=white)](https://docs.docker.com/compose/)

PneumoVision is a computer-vision application that analyzes chest X-ray images and estimates the likelihood of pneumonia. It is designed as a **second-reader aid** that can help highlight scans for closer clinical review; it is not a diagnostic tool.

## What it does

- Upload a JPG or PNG chest X-ray through a simple Gradio interface.
- Run inference with a ResNet50-based binary classifier.
- Return a `Normal` or `Pneumonia` prediction with a confidence score.
- Present confidence as low, borderline, moderate, or high probability.
- Try the included sample X-rays directly from the interface.

The main goal of this repository is the working application: a frontend connected to a backend inference API. The backend downloads `best_model.keras` from the [PneumoVision Hugging Face model repository](https://huggingface.co/OwaisAbuSonaina/PneumoVision) on first startup and caches it in a Docker volume.

## Demo

The application accepts an image, sends it to the API, and displays the predicted class, probability, and an interpretation tier. The frontend also includes sample X-rays so the workflow can be tried without preparing an input image.

## Run locally

Requirements: [Docker](https://docs.docker.com/get-docker/) and Docker Compose.

```bash
git clone https://github.com/OwaisAbuSonaina/PneumoVision.git
cd PneumoVision
docker compose up --build
```

Open the application at [http://localhost:7860](http://localhost:7860). The API is available at [http://localhost:8000](http://localhost:8000), with interactive documentation at [http://localhost:8000/docs](http://localhost:8000/docs).

To call the prediction endpoint directly:

```bash
curl -X POST http://localhost:8000/predict \
  -F "file=@path/to/chest-xray.jpeg"
```

## Model and evaluation

The classifier uses a frozen ImageNet-pretrained ResNet50 feature extractor with a small binary classification head. Images are resized to `224 x 224` and processed with the ResNet preprocessing function. Training uses augmentation and balanced batches for Normal and Pneumonia examples.

The following results are from the recorded evaluation run in `train_model/notebooks/main.ipynb` on 624 test images. The operating threshold was tuned on the validation set using Youden's J statistic.

| Metric | Result |
| --- | ---: |
| Test loss | 0.3065 |
| Accuracy | 87.66% |
| AUC | 0.9434 |
| Tuned threshold | 0.523 |

| Class | Precision | Recall | F1-score |
| --- | ---: | ---: | ---: |
| Normal | 0.86 | 0.81 | 0.83 |
| Pneumonia | 0.89 | 0.92 | 0.91 |


## Project structure

```text
.
├── frontend/       Gradio user interface and sample images
├── backend/        FastAPI service and model inference
├── train_model/    Training, preprocessing, and evaluation reference
└── docker-compose.yml
```

`train_model/` is included for reference if you want to inspect how the model was trained, reproduce the workflow, or experiment with training and evaluation. It is not required to run the main application.

## Responsible use

PneumoVision is an experimental research and educational project. Predictions can be wrong and must not be used as a standalone medical diagnosis or as a substitute for review by a qualified healthcare professional.
