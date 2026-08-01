# ForestGuard Nepal

## Forest Fire and Smoke Detection Using Deep Learning

ForestGuard Nepal is a supervised multiclass image-classification prototype that predicts:

- Fire
- Smoke
- Non-Fire

The project compares a custom CNN with MobileNetV2 transfer learning and provides a Streamlit interface for image-based predictions.

## Setup

```bash
git clone <YOUR-REPOSITORY-URL>
cd ForestGuard-Nepal
python -m venv .venv
```

Linux/macOS:

```bash
source .venv/bin/activate
```

Windows:

```bash
.venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Download the Kaggle dataset and extract it according to `data/README.md`.

## Run training

```bash
python -m src.train
```

## Evaluate

```bash
python -m src.evaluate --model custom_cnn
python -m src.evaluate --model mobilenetv2_finetuned
```

## Run the application

```bash
streamlit run app.py
```

## Expected outputs

- Saved models in `models/`
- Classification reports in `results/metrics/`
- Confusion matrices in `results/figures/`

## Important limitation

The training dataset is global rather than Nepal-specific. The system is a prototype proposed for Nepal and should not be treated as a replacement for official wildfire monitoring or emergency decision-making.
