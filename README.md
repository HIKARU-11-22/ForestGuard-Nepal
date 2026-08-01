# ForestGuard Nepal — Forest Fire / Smoke / Non-Fire Classification

## Project Overview

ForestGuard Nepal is a deep-learning image-classification project for identifying forest/fire-related images as:

- Fire
- Smoke
- Non-Fire

The project is implemented in a Jupyter Notebook using **TensorFlow/Keras**. The notebook covers dataset exploration, preprocessing, model training, evaluation, confusion matrices, error analysis, and single-image prediction.

The project compares two approaches:

1. **Custom CNN**
2. **MobileNetV2 Transfer Learning with Fine-Tuning**

This README is based on the implementation in the project's main notebook and does not assume additional models or techniques that are not part of that implementation.

---

## 1. Project Objectives

The project aims to:

- classify images into Fire, Smoke, and Non-Fire;
- explore and prepare the image dataset;
- train a Custom CNN from scratch;
- train a MobileNetV2-based model using transfer learning;
- fine-tune the MobileNetV2 model;
- evaluate model performance using Accuracy, Precision, Recall and F1-score;
- generate confusion matrices;
- analyse incorrect predictions;
- identify common confusion patterns;
- perform prediction on a single image.

---

## 2. Dataset

The project uses the **Forest Fire, Smoke and Non-Fire Image Dataset** available on Kaggle:

https://www.kaggle.com/datasets/amerzishminha/forest-fire-smoke-and-non-fire-image-dataset

The expected local directory structure is:

```text
data/
└── raw/
    ├── train/
    │   ├── fire/
    │   ├── smoke/
    │   └── non-fire/
    │
    └── test/
        ├── fire/
        ├── smoke/
        └── non-fire/
```

The notebook loads the class names automatically from the training dataset.

### Dataset note

The project is motivated by forest-fire detection in Nepal, but the selected image dataset should not automatically be considered a complete Nepal-specific dataset. Results should therefore be interpreted as results on the selected dataset rather than as guaranteed real-world performance across all regions of Nepal.

---

## 3. Main Notebook

The main project notebook contains the complete workflow:

```text
ForestGuard_Complete_Training.ipynb
```

The notebook includes:

1. Imports and configuration
2. Dataset checking
3. Image/sample exploration
4. Dataset loading
5. Data augmentation
6. Custom CNN construction
7. Custom CNN training
8. MobileNetV2 construction
9. MobileNetV2 transfer learning
10. MobileNetV2 fine-tuning
11. Model evaluation
12. Confusion matrices
13. Error analysis
14. Single-image prediction

---

## 4. Technologies Used

The notebook imports and uses:

- Python
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Pillow
- TensorFlow
- Scikit-learn

The notebook also uses TensorFlow/Keras utilities for image loading, augmentation, model construction and training.

---

## 5. Configuration

The project uses the following configuration from the notebook:

```text
Image size: 224 × 224
Batch size: 32
Random seed: 42
```

The dataset paths are:

```text
data/raw/train
data/raw/test
```

The training data is divided into training and validation subsets using a validation split of:

```text
15%
```

The test set is kept separate and loaded without shuffling for evaluation.

---

## 6. Data Preparation

The notebook loads images using:

```python
tf.keras.utils.image_dataset_from_directory()
```

The training and validation datasets are created from the training directory using the same random seed.

The test dataset is loaded separately from:

```text
data/raw/test
```

The images are resized to:

```text
224 × 224
```

and batches are created using:

```text
batch size = 32
```

The class names are obtained from the dataset itself.

---

## 7. Data Augmentation

The notebook applies data augmentation as part of the model workflow.

The augmentation is used to help the models learn from variations in the training images rather than relying only on the exact appearance of the original images.

The exact augmentation operations should be taken from the notebook when describing the implementation in the final report.

---

# 8. Model 1 — Custom CNN

The first model is a Custom Convolutional Neural Network.

The notebook describes the architecture as using:

- four convolution blocks;
- batch normalization;
- max pooling;
- dropout;
- global average pooling;
- dense classification;
- softmax output.

The final dense layer produces one output for each class.

The model is compiled using:

```text
Optimizer: Adam
Learning rate: 0.001
Loss: sparse_categorical_crossentropy
Metric: accuracy
```

The model is trained for up to:

```text
10 epochs
```

with callbacks for:

- Early Stopping
- ReduceLROnPlateau
- Model Checkpoint

The best Custom CNN model is saved as:

```text
models/custom_cnn.keras
```

---

# 9. Model 2 — MobileNetV2

The second model uses **MobileNetV2** with ImageNet-pretrained weights.

The notebook initially freezes the MobileNetV2 base model and adds a classification head consisting of:

- data augmentation;
- MobileNetV2 preprocessing;
- MobileNetV2 feature extractor;
- Global Average Pooling;
- Dropout;
- Dense softmax classification layer.

The MobileNetV2 input size is:

```text
224 × 224 × 3
```

The initial model uses:

```text
Optimizer: Adam
Learning rate: 0.001
Loss: sparse_categorical_crossentropy
Metric: accuracy
```

---

# 10. MobileNetV2 Fine-Tuning

After the initial transfer-learning stage, the notebook fine-tunes the MobileNetV2 model.

The final 30 layers are unfrozen and trained using a lower learning rate.

This allows the pretrained feature extractor to adapt more closely to the ForestGuard image classification task.

The fine-tuned model is saved as:

```text
models/mobilenetv2_finetuned.keras
```

---

# 11. Model Evaluation

The project evaluates the trained model on the unseen test dataset.

The evaluation includes:

- Accuracy
- Precision
- Recall
- F1-score
- Classification report
- Confusion matrix
- Normalized confusion matrix
- Error analysis
- Incorrect prediction visualisation

The actual numerical results must be obtained by running the notebook. Results should not be manually estimated or invented.

---

## 12. Evaluation Metrics

### Accuracy

Measures the proportion of test images classified correctly.

```text
Accuracy = Correct Predictions / Total Predictions
```

### Precision

Measures how reliable the model's predictions are for a class.

### Recall

Measures how many actual examples of a class are correctly identified.

### F1-score

Combines Precision and Recall into a single measure.

The project reports Precision, Recall and F1-score using the model evaluation code in the notebook.

---

# 13. Confusion Matrix

The notebook generates a confusion matrix to show how the model classifies each actual class.

The matrix helps identify errors such as:

```text
Actual Fire → Predicted Smoke
Actual Smoke → Predicted Fire
Actual Smoke → Predicted Non-Fire
Actual Non-Fire → Predicted Fire
```

A normalized confusion matrix is also generated so that class-level performance can be compared using proportions.

---

# 14. Error Analysis

The notebook analyses incorrect predictions by:

- counting incorrect predictions;
- calculating error rates by class;
- locating incorrect test images;
- displaying incorrect predictions;
- identifying common actual/predicted confusion pairs.

The notebook specifically highlights possible visual confusion involving:

- Fire and Smoke;
- Smoke and Non-Fire;
- Non-Fire and Fire;
- haze;
- clouds;
- distant smoke;
- visually ambiguous scenes.

Only patterns actually observed in the generated results should be reported in the final project report.

---

# 15. Single-Image Prediction

The notebook also contains a function for predicting a single image.

The image is:

1. opened using Pillow;
2. converted to RGB;
3. resized to 224 × 224;
4. converted to a NumPy array;
5. passed to the trained model;
6. assigned the class with the highest probability.

The notebook displays:

```text
Prediction: <class>
Confidence: <percentage>
```

---

# 16. Results

After running the notebook, the final report should include the actual results.

A suitable comparison table is:

| Metric | Custom CNN | MobileNetV2 Fine-Tuned |
|---|---:|---:|
| Accuracy | Actual result | Actual result |
| Precision | Actual result | Actual result |
| Recall | Actual result | Actual result |
| F1-score | Actual result | Actual result |

The final report should also include:

- classification report;
- confusion matrix;
- normalized confusion matrix;
- error analysis;
- examples of incorrect predictions.

**Do not enter numerical values until the notebook has been executed.**

---

# 17. Project Structure

A suitable repository structure for the notebook-based project is:

```text
ForestGuard/
│
├── data/
│   └── raw/
│       ├── train/
│       └── test/
│
├── notebooks/
│   ├── Data_Preprpssing.ipynb
│   ├── Evaluation_and_Demo.ipynb
│   ├── Model1_Baseline_CNN.ipynb
│   ├── Model1_vs_Model2_Comparision.ipynb
│   ├── Model2_Transferign_PyTorch_Colab.ipynb
│   └── ForestGuard_Complete_Training.ipynb
│
├── models/
│   ├── best_model2.pt
│   └── nodel1_baseline_cnn.pth
│
├── results/
│
├── docs/
│
├── requirements.txt
├── README.md
└── .gitignore
```

The raw image dataset should not be committed to GitHub if it is excluded by the project's `.gitignore`.

---

# 18. Installation

Create and activate a Python environment, then install the dependencies listed in:

```
## Setup

```bash
git clone https://github.com/HIKARU-11-22/ForestGuard-Nepal.git
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
```text
requirements.txt
```

Install with:

```bash
pip install -r requirements.txt
```

Then open the notebook:

```bash
jupyter notebook
```

and run:

```text
notebooks/ForestGuard_Complete_Training.ipynb
```

---

# 19. Running the Project

### Step 1 — Place the dataset

Place the downloaded dataset under:

```text
data/raw/
```

with separate:

```text
train/
test/
```

directories.

### Step 2 — Install dependencies

```bash
pip install -r requirements.txt
```

### Step 3 — Open the main notebook

```text
notebooks/ForestGuard_Complete_Training.ipynb
```

### Step 4 — Run the notebook

Run the cells in order.

The notebook will:

```text
Load Dataset
     ↓
Explore Dataset
     ↓
Prepare Train / Validation / Test
     ↓
Train Custom CNN
     ↓
Train MobileNetV2
     ↓
Fine-Tune MobileNetV2
     ↓
Evaluate Model
     ↓
Generate Confusion Matrices
     ↓
Analyse Errors
     ↓
Predict Single Image
```

---

# 20. Member 4 — Evaluation Responsibilities

For the group project, the evaluation work includes:

- evaluating the trained model;
- reporting Accuracy;
- reporting Precision;
- reporting Recall;
- reporting F1-score;
- generating classification reports;
- generating confusion matrices;
- generating normalized confusion matrices;
- analysing errors by class;
- identifying common confusion pairs;
- visualising incorrect predictions;
- writing the experiments and results section;
- documenting ethical and social considerations.

The evaluation notebook should be run using the trained model and unseen test data.

---

# 21. Ethical and Social Considerations

ForestGuard is an academic prototype and should not be treated as an autonomous emergency-response system.

### False Negatives

A Fire image incorrectly classified as Non-Fire could result in a missed warning if the system were used in a real monitoring environment.

### False Positives

A Non-Fire image incorrectly classified as Fire could generate an unnecessary warning.

### Dataset Limitations

The selected image dataset may not represent every:

- forest type;
- geographical region;
- season;
- weather condition;
- camera type;
- lighting condition

found in Nepal.

Therefore, strong test performance on this dataset does not automatically guarantee equivalent performance in real Nepalese forests.

### Human Oversight

A safer future system would use the model as a decision-support tool:

```text
Image
  ↓
Model Prediction
  ↓
Potential Warning
  ↓
Human Verification
  ↓
Decision
```

The model should not independently make emergency decisions.

---

# 22. Environmental Considerations

Training deep-learning models requires computational resources.

The project uses transfer learning with MobileNetV2, which can reduce the amount of training required compared with training a large model completely from scratch.

For future development, unnecessary repeated training should be avoided and trained checkpoints should be reused where appropriate.

---

# 23. Limitations

The current project has several limitations:

1. It performs image classification rather than object detection or fire localisation.
2. The dataset may not fully represent Nepalese forest conditions.
3. Fire and Smoke can have visually similar characteristics.
4. Image quality and background conditions can affect predictions.
5. The model can produce false positives and false negatives.
6. Test-set performance does not guarantee real-world deployment performance.
7. The system is an academic prototype rather than a certified emergency system.

---

# 24. Reproducibility

The notebook uses:

```text
Image size: 224 × 224
Batch size: 32
Random seed: 42
Validation split: 15%
```

Keeping these settings unchanged helps reproduce the experiments.

The test dataset is loaded separately from the training data and is not used for model training.

---

# 25. GitHub Notes

Do not commit unnecessary files such as:

```text
.venv/
__pycache__/
.ipynb_checkpoints/
```

The raw dataset should also remain excluded from GitHub according to the project's `.gitignore`.

For group contributions, each member should commit their own work with a clear message.

Example:

```bash
git add notebooks/Evaluation_Comparison_Member4.ipynb
git commit -m "Add model evaluation and error analysis"
git push
```

---

# 26. Project Status

| Component | Status |
|---|---|
| Dataset exploration | Implemented in notebook |
| Data preprocessing | Implemented |
| Data augmentation | Implemented |
| Custom CNN | Implemented |
| MobileNetV2 | Implemented |
| MobileNetV2 fine-tuning | Implemented |
| Model evaluation | Implemented |
| Classification report | Implemented |
| Confusion matrix | Implemented |
| Normalized confusion matrix | Implemented |
| Error analysis | Implemented |
| Incorrect prediction visualisation | Implemented |
| Single-image prediction | Implemented |
| Final numerical results | Obtain by running notebook |

---

# 27. Conclusion

ForestGuard demonstrates a complete image-classification workflow for Fire, Smoke and Non-Fire images.

The project compares a Custom CNN with a MobileNetV2 transfer-learning approach and evaluates the trained model using standard classification metrics and visual error analysis.

The final model selection should be based on the actual test-set results, including class-level performance and confusion patterns, rather than accuracy alone.

Because the dataset may not represent all Nepalese environments, further Nepal-specific data collection and validation would be required before considering real-world deployment.

---

## Dataset Reference

Forest Fire, Smoke and Non-Fire Image Dataset:

https://www.kaggle.com/datasets/amerzishminha/forest-fire-smoke-and-non-fire-image-dataset
