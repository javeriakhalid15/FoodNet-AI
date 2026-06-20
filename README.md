# FoodNetAI 
## Fruit, Vegetable & Nut Classifier

## Project Description
A deep learning image classifier that identifies 15 classes of fruits, vegetables, and nuts from the Fruits-360 dataset, and displays nutritional information for the predicted item. Three models were trained and compared: MobileNetV2 (transfer learning), VGG16 (transfer learning), and a custom CNN built from scratch. The best-performing model (MobileNetV2) is deployed in an interactive Streamlit app.

## Results Summary

| Model       | Test Accuracy | Parameters |
|-------------|---------------|------------|
| MobileNetV2 | 99.38%        | 2.6M       |
| VGG16       | 99.38%        | 14.7M      |
| Custom CNN  | 98.87%        | 4.86M      |

MobileNetV2 was selected for deployment due to its significantly smaller size and faster inference at equal accuracy to VGG16.

## Setup Instructions
1. Clone the repo: git clone https://github.com/javeriakhalid15/FoodNetAI.git
2. Create and activate a virtual environment:
   - python -m venv venv
   - venv\Scripts\activate (Windows)
3. Install dependencies: pip install -r requirements.txt

## Running the App
python -m streamlit run app.py

This opens the app at http://localhost:8501. Upload an image of a fruit, vegetable, or nut to get a prediction with confidence score and nutrition info.

## Project Structure

```text
FoodNetAI/
├── app.py                  # Streamlit GUI
├── config.json             # Training hyperparameters
├── requirements.txt
├── models/                 # Trained model weights, encoder
├── src/
│   ├── food_info.py        # Nutrition data, class mappings
│   ├── preprocess.py       # Dataset loading and preprocessing
│   └── train.py            # Model architecture, training, evaluation
└── results/                # Training curves, confusion matrices, comparison plots
```

## Model Details
- Input size: 100x100x3
- Classes: 15 (5 fruits, 5 vegetables, 5 nuts)
- Dataset: Fruits-360 (Kaggle)
- Training: MobileNetV2 and VGG16 used transfer learning (ImageNet weights, frozen base + fine-tuning phase); Custom CNN trained entirely from scratch with no pretrained weights, for direct comparison.

## Known Limitations
- This is a closed-set classifier — it will always assign one of its 15 known labels to any input image, including images with no relation to the trained classes. High confidence does not guarantee correctness on out-of-distribution inputs.
- The Fruits-360 dataset consists of clean, lab-photographed images with uniform backgrounds, which may not generalize well to real-world photos with varied lighting, angles, and backgrounds.

## Deployment
Run locally via Streamlit (see above). The app loads model weights and rebuilds the architecture in code (rather than loading a fully serialized model file) to avoid Keras version compatibility issues across environments.
