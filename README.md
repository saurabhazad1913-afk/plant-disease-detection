# Plant Disease Detection

Leaf image se plant disease pehchanne wala deep learning model (MobileNetV2, Transfer Learning) aur Streamlit web app (image upload + live camera).

## Dataset
Kaggle: arjuntejaswi/plant-village (15 classes: Tomato, Potato, Pepper bell)

## Result
Validation accuracy: 90.31%

## Setup
pip install -r requirements.txt

## Run
python train.py
streamlit run app.py

Note: dataset folder repo me nahi hai. Kaggle se download karke `dataset/` naam ke folder me rakho.
