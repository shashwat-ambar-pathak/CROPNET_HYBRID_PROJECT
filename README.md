# 🌱 CropNet-Hybrid: Deep Learning Model for Crop Disease Detection

## 📌 Overview
**CropNet-Hybrid** is a custom Convolutional Neural Network (CNN)-based hybrid model designed for **crop disease detection and classification**.  
It has been trained on the **PlantVillage dataset**, covering **38 different crop–disease classes** across multiple plant species.  

This project integrates **Machine Learning, FastAPI, and React** to build a real-time **Crop Disease Detection Web Application**.  
The goal is to empower farmers and researchers with an intelligent tool to detect diseases early and protect crops effectively.  

---

## 🚀 Features
- ✅ **Hybrid CNN architecture** for high accuracy and robustness  
- ✅ **Trained on PlantVillage dataset** with 38 classes of crop diseases  
- ✅ **Data augmentation** (rotation, scaling, flipping, zooming) for generalization  
- ✅ **Exported & deployed with FastAPI backend** and **React frontend**  
- ✅ **Multilingual accessibility** for farmers and users worldwide  
- ✅ Designed for **sustainable agriculture and early disease detection**  

---

## 🗂️ Dataset
- **Source**: [PlantVillage Dataset](https://www.kaggle.com/datasets/emmarex/plantdisease)  
- **Classes**: 38 crop + disease categories  
- **Preprocessing**:  
  - Resized images to `224x224`  
  - Normalized pixel values  
  - Applied augmentation techniques  

---

## 🏗️ Model Architecture
- Convolutional Layers (feature extraction)  
- Batch Normalization (stability & faster convergence)  
- Dropout (reduce overfitting)  
- Fully Connected Dense Layers  
- Softmax Output Layer (for 38 classes)  

---

## 📊 Results
- **Accuracy**: 98% 
- **Loss Curve & Accuracy Curve** (add graphs if available)  

---

## 💻 Tech Stack
- **Modeling**: Python, TensorFlow / Keras  
- **Backend**: FastAPI  
- **Frontend**: React + Tailwind CSS  
- **Deployment**: GitHub Pages / Render / Vercel  

---

## ⚡ How to Run Locally
### 1️⃣ Clone the Repository
```bash
git clone https://github.com/shashwat-ambar-pathak/cropnet-hybrid.git
cd cropnet-hybrid

---

### 2️⃣ Create Virtual Environment & Install Dependencies
```bash
python -m venv venv
source venv/bin/activate   # (Linux/Mac)
venv\Scripts\activate      # (Windows)
pip install -r requirements.txt

---

### 3️⃣ Train the Model
```bash
python train.py

---

4️⃣ Run Backend (FastAPI)
```bash
uvicorn app.main:app --reload

---

5️⃣ Run Frontend (React)
```bash
cd frontend
npm install
npm start

---

🌍 Deployment

Model integrated with FastAPI + React

Hosted via GitHub Pages / Render / Vercel (update link here)
👉 Live Demo: Your Project Link

---

🤝 Contributing

Contributions, issues, and feature requests are welcome!
Feel free to fork this repo and submit a pull request.

---

📜 License

This project is licensed under the MIT License – see the LICENSE
 file for details.

---

👨‍💻 Author

Shashwat Ambar Pathak
🌐 LinkedIn
 | 🐙 GitHub
