# 🧠 AI-Assisted Dynamic Malware Family Classification using BODMAS Dataset

## **Research Objectives**

This project advances malware analysis by developing an **AI-powered malware family classification system** using the **BODMAS dataset**.

### 🔍 Research Gap
Previous baseline research only performed **binary classification** — distinguishing between *malicious* and *benign* samples.  
Our model goes **beyond binary classification** by accurately identifying the **specific malware family** (e.g., *Trojan, Worm, Ransomware, Downloader*, etc.).  
This enhancement fills a significant **gap in behavioral malware analysis research**.

### Key Goals
- Perform **multi-class malware family classification** (not just benign vs malware).  
- Apply **LightGBM with Optuna optimization** for robust training.  
- Select top features via **SelectKBest** to improve model performance.  
- Deploy a **Flask-based REST API** for real-time malware prediction.

---

## **Methodology**

### 1️⃣ Data Preparation
- Load **BODMAS dataset** (`bodmas.npz` and `bodmas_metadata.csv`).
- Remove benign samples and filter families with less than 5 samples.
- Encode malware family labels using **LabelEncoder**.

### 2️⃣ Data Splitting & Scaling
- Split dataset into **Train (70%)**, **Validation (15%)**, and **Test (15%)**.
- Normalize all features using **StandardScaler**.

### 3️⃣ Feature Selection
- Select top **200 features** with `SelectKBest(f_classif)` for better model generalization.

### 4️⃣ Model Training (LightGBM)
- Train **LightGBM Classifier** with **Optuna hyperparameter tuning**.
- Optimize parameters like `num_leaves`, `max_depth`, `learning_rate`, etc.
- Evaluate model with **macro F1-score** and **accuracy**.

### 5️⃣ Model Evaluation
- Compute **classification report**, **F1-score**, and **confusion matrix**.
- Visualize top-performing malware families.

### 6️⃣ Model Export
All trained artifacts are saved as:
- `final_model.pkl`
- `scaler.pkl`
- `selector.pkl`
- `label_encoder.pkl`
- `label_to_family.pkl`

---

## **Expected Contributions**

- Bridges the research gap from **binary** to **multi-class malware classification**.  
- Demonstrates that **behavioral features** can differentiate malware families.  
- Provides a **ready-to-deploy Flask API** for real-time malware family prediction.  
- Enables security researchers to conduct **fine-grained malware family studies**.

---

## **Dataset**

**Dataset:** BODMAS – Behavioral Dataset for Malware Analysis  
**Source:** [https://whyisyoung.github.io/BODMAS/](https://whyisyoung.github.io/BODMAS/)  
**Type:** Dynamic behavior data collected using **Cuckoo Sandbox**

**Files:**
- `bodmas.npz` → Numerical feature vectors
- `bodmas_metadata.csv` → Family names and metadata

Filtering criteria:
- Only malware samples retained (benign removed)
- Families with <5 samples excluded

---

## **Running Application**

### **1️⃣ Train the Model**
```bash
python final.py
```
Trains LightGBM classifier and saves model artifacts.

### **2️⃣ Launch the API**
```bash
python app.py
```
**Endpoint:** `POST http://127.0.0.1:5000/predict`

**Sample Request:**
```json
{"features": [0.123, 1.456, 0.789, ...]}
```

**Sample Response:**
```json
{
  "prediction": {
    "family_name": "Trojan",
    "encoded_label": 3,
    "confidence": 0.92
  }
}
```

### **3️⃣ Test the API**
```bash
python test.py
```

---

## **Folder Structure**
```
├── BODMAS/
│   ├── bodmas.npz
│   ├── bodmas_metadata.csv
│
├── models/
│   ├── final_model.pkl
│   ├── scaler.pkl
│   ├── selector.pkl
│   ├── label_encoder.pkl
│   ├── label_to_family.pkl
│
├── app.py
├── final.py
├── sample.py
├── test.py
└── README.md
```

---

## **Citation**
> WhyisYoung, *BODMAS: A Behavioral Dataset for Malware Analysis*, GitHub.io (2022).  
> [https://whyisyoung.github.io/BODMAS/](https://whyisyoung.github.io/BODMAS/)

---

## **Summary**
This project introduces a **novel malware family classification system** that moves beyond binary detection.  
By leveraging **behavioral analysis** and **machine learning**, it enables precise identification of malware families—closing a crucial research gap and contributing to stronger AI-based cybersecurity defense systems.
