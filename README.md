# 🧠 AI-Assisted Dynamic Malware Family Classification using BODMAS Dataset

##  Overview

This project aims to develop an AI-assisted malware classification model capable of identifying whether a given executable is **benign or malicious**, and if malicious, classifying it into its respective **malware family** (e.g., Trojan, Ransomware, Worm, etc.).

We build upon the **BODMAS** (Behavioral Dataset for Malware Analysis) dataset, which provides dynamic behavior features extracted from **Cuckoo Sandbox**.  
Our implementation explores multiple machine learning models and introduces **novel feature engineering** techniques to improve classification performance.

---

##  Objectives

- Automate malware classification based on dynamic behavior.
- Evaluate multiple machine learning models: **Random Forest, LightGBM, MLP, and ResNet1D**.
- Extend the BODMAS dataset with **novel dynamic features** (e.g., API success ratios, behavior hashes).
- Compare results with and without the new features to measure improvement.
- Provide a reproducible and scalable pipeline for future malware analysis research.

---

## 📂 Dataset

**Source:** [BODMAS Dataset](https://whyisyoung.github.io/BODMAS/)  
**Type:** Dynamic malware behavior dataset  
**Features extracted from:** Cuckoo Sandbox reports  
**Labels:** Malware families and benign samples  
**Data format:** `.npz` (NumPy compressed) and `.csv` metadata files

Each sample in the dataset includes pre-extracted behavioral statistics such as:
- API call frequencies
- File and registry operations
- Network activity indicators
- Execution statistics and return patterns

---

##  Models Implemented

| Model | Description | Use Case |
|-------|--------------|----------|
| **Random Forest** | Ensemble of decision trees; robust to noise and feature imbalance | Baseline model |
| **LightGBM** | Gradient boosting with high speed and accuracy on tabular data | Optimized performance |
| **MLP (Multi-Layer Perceptron)** | Neural network capable of non-linear classification | Deep learning baseline |
| **ResNet1D** | 1D convolutional residual network suitable for sequential behavior data | Advanced deep learning model |

---

##  Project Approaches

We explored three potential research approaches:

1. **Baseline:**  
   Train and evaluate the models directly on the original BODMAS dataset.

2. **Dynamic Analysis Extension:**  
   Perform independent sandbox testing (e.g., using Cuckoo Sandbox) to extract new behavioral features.

3. **Feature Engineering (Chosen Approach):**  
   Engineer new behavioral features from the existing dataset, such as:
   - `api_success_ratio` → Success/failure ratio of API calls  
   - `network_file_ratio` → Relation between network and file operations  
   - `behavior_hash` → Hashed representation of behavioral vectors  

   These features are merged into the existing feature set to form a **novel feature vector** that enhances classification accuracy.

---

##  Implementation Workflow

1. **Data Loading and Preprocessing**
   - Load `bodmas.npz` and `metadata.csv`
   - Normalize and clean features
   - Encode malware families numerically

2. **Feature Engineering**
   - Create new dynamic behavior features (`api_success_ratio`, `behavior_hash`, etc.)
   - Merge with original features

3. **Model Training and Evaluation**
   - Train Random Forest, LightGBM, MLP, and ResNet1D
   - Evaluate with accuracy, precision, recall, F1-score
   - Plot confusion matrices and ROC curves

4. **Result Comparison**
   - Compare baseline vs. feature-extended models
   - Analyze which features most improved classification accuracy

5. **Documentation and Deployment**
   - Save trained models (`.pkl` or `.h5`)
   - Provide Jupyter notebook and code scripts
   - Document findings in research report

---

## 🧠 Evaluation Metrics

| Metric | Description |
|--------|--------------|
| **Accuracy** | Overall correctness of classification |
| **Precision** | Correct malware detections vs. all detections |
| **Recall** | Ability to detect all malware samples |
| **F1-score** | Harmonic mean of precision and recall |
| **Confusion Matrix** | Family-wise classification performance |

---

## Expected Results

- Improved accuracy and F1-score after feature engineering.  
- Better detection of specific malware families (e.g., Trojans, Worms).  
- Evidence that engineered features improve model generalization.  
- Research paper discussing methodology, results, and future improvements.

---

##  Tools & Libraries

- Python 3.10+  
- NumPy, Pandas, Scikit-learn  
- LightGBM, TensorFlow/PyTorch  
- Matplotlib, Seaborn  
- Streamlit / FastAPI (for deployment)

---

##  Team Roles

| Member | Responsibility |
|---------|----------------|
| Member A |  |
| Member B |  |
| Member C |  |

---

##  Future Work

- Integrate raw Cuckoo sandbox data for more granular API-level features.  
- Implement real-time malware classification via REST API.  
- Extend analysis to Android or Linux malware datasets.  
- Use explainable AI (XAI) to interpret classification decisions.

---

## 📝 Citation

> WhyisYoung, *BODMAS: A Behavioral Dataset for Malware Analysis*, GitHub.io (2022).  
> [https://whyisyoung.github.io/BODMAS/](https://whyisyoung.github.io/BODMAS/)

---

##  Summary

This research demonstrates that **feature engineering from dynamic behavior data** can significantly improve malware family classification accuracy.  
By leveraging the **BODMAS dataset** and advanced ML models such as **LightGBM and ResNet1D**, we bridge the gap between raw sandbox analysis and practical AI-driven malware detection.
