# 🧠 Detection and Classification of Brain Alzheimer’s Using CNN Model

## 📊 Project Overview
This project focuses on using **Convolutional Neural Networks (CNNs)** to detect and classify **Alzheimer’s Disease (AD)** from **MRI brain scans**. By leveraging deep learning, the system categorizes MRI images into four stages: **Non-Demented, Very Mild Demented, Mild Demented, and Moderate Demented**. The model aims to assist **early diagnosis** and enhance **treatment planning** for patients.

## 👥 Team Members
- **Soham Vasudeo**  
- **Shardul Patki**  

**Course:** Machine Learning Project 2022-23  
**Institution:** NMIMS University  

## 🎯 Objectives
The goal of this project is to develop a **high-accuracy deep learning model** that can detect Alzheimer’s disease from MRI images with **minimal manual intervention**.

### **Key Goals:**
✔ Improve **early detection of AD** using CNN-based classification.  
✔ Train a **deep learning model** on MRI images with **SMOTE** for class balance.  
✔ Enhance **model accuracy** through data augmentation and hyperparameter tuning.  
✔ Automate the **detection and classification** process for real-time diagnosis.  

---

## 🗂 Data Understanding
The dataset consists of **MRI brain scans** categorized into four stages of Alzheimer's disease.

### **Dataset Components:**
📌 **MRI Images:** Collected from the **Open Access Series of Imaging Studies (OASIS) database**.  
📌 **Class Labels:**  
  - **Non-Demented:** 3,210 images  
  - **Very Mild Demented:** 2,240 images  
  - **Mild Demented:** 896 images  
  - **Moderate Demented:** 64 images  

### **Challenges Addressed:**
🚧 **Class Imbalance** – Used **SMOTE (Synthetic Minority Oversampling Technique)** to balance the dataset.  
🚧 **High Computational Load** – Optimized CNN architecture for **faster processing**.  
🚧 **Variability in MRI Scans** – Applied **data augmentation** for model generalization.  

---

## ⚙️ Data Preparation
### **Data Preprocessing:**
🔹 Applied **data augmentation** using `ImageDataGenerator` to increase dataset size.  
🔹 Used **SMOTE** to generate synthetic samples for underrepresented classes.  
🔹 Standardized images to **176x176x3 resolution** for uniformity.  

### **Feature Engineering:**
🔹 Extracted key MRI features using **CNN layers**.  
🔹 Implemented **brightness adjustment, horizontal flips, and zoom transformations** for better generalization.  

### **Data Splitting:**
📌 **Training Set** – 70%  
📌 **Validation Set** – 15%  
📌 **Testing Set** – 15%  

---

## 🧠 Modeling
This project uses **deep learning-based CNN models** to classify Alzheimer’s disease.

### **CNN Model Architecture:**
✅ **Conv2D Layers:** Extract spatial features from MRI images.  
✅ **MaxPooling Layers:** Reduce dimensionality and computational complexity.  
✅ **Batch Normalization & Dropout:** Prevent overfitting and improve model generalization.  
✅ **Fully Connected Dense Layers:** Perform final classification into four AD stages.  

### **Training Process:**
✔ Optimized using **Adam optimizer** and **Categorical Cross-Entropy loss function**.  
✔ Hyperparameter tuning for **learning rate, kernel size, and dropout rates**.  

---

## 📈 Model Evaluation
The CNN model was evaluated using **various performance metrics**.

### **Performance Metrics:**
📌 **Accuracy:** 93.7% on test data  
📌 **Precision & Recall:** Ensured high detection accuracy with minimal false positives.  
📌 **F1-Score:** Balanced measure of classification performance.  

### **Comparison with Other Methods:**
| Method | Accuracy |
|--------|---------|
| Multi-task learning | 87.5% |
| 3D CNN Model | 89.7% |
| Sparse Representation | 86.0% |
| Deep Belief Network | 91.0% |
| **Proposed CNN Model** | **93.7%** |

⏳ **Inference Speed:** Optimized CNN **reduces processing time per MRI scan**, improving diagnostic efficiency.

---

## 🚀 Deployment Strategy
The model is designed to be **scalable** and **deployable** in clinical settings.

### **1. Infrastructure Setup:**  
☁ Deployed on **Google Cloud / AWS** for scalability.  

### **2. Model Deployment:**  
✅ Flask API & TensorFlow Serving for real-time inference.  
✅ Integrated with **hospital EMR systems** for clinical use.  

### **3. Continuous Monitoring:**  
📊 Used **CloudWatch** and **Azure Monitor** for performance tracking.  

---

## 📊 Key Insights and Results
📌 **High Classification Accuracy:** 93.7% test accuracy achieved.  
📌 **Class Imbalance Addressed:** SMOTE balanced dataset for better predictions.  
📌 **Fast Model Inference:** Optimized CNN reduces processing time.  
📌 **Assistive for Early Diagnosis:** Helps in identifying Alzheimer's **before severe symptoms appear**.  

---

## 🛠 Tools & Technologies Used
### **Programming & Frameworks:**  
🖥 **Python, TensorFlow, Keras, NumPy, Pandas**  

### **Deep Learning Models:**  
🧠 **Convolutional Neural Networks (CNNs)**  

### **Data Processing & Augmentation:**  
🖼 **OpenCV, ImageDataGenerator, SMOTE**  

### **Deployment & Integration:**  
🌍 **Flask API, TensorFlow Lite (TFLite), Google Cloud**  

---

## 🏆 Achievements
🏅 Developed a **high-accuracy deep learning model** for Alzheimer’s detection.  
🏅 Successfully **classified MRI brain scans** into four AD stages.  
🏅 Integrated **SMOTE to resolve dataset imbalance**, improving model performance.  
🏅 Achieved **93.7% accuracy**, surpassing traditional detection methods.  

---

## 📊 Visuals
📸 **Sample MRI Images:** Showing different Alzheimer’s stages.  
![image](https://github.com/user-attachments/assets/e6b392a6-5a99-4e3b-bb65-0922e2880901)

📊 **Model Performance Graphs:** Accuracy, loss curves, confusion matrix.  
![image](https://github.com/user-attachments/assets/46bbce85-e752-4e78-828d-f0762f91a751)
![image](https://github.com/user-attachments/assets/937d613c-db36-4c2a-8763-c2ff08d029c2)

---

## ⚡ Future Scope
🔹 Enhance the CNN model with **transfer learning techniques**.  
🔹 Implement **real-time MRI analysis** for **clinical use**.  
🔹 Expand dataset to include **multi-modal medical imaging** for better insights.  
🔹 Deploy the model into a **user-friendly mobile or web application** for accessibility.  

---

## 📜 License
This project is open-source and available under the **MIT License**.  

---

