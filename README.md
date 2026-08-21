# Human_Attention_Span_Level

**A data-driven human attention span classification system that uses lifestyle and behavioral factors to predict attention levels as Low, Medium, or High using Machine Learning, Deep Learning, and a Hybrid ML-DL ensemble approach.**

---

## 🧠 About

**Human Attention Span Level Classification System** is a machine learning-based classification system designed to predict an individual's attention span level from lifestyle and behavioral patterns. The system analyzes factors such as sleep, screen time, physical activity, stress, caffeine intake, multitasking, and long-form content engagement to classify attention levels into **Low, Medium, and High** categories.

The project uses a survey-based dataset containing **2,132 responses** and applies a structured preprocessing and modeling pipeline. Multiple Machine Learning and Deep Learning models were implemented, evaluated, and compared, followed by the development of a **hybrid ensemble model** to improve prediction performance. 

### Key Highlights

* 🤖 **Multiple ML Models** - Random Forest, XGBoost, Extra Trees, Logistic Regression, and SVM
* 🧠 **Deep Learning Models** - ANN, MLP, DNN, and TabNet
* 🔗 **Hybrid ML-DL Ensemble** - Combines predictions from selected high-performing models
* 📊 **High Prediction Accuracy** - Hybrid model achieved **98.36% accuracy**
* ⚖️ **Class Balancing** - SMOTE used to address class imbalance
* 🔍 **Feature Importance Analysis** - Identifies the major factors influencing attention span
* 📈 **Model Evaluation** - Accuracy, precision, recall, F1-score, confusion matrix, ROC-AUC
* 💡 **Interpretable AI** - Permutation importance and Partial Dependence Plots used for model interpretation 

---

## ✨ Features

### 👤 User Information

The system considers demographic and lifestyle-related information such as:

* Age range
* Gender
* Occupation
* Education level

### 😴 Lifestyle Factors

* Sleep duration
* Sleep quality
* Physical activity
* Stress level
* Caffeine intake

### 📱 Digital & Behavioral Factors

* Screen time
* Multitasking behavior
* Long-form content engagement

### 🤖 Machine Learning

Several traditional machine learning algorithms were implemented and compared:

* Random Forest
* XGBoost
* Extra Trees
* Logistic Regression
* Support Vector Machine (SVM)

### 🧠 Deep Learning

The project also evaluates multiple deep learning architectures:

* Artificial Neural Network (ANN)
* Multi-Layer Perceptron (MLP)
* Deep Neural Network (DNN)
* TabNet

### 🔗 Hybrid ML-DL Model

A hybrid ensemble model was developed by combining the probability outputs of selected high-performing models using a **weighted averaging approach**. The hybrid model achieved **98.36% accuracy**, outperforming the individual models in the experimental evaluation. 

### 📊 Prediction & Analysis

The system provides:

* Predicted attention level
* Prediction confidence scores
* Model-based feature importance
* Result explanation
* Lifestyle factor analysis

The system classifies users into:

* 🟢 **High Attention**
* 🟡 **Medium Attention**
* 🔴 **Low Attention**

### 🔍 Feature Importance

Interpretability analysis was performed to identify the factors that contribute most to attention span prediction.

The major contributing factors identified in the study include:

* **Screen Time**
* **Physical Activity**
* **Sleep Hours**
* **Sleep Quality**
* **Long-form Content Engagement**

Screen time, sleep quality, and physical activity were identified among the strongest contributors through the feature importance analysis. 

### 📈 Model Evaluation

The implemented models were evaluated using several performance metrics:

* Accuracy
* Precision
* Recall
* F1-Score
* Confusion Matrix
* ROC Curve
* AUC Score 

---

## 🖼️ Screenshots

### Input & Prediction Interface

| **User Information**                 | **Lifestyle Factors**                        | **Behavioral Factors**                         |
| ------------------------------------ | -------------------------------------------- | ---------------------------------------------- |
| *Age, Gender, Occupation, Education* | *Sleep, Physical Activity, Stress, Caffeine* | *Screen Time, Multitasking, Long-form Content* |

### Prediction Results

| **Attention Level**   | **Prediction Confidence**          | **Result Explanation**              |
| --------------------- | ---------------------------------- | ----------------------------------- |
| *Low / Medium / High* | *Confidence scores for each class* | *Factors supporting the prediction* |

### Model Insights

| **Feature Importance**             | **Model Performance**                  | **Interpretability Analysis**  |
| ---------------------------------- | -------------------------------------- | ------------------------------ |
| *Most important lifestyle factors* | *Comparison of ML, DL & Hybrid models* | *Permutation Importance & PDP* |

---

## ⚙️ Methodology

The system follows a structured pipeline:

**Data Collection → Data Cleaning → Encoding → SMOTE → Feature Selection → Feature Scaling → ML/DL Model Training → Hybrid Ensemble → Evaluation → Interpretability Analysis → Attention Prediction**

The dataset was collected through an online survey and includes lifestyle, behavioral, and attention-related variables. The preprocessing pipeline includes categorical encoding, SMOTE-based class balancing, feature selection, and feature scaling. 

---

## 🏆 Project Outcome

The project successfully developed a multi-class classification system capable of predicting **Low, Medium, and High attention span levels** from lifestyle and behavioral data. The **hybrid ensemble model achieved 98.36% accuracy**, demonstrating the effectiveness of combining multiple predictive models for complex tabular data. 

The system also provides interpretable insights into how lifestyle and behavioral factors are associated with attention span, making it potentially useful in areas such as **education, productivity analysis, and behavioral analysis**. 
