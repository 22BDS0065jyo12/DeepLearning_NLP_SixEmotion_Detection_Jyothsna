# 🧠 NLP Six Emotion Detection and Classification

This project focuses on detecting and classifying six basic human emotions from text using Natural Language Processing (NLP) and Deep Learning techniques. It includes traditional machine learning models as well as an LSTM-based neural network for comparison.

---

## 📌 Emotions Detected
- **0** – Anger  
- **1** – Disgust  
- **2** – Fear  
- **3** – Joy  
- **4** – Sadness  
- **5** – Surprise  

---

## 📁 Dataset

The training dataset is stored in a file named `train.txt`, containing two columns:
- **Comment** – The text input (e.g., a sentence or phrase)
- **Emotion** – The emotion label corresponding to the comment

Format:
I am so happy today! ; Joy
---

## 🛠️ Technologies Used

### ✅ Libraries & Tools
- **TensorFlow / Keras** – for deep learning (LSTM)
- **Scikit-learn** – for machine learning models and metrics
- **NLTK** – for stopword removal and stemming
- **Matplotlib / Seaborn / WordCloud** – for data visualization
- **Pandas / NumPy** – for data manipulation
- **Pickle** – for saving models

---

## 🧪 Models Implemented

### 🔹 Traditional Machine Learning
- **Multinomial Naive Bayes**
- **Logistic Regression**
- **Random Forest**
- **Support Vector Machine (SVM)**

### 🔹 Deep Learning
- **LSTM** (Long Short-Term Memory) network with Embedding layer

---

## 🧼 Data Preprocessing

- Removal of non-alphabet characters
- Lowercasing text
- Stemming using PorterStemmer
- Stopword removal using NLTK
- TF-IDF vectorization
- Padding for LSTM inputs

---

## 📊 Model Evaluation

Each classifier is evaluated using:
- **Accuracy**
- **Classification Report** (Precision, Recall, F1-Score)
### 📈 Example Accuracy Results (TF-IDF based)
| Model                 | Accuracy |
|----------------------|----------|
| Naive Bayes          | 65.9%    |
| Logistic Regression  | 82.5%    |
| Random Forest        | 84.7%    |
| SVM                  | 81.6%    |
| LSTM (Deep Learning) | 92.9%    |

---

## 🔮 Sample Predictions

```python
sentences = [
    "I didn't feel humiliated",
    "I feel strong and good overall",
    "He was speechless when he got the job",
    "This is outrageous!",
    "I'm feeling grouchy",
    "Mom is really sweet and caring"
]
The model predicts the emotion and label with high confidence using either the ML or DL approach.

📦 Saved Artifacts
The following components are saved using pickle for future inference:

logistic_regression.pkl
label_encoder.pkl
tfidf_vectorizer.pkl
---

📌 Future Enhancements
Add UI using Streamlit or Flask

Deploy as a web app

Expand to more emotion classes

Use pre-trained language models like BERT for better accuracy
## 📸 Screenshots

### 😡 Anger

![Anger](https://github.com/22BDS0065jyo12/DeepLearning_NLP_SixEmotion_Detection_Jyothsna/blob/main/anger.png?raw=true)



👩‍💻 Author
Jyothsna Hanumanthu

