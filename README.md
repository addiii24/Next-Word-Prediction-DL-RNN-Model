# 🧠 Next Word Prediction — LSTM Neural Network

A deep learning-powered **Next Word Prediction** web application built with a Long Short-Term Memory (LSTM) neural network and served through a beautiful **Streamlit** interface.

---

## 🚀 Demo

> Type any seed phrase and let the model predict what comes next — powered by a trained LSTM language model.

**Three prediction modes:**
- 🔮 **Single Next Word** — predicts the single most likely next word
- 🎯 **Top-K Suggestions** — ranks the top K candidate words with probabilities
- ✨ **Auto-Generate Sequence** — autoregressively generates multiple words

---

## 📁 Project Structure

```
Next Word Prediction/
│
├── app.py               # Streamlit web application (UI + inference)
├── lstm_model.h5        # Trained LSTM model weights (Keras H5 format)
├── tokenizer.pkl        # Fitted Keras Tokenizer (word → index mapping)
├── max_len.pkl          # Maximum sequence length used during training
└── README.md            # Project documentation
```

---

## 🧰 Tech Stack

| Component        | Technology                          |
|------------------|-------------------------------------|
| Language         | Python 3.8+                         |
| Deep Learning    | TensorFlow / Keras                  |
| Web Framework    | Streamlit                           |
| Data Handling    | NumPy, Pickle                       |
| H5 Patching      | h5py (Keras version compatibility)  |
| UI Styling       | Custom CSS (Glassmorphism, dark mode)|

---

## 🏗️ Model Architecture

The model is a **sequential LSTM** network trained for next-word prediction:

```
Input (token sequence)
        ↓
Embedding Layer  [input_dim=10000, output_dim=50]
        ↓
LSTM Layer(s)
        ↓
Dense Layer  [units = vocab_size, activation = softmax]
        ↓
Output (probability distribution over vocabulary)
```

- **Task:** Multi-class classification over vocabulary  
- **Loss:** Categorical Cross-Entropy  
- **Optimizer:** Adam  
- **Input:** Padded token sequences of length `max_len - 1`  
- **Output:** Softmax probabilities for each word in the vocabulary  

---

## ⚙️ Setup & Installation

### 1. Clone the repository

```bash
git clone https://github.com/addiii24/Next-Word-Prediction-DL-RNN-Model.git
cd "Next Word Prediction"
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install streamlit tensorflow numpy h5py
```

> **Note:** If you have a GPU, install `tensorflow-gpu` instead for faster inference.

---

## ▶️ Running the App

```bash
streamlit run app.py
```

The app will open automatically in your browser at `http://localhost:8501`.

---

## 🖥️ App Features

### 🎨 UI Highlights
- **Dark glassmorphism** design with animated gradient background
- Smooth **glow animations** on result cards
- Responsive **stats dashboard** (vocabulary size, max sequence length, model status)
- Modern **Inter** typography via Google Fonts

### ⚙️ Sidebar Controls
| Setting | Description |
|---------|-------------|
| Prediction Mode | Switch between Single / Top-K / Auto-Generate |
| K (suggestions) | Number of candidate words to display (Top-K mode) |
| Words to Generate | Number of words to auto-generate (Sequence mode) |

### 📊 Probability Chart
In **Top-K mode**, a live bar chart displays the probability distribution across the top candidate words.

---

## 🔧 Keras Version Compatibility

If you see this error on model load:

```
Unrecognized keyword arguments passed to Embedding: {'quantization_config': None}
```

This means the `.h5` file was saved with **Keras 3.x** but your environment uses an older version. The app automatically handles this by:

1. Attempting a normal `load_model()` first
2. If that fails, copying the H5 to a temp file, **patching the model config JSON** via `h5py` to strip unsupported keys, then loading the patched copy
3. The original `lstm_model.h5` is **never modified**

---

## 📊 How It Works

```
User types seed text
        ↓
Text → tokenized using saved Tokenizer
        ↓
Sequence padded to (max_len - 1) with pre-padding
        ↓
LSTM model outputs softmax probability vector
        ↓
argmax (or top-K) index → decoded back to word(s)
        ↓
Result displayed in the UI
```

---

## 💡 Usage Tips

- Provide **at least 2–3 words** for contextually richer predictions
- Words outside the training vocabulary are mapped to `OOV` (out-of-vocabulary) and may reduce accuracy
- The **Auto-Generate** mode compounds predictions, so initial context quality matters
- For best results, use text that is stylistically similar to the model's training corpus

---

## 📦 Dependencies

```txt
streamlit>=1.28.0
tensorflow>=2.10.0
numpy>=1.23.0
h5py>=3.7.0
```

Install all at once:

```bash
pip install streamlit tensorflow numpy h5py
```

---

## 🙋 Author

**Adii** — [@addiii24](https://github.com/addiii24)

---

## 📄 License

This project is open-source and available under the [MIT License](LICENSE).
