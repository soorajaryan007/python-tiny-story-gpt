---

# 📚 **README.md — Tiny StoryGPT (Keyword → Story Generator)**

```markdown
# 🌟 Tiny StoryGPT  
A small Transformer-based model that generates children’s stories from simple keywords.  
Example:

**Input:**  
`dragon adventure`

**Output:**  
`Once upon a time a brave dragon flew across the mountains to help a little girl...`

This project uses a lightweight GPT-style decoder-only Transformer that can run **fully offline on CPU**.

---

## 📁 Project Structure

```

story_project/
├── model.py              # Transformer model architecture
├── train.py              # Train model on stories.txt (run once)
├── generate.py           # Load trained model & generate stories
├── stories.txt           # Training dataset (<IN> ... <OUT> ...)
├── story_model.pth       # Saved trained model (generated after training)
├── vocab.pkl             # Saved vocabulary (generated after training)
└── README.md

````

---

## 🚀 Features

- Generate stories from keywords (e.g., `magic forest`, `space travel`)
- Offline local generation (CPU only required)
- Fast training on Google Colab (T4 GPU)
- Fully customizable dataset
- Clean modular code (`model.py`, `train.py`, `generate.py`)

---

## 🛠️ Installation

### 1️⃣ Clone the repository
```bash
git clone https://github.com/yourusername/storygpt.git
cd storygpt
````

### 2️⃣ Install dependencies

You only need PyTorch:

```bash
pip install torch
```

(Optional) On CPU-only machines:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

---

## 📦 Dataset Format (`stories.txt`)

Each line should follow:

```
<IN> keywords <OUT> full story text...
```

Example:

```
<IN> dragon adventure <OUT> Once upon a time a brave dragon helped a little boy in the mountains.
<IN> space travel <OUT> A young girl built her own rocket and visited friendly aliens.
<IN> magic forest <OUT> In a glowing forest, a tiny fairy protected lost travelers.
```

Add 20–100 lines for better results.

---

## 🧠 Training the Model (Optional)

You **only need to train once** on Colab or any GPU system.

Run:

```bash
python train.py
```

After training completes, you will see two new files:

```
story_model.pth
vocab.pkl
```

These contain the trained model and vocabulary.

Copy them to your local machine if you trained on Colab.

---

## ✨ Generating Stories (Local Machine)

Make sure you have:

✔ `model.py`
✔ `generate.py`
✔ `story_model.pth`
✔ `vocab.pkl`

Then simply run:

```bash
python generate.py
```

Enter your keywords:

```
Enter story keywords: dragon adventure
```

Output:

```
Once upon a time a mighty dragon soared across the mountains to help a small child...
```

➡️ **No GPU required**
➡️ **No retraining needed**

---

## 📌 Notes

* Model runs **fully offline**
* CPU is enough to generate stories
* GPU is only needed for training (optional)
* You can modify `stories.txt` to teach your model new types of stories
* For better results, add at least 50–100 examples

---

## 🧩 Common Issues

### ❗ `KeyError: word not in vocabulary`

You used a keyword not present in `stories.txt` during training.
Fix: retrain after adding the word to your dataset.

### ❗ Slow training on CPU

Use Google Colab and select **GPU (T4)**.

---

## 📷 Screenshots (optional)

*Add your outputs here*

---

## ❤️ Contribute

Pull requests with:

* better dataset
* UI improvements
* sampling (top-k, temperature)

are welcome.

---

## 📜 License

MIT License

```

---
