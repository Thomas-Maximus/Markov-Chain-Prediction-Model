# 📈 Markov Chain Stock Predictor

A lightweight Python-based tool that uses a **Markov Chain model** to predict market trends — **profit**, **loss**, or **stability** — based on recent stock price changes.

---

## 📦 What's Included

MarkovPredictor/
├── markov_model.json # Trained transition matrix for prediction
├── MarkovStrategyModel.py # Core model logic
├── MarkovModelTest.py # CLI script to test predictions
├── requirements.txt # (Optional) Dependencies for training
└── README.md # You're reading this!

---

## 🚀 Getting Started

### 🔧 Requirements

- Python 3.7+
- No external libraries needed for prediction
- Only `numpy` required for training

Install optional dependency if retraining:
```bash
pip install numpy
```

▶️ How to Use
🧪 Run the Prediction Script
```bash
python MarkovModelTest.py
```
💬 Example Prompt
```yaml
Enter previous price: 100
Enter current price: 104
```
✅ Output Example
```yaml
🧾 Current State: profit
📊 Next State Probabilities:
→ Profit: 47.5%
→ Loss: 18.2%
→ Stable: 34.3%

📌 Long-run (Steady-State) Probabilities:
📉 Loss %: 28.65%
⏸️ Stability %: 36.79%
📈 Profit %: 34.56%
```
# 📚 How It Works
The model uses a 3-state Markov Chain:
  profit: price went up

  loss: price went down

  stable: price changed within ±0.1%

Transition probabilities are learned from historical price sequences
Predictions are based on the current price movement’s transition row


# 🔄 Retraining the Model (Optional)

If you want to use new data:
Edit and run save_model.py (not included in default bundle)
It fetches stock data from Yahoo Finance using yfinance
Trains and saves a new markov_model.json

# 💡 Notes
This is a probabilistic model, not a deterministic predictor.
It’s best used for trend tendency analysis, not exact price prediction.
You can extend it with live price fetching, Kivy mobile GUI, or visual dashboards.

# 📤 Author & License
Author: Thomas Maximus
License: MIT



