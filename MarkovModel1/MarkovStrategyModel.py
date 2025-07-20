import numpy as np
import json

class MarkovChainModel:
    def __init__(self, threshold=0.001):
        self.threshold = threshold
        self.transition_matrix = np.zeros((3, 3))
        self.states = ['profit', 'loss', 'stable']
        self.state_indices = {s: i for i, s in enumerate(self.states)}

    def _get_state(self, change):
        if change > self.threshold:
            return 'profit'
        elif change < -self.threshold:
            return 'loss'
        else:
            return 'stable'

    def train(self, prices):
        state_sequence = [
            self._get_state((prices[i] - prices[i - 1]) / prices[i - 1])
            for i in range(1, len(prices))
        ]

        for i in range(len(state_sequence) - 1):
            current = self.state_indices[state_sequence[i]]
            next_ = self.state_indices[state_sequence[i + 1]]
            self.transition_matrix[current][next_] += 1

        row_sums = self.transition_matrix.sum(axis=1, keepdims=True)
        self.transition_matrix = np.divide(
            self.transition_matrix,
            row_sums,
            out=np.zeros_like(self.transition_matrix),
            where=row_sums != 0
        )

    def predict_next(self, last_price, current_price):
        change = (current_price - last_price) / last_price
        current_state = self._get_state(change)
        idx = self.state_indices[current_state]
        self.transition_matrix[idx]
        return {
            'current_state': current_state,
            'next_state_probabilities': {
                s: round(float(self.transition_matrix[idx][j]), 4)
                for j, s in enumerate(self.states)
            }
        }

    def steady_state_probabilities(self):
        matrix = np.array(self.transition_matrix)
        eigvals, eigvecs = np.linalg.eig(matrix.T)
        idx = np.argmin(np.abs(eigvals - 1))
        steady_state = np.real(eigvecs[:, idx]).flatten()
        steady_state = steady_state / steady_state.sum()
        return steady_state.tolist()

    def save(self, filepath):
        data = {
            'threshold': self.threshold,
            'transition_matrix': self.transition_matrix.tolist() if isinstance(self.transition_matrix, np.ndarray) else self.transition_matrix
        }
        with open(filepath, 'w') as f:
            json.dump(data, f)

    def load(self, filepath):
        with open(filepath, 'r') as f:
            data = json.load(f)
            self.threshold = data['threshold']
            self.transition_matrix = np.array(data['transition_matrix'])

# save_model.py
import yfinance as yf

# Choose the stock ticker and data period
ticker = "RELIANCE.NS"  # Replace with the desired stock ticker
period = "6mo"
interval = "1d"

# Download live historical stock candle data
data = yf.download(ticker, period=period, interval=interval)
print(type(data["Close"]))
prices = data["Close"].dropna().squeeze().tolist()

# Train and save the model
model = MarkovChainModel(threshold=0.001)
model.train(prices)
model.save("markov_model.json")

print(f"✅ Model trained on {ticker} data and saved to markov_model.json")
