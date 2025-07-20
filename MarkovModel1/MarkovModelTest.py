from MarkovStrategyModel import MarkovChainModel

model = MarkovChainModel()
model.load("markov_model.json")

# User input
prev_price = float(input("Enter previous price: "))
curr_price = float(input("Enter current price: "))

prediction = model.predict_next(prev_price, curr_price)

print(f"\n🧾 Current State: {prediction['current_state']}")
print("📊 Next State Probabilities:")
for state, prob in prediction['next_state_probabilities'].items():
    print(f"→ {state.capitalize()}: {round(prob * 100, 2)}%")

probs = model.steady_state_probabilities()
print("\n📌 Long-run (Steady-State) Probabilities:")
loss, stable, profit = probs
print(f"📉 Loss %: {round(loss * 100, 2)}%")
print(f"⏸️ Stability %: {round(stable * 100, 2)}%")
print(f"📈 Profit %: {round(profit * 100, 2)}%")

print("\n📋 Transition Matrix:")
print(model.transition_matrix)
