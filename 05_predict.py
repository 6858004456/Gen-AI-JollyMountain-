"""Type or paste a text message and get a scam prediction."""
import joblib

try:
    model = joblib.load("scam_model.joblib")
except FileNotFoundError:
    raise SystemExit("Run 'python 04_train_model.py' first to create the model.")

print("Scam text detector. Type a message and press Enter (type 'quit' to exit).\n")

while True:
    text = input("Message: ").strip()
    if text.lower() in {"quit", "exit", "q"}:
        break
    if not text:
        continue
    scam_probability = model.predict_proba([text])[0][1]
    verdict = "LIKELY SCAM" if scam_probability >= 0.5 else "Looks legitimate"
    print(f"  -> {verdict} (scam probability: {scam_probability:.0%})\n")
