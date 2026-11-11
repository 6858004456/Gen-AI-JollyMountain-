# JollyMountain Predict Lab: AI Predictive Systems for Scam Text Detection

A Gen AI class project by group JollyMountain.

| Name | Nickname |
|---|---|
| Girati Yamali | Phukao |
| Anuphan Jarasyingdamrong | Mac |

## What the project does

Our topic is **AI predictive systems**. We study one clear example: predicting whether a text message is a scam.

The project has two parts:

1. **A predictive model we train ourselves.** `04_train_model.py` turns text messages into numbers (TF-IDF) and trains a Logistic Regression model that predicts scam or legitimate. `05_predict.py` lets you test it in the terminal.
2. **A local AI app with our own knowledge.** `06_app.py` is a small website. We upload research PDFs about scam and spam detection, and the AI answers questions using those papers and shows the file and page it used. This is Retrieval-Augmented Generation (RAG). The same page also lets you paste a message and get a prediction from our trained model.

Everything runs on our own computer. The language model runs through [Ollama](https://ollama.com/).

## How the app answers a question

1. Each uploaded PDF is read page by page (pypdf) and cut into short text pieces made of whole sentences.
2. Each piece is turned into a vector by the `nomic-embed-text` model and saved in `library/index.json`.
3. The question is turned into a vector too, and the closest pieces are picked.
4. Those pieces are sent to the `gemma3:1b` model with the question. The answer lists its sources.
5. Untick "Use the uploaded papers" to see the answer without our documents and compare.

## Papers we upload

The PDFs are not stored in this repository.

1. Hammad, Ahmed, Hafez and Voon (2026). *A Secure Lightweight SMS Spam Detection Framework with Robustness to Text Obfuscation Attacks.* Computers, 15(7), 451. https://www.mdpi.com/2073-431X/15/7/451
2. Shinde, Shahra, Basurra, Saeed, AlSewari and Jabbar (2024). *SMS Scam Detection Application Based on Optical Character Recognition for Image Data Using Unsupervised and Deep Semi-Supervised Learning.* Sensors, 24. https://www.open-access.bcu.ac.uk/16359/
3. Songram, Kummanee, Chomphuwiset, Jareanpon, Boongasame and Kawattikul (2023). *Deep learning for classifying Thai deceptive messages.* Indonesian Journal of Electrical Engineering and Computer Science, 30(2), 1232-1241. https://zenodo.org/record/7972449

## How to run

You need Python 3.9 or newer and Ollama.

```
ollama pull gemma3:1b
ollama pull nomic-embed-text
pip install -r 02_requirements.txt
python 04_train_model.py
python 06_app.py
```

Open http://127.0.0.1:5001, upload the PDFs in the Paper library, then ask a question.

## Files

| File | What it does |
|---|---|
| `02_requirements.txt` | Python packages |
| `03_sample_data.csv` | Small set of labelled example messages |
| `04_train_model.py` | Trains and saves the scam prediction model |
| `05_predict.py` | Tests the model in the terminal |
| `06_app.py` | The web app: ask the papers, check a message, manage the paper library |

## Notes

- The sample data is small and for learning only. For better accuracy, add the public "SMS Spam Collection" as `spam.csv` and train again.
- Scams are rarer than normal messages in real life, so look at precision and recall, not only accuracy.
