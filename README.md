# Financial News Sentiment Analyzer

This small Flask app serves a trained sentiment model (Decision Tree + TF-IDF) to predict sentiment for short financial news snippets.

Project structure

- `app.py` - Flask application that loads the saved model and vectorizer and exposes `/` and `/predict` endpoints.
- `templates/index.html` - Front-end (inline CSS + JS) to submit text and show predictions.
- `sentiment_model.pkl` - Trained model (saved by the notebook). **Not included** — generate by running the notebook or place your model file here.
- `vectorizer.pkl` - Saved TF-IDF vectorizer used to transform input text.
- `all-data.csv` - Dataset used to train the model (optional).
- `requirements.txt` - Python dependencies.

Quickstart

1. Create a virtual environment (recommended):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

2. Install dependencies:

```powershell
pip install -r requirements.txt
```

3. Ensure `sentiment_model.pkl` and `vectorizer.pkl` exist in the project root. You can generate them by running the `finacial.ipynb` notebook (cells save `vectorizer.pkl` and `sentiment_model.pkl`).

4. Run the app:

```powershell
python app.py
```

5. Open http://127.0.0.1:5000/ in your browser and enter a news snippet to analyze.

Notes

- If your model returns string labels (e.g., 'positive', 'neutral', 'negative') the front-end will display them as-is. If it returns numeric labels, the app tries to convert them to integers for JSON transport.
- If you encounter issues with NLTK data in the notebook, the notebook includes a fallback that uses a simple regex-based tokenizer and a basic stopword list to avoid download issues.

Want me to run a quick smoke test of the Flask app here? I can try importing the app and sending a test request, but I won't be able to start a visible browser session from this environment. If you want that, say so and I'll run the smoke test now.