# SQL Injection Attack Detection

A desktop application that detects SQL injection patterns using **machine learning** and optional **Google Gemini AI** analysis. The project combines text mining (bag-of-words features), classical ML, and a neural network inside a simple Tkinter GUI.

> **Educational use only.** Use this project to learn about web security and ML-based detection. Do not use it to attack systems you do not own or have permission to test.

## Features

- Upload and preprocess SQL injection datasets (CSV)
- Train **Logistic Regression** and a **feed-forward Neural Network**
- Visualize label distribution, confusion matrices, and accuracy comparison
- Run detection on test data with both ML and Gemini AI results
- Jupyter notebook included for exploratory analysis

## Project Structure

```
.
├── main.py                                      # Desktop GUI application
├── sql injection dectection using ML.ipynb      # Jupyter notebook version
├── Datasets/
│   ├── sqli.csv                                 # Training dataset (~4200 samples)
│   └── test.csv                                 # Sample test inputs
├── requirements.txt
├── run.bat                                      # One-click setup + run (Windows)
├── .env.example                                 # API key template
└── README.md
```

## Requirements

- Python **3.10–3.12** (TensorFlow does not support Python 3.13+ yet)
- Windows, macOS, or Linux
- Optional: [Google Gemini API key](https://aistudio.google.com/apikey) for generative-AI analysis

## Quick Start (Windows)

1. **Clone or download** this repository.
2. Double-click **`run.bat`**  
   - Creates a virtual environment  
   - Installs dependencies  
   - Copies `.env.example` to `.env` if needed  
   - Launches the GUI

Or run manually:

```powershell
cd "path\to\sql-injection-detection"
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
python main.py
```

## Quick Start (macOS / Linux)

```bash
cd path/to/sql-injection-detection
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
python main.py
```

## Gemini API Key (Optional)

1. Copy `.env.example` to `.env`
2. Add your key:

```env
GEMINI_API_KEY=your_gemini_api_key_here
```

Without an API key, ML detection still works; only the Gemini analysis step is skipped.

## How to Use the GUI

Follow this order:

1. **Upload Dataset** — select `Datasets/sqli.csv` (UTF-16 encoded CSV with `Sentence` and `Label` columns)
2. **Preprocess** — vectorizes text and splits train/test data; shows a label distribution chart
3. **Logistic Regression** — trains and evaluates the baseline classifier
4. **Neural Network** — trains the Keras model (takes ~1–2 minutes)
5. **Compare Accuracy** — bar chart comparing both models
6. **Detect (Test Data)** — select `Datasets/test.csv` or your own file with a `Sentence` column

### Dataset Format

**Training CSV** (`sqli.csv`):

| Sentence | Label |
|----------|-------|
| `a' or 1 = 1--` | 0 |
| `hello world` | 1 |

- `Label = 0` → SQL injection
- `Label = 1` → Benign / no attack

**Test CSV** (`test.csv`):

| Sentence |
|----------|
| `admin" ) or "1" = "1"--` |

## Jupyter Notebook

Open `sql injection dectection using ML.ipynb` in Jupyter Lab or VS Code to explore the same workflow interactively:

```bash
pip install jupyter
jupyter lab
```

## Publish to GitHub

```bash
git init
git add .
git commit -m "Initial commit: SQL injection detection with ML and Gemini AI"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/sql-injection-detection.git
git push -u origin main
```

### Before pushing

- Never commit `.env` or API keys (already listed in `.gitignore`)
- Consider renaming the folder to `sql-injection-detection` (no spaces) for a cleaner repo URL
- If an API key was previously committed, rotate it in Google AI Studio immediately

## Models & Approach

1. **Text vectorization** — `CountVectorizer` with English stop words
2. **Logistic Regression** — fast baseline classifier
3. **Neural Network** — dense layers with batch normalization and dropout
4. **Gemini AI** — optional second opinion on suspicious input strings

Typical accuracy on the included dataset is **~93%** (logistic regression) and **~97%** (neural network), depending on your environment.

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `No module named 'keras'` | Run `pip install -r requirements.txt` (TensorFlow includes Keras) |
| NLTK errors | The app auto-downloads required NLTK data on first run |
| CSV encoding error | Training data uses UTF-16; test files may use UTF-8 (both are handled) |
| Gemini returns an error | Check `GEMINI_API_KEY` in `.env` and your internet connection |
| GUI does not open | Ensure Tkinter is installed (`python -m tkinter` to test) |

## License

MIT License — see [LICENSE](LICENSE).

## Disclaimer

This tool is for **security education and research** on datasets you are allowed to use. Unauthorized testing of live applications is illegal and unethical.
