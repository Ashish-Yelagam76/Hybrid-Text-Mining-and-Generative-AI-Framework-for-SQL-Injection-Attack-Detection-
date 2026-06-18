"""
Hybrid Text Mining and Generative-AI Framework for SQL Injection Attack Detection.

Desktop GUI for training ML models and detecting SQL injection patterns in text.
"""

from __future__ import annotations

import os
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import nltk
import numpy as np
import pandas as pd
import requests
import seaborn as sns
import tkinter as tk
from keras import layers
from keras.models import Sequential
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from tkinter import END, Button, Label, Scrollbar, Text, filedialog, messagebox

warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).resolve().parent
DATASETS_DIR = BASE_DIR / "Datasets"
DEFAULT_DATASET = DATASETS_DIR / "sqli.csv"
DEFAULT_TEST_DATASET = DATASETS_DIR / "test.csv"

# Optional Gemini API key for generative-AI analysis (never commit real keys).
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

# Global state used by the GUI workflow.
dataset: pd.DataFrame | None = None
vectorizer: CountVectorizer | None = None
model: Sequential | None = None
clf: LogisticRegression | None = None

X_train = X_test = y_train = y_test = None
accuracy = 0.0
accuracy1 = 0.0


def ensure_nltk_data() -> None:
    """Download NLTK resources required for text preprocessing."""
    for resource in ("punkt", "stopwords", "wordnet", "omw-1.4"):
        try:
            nltk.data.find(
                f"corpora/{resource}" if resource != "punkt" else f"tokenizers/{resource}"
            )
        except LookupError:
            nltk.download(resource, quiet=True)


def insert_message(widget: Text, message: str) -> None:
    widget.insert(END, message + "\n")
    widget.see(END)


def show_error(title: str, message: str) -> None:
    messagebox.showerror(title, message)


def clean_dataset(frame: pd.DataFrame) -> pd.DataFrame:
    """Remove invalid rows and normalize text for vectorization."""
    cleaned = frame.copy()
    cleaned["Sentence"] = cleaned["Sentence"].fillna("").astype(str).str.strip()
    cleaned["Label"] = pd.to_numeric(cleaned["Label"], errors="coerce")
    cleaned = cleaned[cleaned["Sentence"] != ""]
    cleaned = cleaned.dropna(subset=["Label"])
    cleaned["Label"] = cleaned["Label"].astype(int)
    return cleaned.reset_index(drop=True)


def sentences_to_documents(series: pd.Series) -> list[str]:
    """Convert sentence column to strings sklearn can tokenize."""
    return series.fillna("").astype(str).str.strip().tolist()


def require_preprocessed(text_widget: Text) -> bool:
    if X_train is None or vectorizer is None:
        show_error("Preprocessing required", "Upload a dataset and run preprocessing first.")
        return False
    return True


def upload(text_widget: Text) -> None:
    global dataset

    initial_dir = str(DATASETS_DIR if DATASETS_DIR.exists() else BASE_DIR)
    filename = filedialog.askopenfilename(
        initialdir=initial_dir,
        title="Select SQL injection dataset",
        filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
    )
    if not filename:
        return

    text_widget.delete("1.0", END)
    try:
        dataset = pd.read_csv(filename, encoding="utf-16")
    except UnicodeError:
        dataset = pd.read_csv(filename)

    if not {"Sentence", "Label"}.issubset(dataset.columns):
        dataset = None
        show_error(
            "Invalid dataset",
            "CSV must contain 'Sentence' and 'Label' columns.",
        )
        return

    original_count = len(dataset)
    dataset = clean_dataset(dataset)
    removed = original_count - len(dataset)

    insert_message(text_widget, f"Loaded dataset: {filename}")
    insert_message(text_widget, f"Records: {len(dataset)}")
    if removed:
        insert_message(text_widget, f"Removed {removed} empty/invalid rows during cleanup.")
    insert_message(text_widget, str(dataset.head()))


def preprocess(text_widget: Text) -> None:
    global dataset, vectorizer, X_train, X_test, y_train, y_test

    if dataset is None:
        show_error("No dataset", "Upload a dataset before preprocessing.")
        return

    dataset = clean_dataset(dataset)
    text_widget.delete("1.0", END)

    sns.set(style="darkgrid")
    plt.figure(figsize=(8, 6))
    ax = sns.countplot(data=dataset, x="Label", palette="Set2")
    for patch in ax.patches:
        ax.annotate(
            f"{int(patch.get_height())}",
            (patch.get_x() + patch.get_width() / 2, patch.get_height()),
            ha="center",
            va="bottom",
        )
    plt.xlabel("Label")
    plt.ylabel("Count")
    plt.title("SQL Injection vs Benign Samples")
    plt.tight_layout()
    plt.show()

    vectorizer = CountVectorizer(
        min_df=2,
        max_df=0.7,
        stop_words=stopwords.words("english"),
    )
    posts = vectorizer.fit_transform(sentences_to_documents(dataset["Sentence"])).toarray()
    transformed_posts = pd.DataFrame(posts)
    processed = pd.concat([dataset.reset_index(drop=True), transformed_posts], axis=1)

    X = processed[processed.columns[2:]]
    y = processed["Label"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    insert_message(text_widget, f"Training samples: {X_train.shape}")
    insert_message(text_widget, f"Testing samples: {X_test.shape}")
    insert_message(text_widget, f"Feature count: {X_train.shape[1]}")


def run_logistic_regression(text_widget: Text) -> None:
    global clf, accuracy

    if not require_preprocessed(text_widget):
        return

    text_widget.delete("1.0", END)

    try:
        clf = LogisticRegression(random_state=0, max_iter=1000)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred) * 100
        report = classification_report(y_test, y_pred)

        insert_message(text_widget, f"Logistic Regression accuracy: {accuracy:.2f}%")
        insert_message(text_widget, f"Classification report:\n{report}")

        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title("Logistic Regression Confusion Matrix")
        plt.tight_layout()
        plt.show()
    except Exception as exc:
        show_error("Logistic Regression failed", str(exc))
        insert_message(text_widget, f"Error: {exc}")


def simple_neural_network(text_widget: Text) -> None:
    global model, accuracy1

    if not require_preprocessed(text_widget):
        return

    text_widget.delete("1.0", END)

    input_dim = X_train.shape[1]
    model = Sequential(
        [
            layers.Dense(20, input_dim=input_dim, activation="relu"),
            layers.Dense(10, activation="tanh"),
            layers.Dense(1024, activation="relu"),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(1, activation="sigmoid"),
        ]
    )
    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
    model.summary()

    model.fit(
        X_train,
        y_train,
        epochs=10,
        verbose=True,
        validation_data=(X_test, y_test),
        batch_size=15,
    )

    y_pred = (model.predict(X_test) > 0.5).astype(int).ravel()
    accuracy1 = accuracy_score(y_test, y_pred) * 100
    report = classification_report(y_test, y_pred)

    insert_message(text_widget, f"Neural network accuracy: {accuracy1:.2f}%")
    insert_message(text_widget, f"Classification report:\n{report}")

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Neural Network Confusion Matrix")
    plt.tight_layout()
    plt.show()


def label_to_text(label: int) -> str:
    return "SQL injection detected" if label == 0 else "Benign (no attack)"


def analyze_with_gemini(error_text: str) -> dict:
    """Use Google Gemini to classify whether text looks like SQL injection."""
    if not GEMINI_API_KEY:
        return {
            "error": "GEMINI_API_KEY is not set. Add it to your environment or .env file."
        }

    url = (
        "https://generativelanguage.googleapis.com/v1/models/"
        f"gemini-1.5-flash:generateContent?key={GEMINI_API_KEY}"
    )
    prompt = (
        "Analyze the following input and determine if it indicates an SQL injection attack. "
        'Respond with only "yes" or "no".\n\n'
        f"Input:\n{error_text}"
    )
    payload = {"contents": [{"role": "user", "parts": [{"text": prompt}]}]}

    try:
        response = requests.post(url, json=payload, timeout=30)
        response.raise_for_status()
        text_response = (
            response.json()
            .get("candidates", [{}])[0]
            .get("content", {})
            .get("parts", [{}])[0]
            .get("text", "")
            .strip()
            .lower()
        )
        if text_response == "yes":
            return {"is_sql_injection": "yes"}
        if text_response == "no":
            return {"is_sql_injection": "no"}
        return {"error": f"Unexpected Gemini response: {text_response!r}"}
    except requests.RequestException as exc:
        return {"error": f"API request failed: {exc}"}


def predict(text_widget: Text) -> None:
    if vectorizer is None:
        show_error("Preprocessing required", "Run preprocessing before detection.")
        return
    if model is None:
        show_error("Model required", "Train the neural network before running detection.")
        return

    initial_dir = str(DATASETS_DIR if DATASETS_DIR.exists() else BASE_DIR)
    test_path = filedialog.askopenfilename(
        initialdir=initial_dir,
        title="Select test CSV",
        filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
    )
    if not test_path:
        return

    text_widget.delete("1.0", END)

    try:
        test_df = pd.read_csv(test_path)
    except UnicodeError:
        test_df = pd.read_csv(test_path, encoding="utf-16")

    if "Sentence" not in test_df.columns:
        show_error("Invalid test file", "Test CSV must contain a 'Sentence' column.")
        return

    features = vectorizer.transform(sentences_to_documents(test_df["Sentence"])).toarray()
    predictions = (model.predict(features) > 0.5).astype(int).ravel()

    insert_message(text_widget, f"Analyzing {len(test_df)} samples from {test_path}\n")

    for index, row in test_df.iterrows():
        sentence = str(row["Sentence"])
        ml_label = int(predictions[index])
        ml_result = label_to_text(ml_label)
        gemini_result = analyze_with_gemini(sentence)
        insert_message(
            text_widget,
            f"Input: {sentence}\n"
            f"  ML model   -> {ml_result}\n"
            f"  Gemini AI  -> {gemini_result}\n",
        )


def graph() -> None:
    if accuracy == 0.0 and accuracy1 == 0.0:
        show_error("No results", "Train at least one model before viewing the comparison graph.")
        return

    plt.figure(figsize=(8, 5))
    plt.bar(
        ["Logistic Regression", "Neural Network"],
        [accuracy, accuracy1],
        color=["#e74c3c", "#2ecc71"],
    )
    plt.xlabel("Model")
    plt.ylabel("Accuracy (%)")
    plt.title("Model Accuracy Comparison")
    plt.ylim(0, 100)
    plt.tight_layout()
    plt.show()


def build_gui() -> tk.Tk:
    root = tk.Tk()
    root.title("SQL Injection Attack Detection")
    root.geometry("1050x700")
    root.minsize(900, 600)

    title_font = ("Segoe UI", 15, "bold")
    button_font = ("Segoe UI", 11, "bold")

    title = Label(
        root,
        text="Hybrid Text Mining and Generative-AI Framework\nfor SQL Injection Attack Detection",
        justify="center",
        bg="#f8f4ff",
        fg="#5b2c8f",
        font=title_font,
        padx=12,
        pady=12,
    )
    title.pack(fill="x", padx=12, pady=(12, 8))

    button_frame = tk.Frame(root)
    button_frame.pack(fill="x", padx=12, pady=8)

    output = Text(root, height=24, width=120, font=("Consolas", 10))
    scroll = Scrollbar(output, command=output.yview)
    output.configure(yscrollcommand=scroll.set)
    output.pack(side="left", fill="both", expand=True, padx=(12, 0), pady=(0, 12))
    scroll.pack(side="right", fill="y", padx=(0, 12), pady=(0, 12))

    buttons = [
        ("Upload Dataset", lambda: upload(output)),
        ("Preprocess", lambda: preprocess(output)),
        ("Logistic Regression", lambda: run_logistic_regression(output)),
        ("Neural Network", lambda: simple_neural_network(output)),
        ("Compare Accuracy", graph),
        ("Detect (Test Data)", lambda: predict(output)),
        ("Exit", root.destroy),
    ]

    for index, (label, command) in enumerate(buttons):
        Button(
            button_frame,
            text=label,
            command=command,
            font=button_font,
            width=18,
            pady=4,
        ).grid(row=index // 4, column=index % 4, padx=6, pady=6)

    if DEFAULT_DATASET.exists():
        insert_message(
            output,
            f"Ready. Default training dataset: {DEFAULT_DATASET.name}\n"
            f"Default test dataset: {DEFAULT_TEST_DATASET.name}\n"
            "Workflow: Upload -> Preprocess -> Train models -> Detect\n",
        )
    if not GEMINI_API_KEY:
        insert_message(
            output,
            "Note: Set GEMINI_API_KEY in a .env file for Gemini AI analysis.\n",
        )

    return root


def main() -> None:
    try:
        from dotenv import load_dotenv

        load_dotenv(BASE_DIR / ".env")
        global GEMINI_API_KEY
        GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
    except ImportError:
        pass

    ensure_nltk_data()
    root = build_gui()
    root.mainloop()


if __name__ == "__main__":
    main()
