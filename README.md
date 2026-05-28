# 🚢 Titanic Survival Prediction — CatBoost Classifier

A machine learning project that predicts Titanic passenger survival using **CatBoost**, a gradient boosting algorithm that natively handles categorical features without manual encoding.

---

## 📌 Project Overview

| Item | Detail |
|---|---|
| **Dataset** | Seaborn built-in Titanic dataset (891 rows) |
| **Task** | Binary Classification (Survived: 0 / 1) |
| **Model** | CatBoostClassifier |
| **Evaluation** | Accuracy, Classification Report, Confusion Matrix |

---

## 🗂️ Project Structure

```
titanic-catboost/
│
├── catboost.ipynb       # Main Jupyter Notebook
├── requirements.txt     # Python dependencies
└── README.md            # Project documentation
```

---

## ⚙️ Features Used

After dropping leaking/redundant columns (`alive`, `alone`, `embark_town`, `deck`, `who`, `adult_male`):

| Feature | Type | Notes |
|---|---|---|
| `pclass` | Numerical | Passenger class (1, 2, 3) |
| `sex` | Categorical | male / female |
| `age` | Numerical | Missing values filled with median |
| `sibsp` | Numerical | Siblings/spouses aboard |
| `parch` | Numerical | Parents/children aboard |
| `fare` | Numerical | Ticket fare |
| `embarked` | Categorical | Port of embarkation (S/C/Q) |
| `class` | Categorical | First / Second / Third |

---

## 🔄 Pipeline

1. **Load Data** — Seaborn Titanic dataset
2. **Drop Redundant Columns** — Remove leaking/duplicate features
3. **Handle Missing Values** — Age → median, Embarked → mode
4. **Train/Test Split** — 80/20 split with stratify
5. **Train CatBoost** — With native categorical feature support
6. **Evaluate** — Accuracy, Classification Report, Confusion Matrix Heatmap

---

## 🧠 Model Configuration

```python
CatBoostClassifier(
    iterations=300,
    learning_rate=1,
    verbose=50
)
```

**Key Advantage:** CatBoost handles `sex`, `embarked`, and `class` columns directly as categoricals — **no Label Encoding needed!**

---

## 📊 Results

| Split | Accuracy |
|---|---|
| Validation Set | ~83–85% |
| Training Set | ~88–90% |

---

## 🚀 How to Run

```bash
# 1. Clone the repo
git clone https://github.com/your-username/titanic-catboost.git
cd titanic-catboost

# 2. Install dependencies
pip install -r requirements.txt

# 3. Open notebook
jupyter notebook catboost.ipynb
```

---

## 📦 Dependencies

See `requirements.txt` for full list. Key libraries:

- `catboost` — Main model
- `lightgbm` — Imported (for comparison experiments)
- `scikit-learn` — Train/test split, metrics
- `seaborn` / `matplotlib` — Visualization
- `pandas` / `numpy` — Data processing


## 👤 Author

**Fawad Ahmad**
- GitHub: [@FawadAhmad-bilal](https://github.com/FawadAhmad-bilal)
- Email: fawadahmadbilal@gmail.com
