# 🎬 Cinéma — Movie Recommendation System

A full-stack machine learning web application that recommends films using
three different recommendation algorithms, powered by a **Flask** REST API
and a cinematic dark-themed **HTML/CSS/JS** frontend.

---

## Project structure

```
movie-recsys/
│
├── backend/
│   ├── train_model.py       # Generates data, trains all 3 models, saves artefacts
│   └── app.py               # Flask REST API (7 endpoints + serves frontend)
│
├── frontend/
│   ├── index.html           # Single-page UI (4 views: Discover / For You / Browse / Model)
│   └── static/
│       ├── css/style.css    # Cinematic dark theme (Playfair Display + Barlow Condensed)
│       └── js/main.js       # API calls, routing, modal, genre filter, recommendations
│
├── data/
│   ├── movies.csv           # 200 synthetic movies (title, genres, director, year…)
│   └── ratings.csv          # ~25,000 synthetic ratings (user_id, movie_id, rating)
│
├── models/                  # Generated after running train_model.py
│   ├── svd_model.pkl        # Trained SVD model (Surprise)
│   ├── content_sim.pkl      # 200×200 cosine similarity matrix (content-based)
│   ├── user_sim.pkl         # 500×500 user similarity matrix (CF)
│   ├── user_item_matrix.pkl # Sparse user-item ratings matrix
│   ├── movies_lookup.json   # Movie metadata dictionary
│   ├── movie_ids.json       # Ordered list of movie IDs
│   └── metrics.json         # RMSE, MAE, CV scores, hyperparameters
│
├── notebooks/               # (optional) place Jupyter notebooks here
├── requirements.txt
└── README.md
```

---

## Quickstart

### 1 · Install dependencies

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2 · Train all three models

```bash
python backend/train_model.py
```

Output:
```
Movies: 200  |  Ratings: 25161  |  Sparsity: 74.8%
SVD  RMSE=0.6095  MAE=0.4902
CV   RMSE=0.6065 ± 0.0065
All artefacts saved to models/
```

### 3 · Start the server

```bash
python backend/app.py
```

Open **http://localhost:5000** in your browser.

---

## API reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/movies` | All movies (filter by `genre`, sort by `rating/year/title/popular`) |
| GET | `/api/movies/<id>` | Single movie by ID |
| GET | `/api/trending` | Top movies ranked by rating × log(popularity) |
| GET | `/api/genres` | All unique genre strings |
| GET | `/api/similar/<id>` | Content-based similar movies |
| GET | `/api/recommend/svd/<user_id>` | SVD personalised top-N |
| GET | `/api/recommend/cf/<user_id>` | User-based CF personalised top-N |
| GET | `/api/model-info` | Metrics, hyperparameters, dataset stats |

### POST `/api/recommend/svd/42` — example response

```json
{
  "user_id": 42,
  "method": "SVD Matrix Factorisation",
  "n_seen": 47,
  "recommendations": [
    {
      "id": 133,
      "title": "Burning Kingdom",
      "genres": ["Action", "Thriller"],
      "year": 2018,
      "director": "Villeneuve",
      "predicted_rating": 4.62,
      "poster_color": "#1a1a2e"
    }
  ]
}
```

---

## ML concepts covered

| Concept | Where it appears |
|---|---|
| TF-IDF vectorisation | Content-based: genres + director + year → feature vectors |
| Cosine similarity | Both content-based (movies) and user-based CF (users) |
| User-item matrix | Pivot table of ratings; sparsity ~75% is the core challenge |
| Collaborative filtering | User-based: k nearest users predict unseen ratings |
| Matrix factorisation | SVD decomposes R ≈ U × Σ × Vᵀ into latent factors |
| Latent factors | Hidden "taste axes" — e.g. "prefers slow-burn dramas" |
| Train/test split | 80/20, `random_state=42` for reproducibility |
| Cross-validation | 3-fold CV to estimate generalisation error |
| RMSE / MAE | Evaluation metrics for rating prediction quality |
| Cold-start problem | New users → fall back to content-based (no ratings needed) |

---

## Three recommendation approaches explained

### 1. Content-Based Filtering
Turns each movie's attributes (genres, director, year) into a TF-IDF vector.
Similarity between movies = cosine similarity between their vectors.
To find films similar to *Burning Kingdom*, find the closest vectors in the 200-dimensional space.

**Strength:** Works for brand-new users with no ratings.
**Weakness:** Can't recommend "outside the box" — only finds similar films, not surprising ones.

### 2. User-Based Collaborative Filtering
Builds a 500×200 user-item ratings matrix. Finds the K users most similar to you (cosine similarity between rating rows). Predicts your rating for an unseen movie as the similarity-weighted average of those users' ratings.

**Strength:** Captures complex taste patterns.
**Weakness:** Doesn't scale — pairwise similarity is O(n²) users.

### 3. SVD Matrix Factorisation (production standard)
Decomposes the ratings matrix: **R ≈ U × Σ × Vᵀ**
- **U**: user latent factor matrix (500 × k)
- **Σ**: diagonal scaling matrix
- **Vᵀ**: item latent factor matrix (k × 200)
- **k=50**: number of latent dimensions

Predicts your rating for a movie as the dot product of your user vector × the movie's item vector. Trained with SGD to minimise RMSE with L2 regularisation.

**Strength:** Best accuracy, scales well, captures subtle preferences.
**Weakness:** Cold-start problem — needs at least a few ratings per user.

---

## Key hyperparameters

| Parameter | Default | Effect |
|---|---|---|
| `n_factors` | 50 | Number of latent dimensions — higher = more expressive but slower |
| `n_epochs` | 30 | SGD training iterations — more = better fit (watch for overfitting) |
| `lr_all` | 0.005 | Learning rate — too high = diverges, too low = slow convergence |
| `reg_all` | 0.02 | L2 regularisation — higher = less overfitting, lower bias |
| `k` (CF) | 10 | Number of similar users in CF — higher = smoother but noisier |

---

## Frontend pages

| Page | What it shows |
|---|---|
| **Discover** | Trending films (ranked by weighted score) + genre chips |
| **For You** | Personalised recs — enter any User ID 1–500, switch SVD ↔ CF |
| **Browse** | Full 200-film catalogue with genre filter + sort controls |
| **Model** | Live metrics, algorithm explanations, ML pipeline diagram |

---

## Extending the project

- **Real data** — download [MovieLens 1M](https://grouplens.org/datasets/movielens/1m/) and replace the synthetic generator.
- **Hybrid model** — combine SVD score + content similarity as a weighted blend.
- **Item-based CF** — compare movie vectors instead of user vectors (more scalable).
- **Precision@K / Recall@K** — add ranking metrics beyond RMSE (see `surprise.model_selection`).
- **GridSearchCV** — tune `n_factors` and `reg_all` automatically:
  ```python
  from surprise.model_selection import GridSearchCV
  params = {"n_factors":[20,50,100], "reg_all":[0.01,0.02,0.05]}
  gs = GridSearchCV(SVD, params, measures=["rmse"], cv=3)
  gs.fit(data)
  ```
- **Deploy** — Railway, Render, or any Python PaaS. Set `debug=False` and add a `Procfile`:
  ```
  web: python backend/app.py
  ```

---

## Tech stack

| Layer | Technology |
|---|---|
| ML / data | Python · pandas · NumPy · scikit-learn · scikit-surprise |
| API | Flask · Flask-CORS · joblib |
| Frontend | HTML5 · CSS3 · Vanilla JavaScript |
| Fonts | Playfair Display · Barlow Condensed (Google Fonts) |

---

> **Note:** Uses synthetic data generated at training time — not real movie ratings.
> To use real data, replace `train_model.py`'s data generation section with a MovieLens loader.
