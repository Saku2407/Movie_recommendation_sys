"""
train_model.py
--------------
Generates a realistic synthetic movie dataset, trains three recommendation
approaches (content-based, user-based CF, SVD matrix factorisation) and
saves all artefacts for the Flask API to load at startup.

Run:  python backend/train_model.py
"""

import os, json, random
import numpy as np
import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from surprise import SVD, Dataset, Reader
from surprise.model_selection import cross_validate, train_test_split
from surprise import accuracy

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# ── 1. Synthetic movie catalogue (200 movies) ─────────────────────────────────
GENRES = ["Action","Adventure","Animation","Comedy","Crime",
          "Documentary","Drama","Fantasy","Horror","Mystery",
          "Romance","Sci-Fi","Thriller","Western"]

DIRECTORS = ["Nolan","Spielberg","Tarantino","Scorsese","Kubrick",
             "Fincher","Villeneuve","Anderson","Coppola","Lynch",
             "Kurosawa","Bergman","Hitchcock","Chaplin","Lean"]

ADJECTIVES = ["Dark","Silent","Last","Lost","Hidden","Eternal","Broken",
              "Golden","Crimson","Forgotten","Hollow","Sacred","Electric",
              "Phantom","Burning","Distant","Infinite","Ancient","Twisted","Wild"]

NOUNS = ["City","Horizon","Dream","Echo","Storm","Ember","Kingdom",
         "Mirror","Signal","Shadow","River","Crown","Machine","Garden",
         "Labyrinth","Hour","Star","Vessel","Legacy","Code"]

def make_title():
    pattern = random.choice(["adj_noun","The_adj","noun_of_noun","adj_adj"])
    if pattern == "adj_noun":
        return f"{random.choice(ADJECTIVES)} {random.choice(NOUNS)}"
    elif pattern == "The_adj":
        return f"The {random.choice(ADJECTIVES)} {random.choice(NOUNS)}"
    elif pattern == "noun_of_noun":
        return f"{random.choice(NOUNS)} of {random.choice(NOUNS)}"
    else:
        return f"{random.choice(ADJECTIVES)} {random.choice(ADJECTIVES)} {random.choice(NOUNS)}"

POSTERS = [
    "#1a1a2e","#16213e","#0f3460","#533483","#2b2d42",
    "#8d0801","#264653","#2a9d8f","#4a4e69","#22223b",
    "#3d405b","#81b29a","#f4a261","#e76f51","#023e8a",
    "#1b4332","#6d2b3d","#4a1942","#7b2d8b","#1f2041"
]

N_MOVIES = 200
titles_seen = set()
movies = []
for i in range(1, N_MOVIES + 1):
    title = make_title()
    while title in titles_seen:
        title = make_title()
    titles_seen.add(title)

    n_genres = random.randint(1, 3)
    movie_genres = random.sample(GENRES, n_genres)
    year = random.randint(1975, 2024)
    director = random.choice(DIRECTORS)
    rating_bias = round(random.uniform(2.5, 4.8), 1)
    n_votes = random.randint(50, 2000)
    poster_color = random.choice(POSTERS)
    overview = (f"A {movie_genres[0].lower()} film directed by {director}, "
                f"released in {year}. " +
                random.choice([
                    "A gripping tale of survival and redemption.",
                    "An intimate portrait of human connection.",
                    "A visually stunning journey into the unknown.",
                    "A sharp and darkly comic thriller.",
                    "A sweeping epic that spans decades.",
                    "A quiet, devastating character study.",
                    "An explosive action-adventure with heart.",
                    "A mystery that keeps you guessing.",
                ]))
    movies.append({
        "movie_id": i,
        "title": title,
        "genres": "|".join(movie_genres),
        "year": year,
        "director": director,
        "avg_rating": rating_bias,
        "n_ratings": n_votes,
        "poster_color": poster_color,
        "overview": overview,
    })

movies_df = pd.DataFrame(movies)

# ── 2. Synthetic ratings (500 users) ─────────────────────────────────────────
N_USERS = 500
ratings = []
for user_id in range(1, N_USERS + 1):
    # Each user has genre preferences
    fav_genres = random.sample(GENRES, random.randint(2, 5))
    n_rated = random.randint(20, 80)
    rated_movies = random.sample(list(movies_df["movie_id"]), n_rated)

    for mid in rated_movies:
        movie = movies_df[movies_df["movie_id"] == mid].iloc[0]
        movie_genre_list = movie["genres"].split("|")
        overlap = len(set(fav_genres) & set(movie_genre_list))
        base = movie["avg_rating"]
        noise = np.random.normal(0, 0.6)
        genre_boost = overlap * 0.3
        rating = round(np.clip(base + noise + genre_boost, 1, 5) * 2) / 2
        ratings.append({"user_id": user_id, "movie_id": mid, "rating": rating})

ratings_df = pd.DataFrame(ratings)
sparsity = 1 - len(ratings_df) / (N_USERS * N_MOVIES)

os.makedirs("data", exist_ok=True)
movies_df.to_csv("data/movies.csv", index=False)
ratings_df.to_csv("data/ratings.csv", index=False)
print(f"Movies: {len(movies_df)}  |  Ratings: {len(ratings_df)}  |  Sparsity: {sparsity:.1%}")

# ── 3. Content-based filtering ────────────────────────────────────────────────
movies_df["features"] = (
    movies_df["genres"].str.replace("|", " ") + " " +
    movies_df["director"] + " " +
    movies_df["year"].astype(str)
)
tfidf = TfidfVectorizer(stop_words="english")
tfidf_matrix = tfidf.fit_transform(movies_df["features"])
content_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# ── 4. User-based collaborative filtering ────────────────────────────────────
user_item = ratings_df.pivot_table(
    index="user_id", columns="movie_id", values="rating"
).fillna(0)
user_sim = cosine_similarity(user_item)
user_sim_df = pd.DataFrame(user_sim,
                            index=user_item.index,
                            columns=user_item.index)

# ── 5. SVD matrix factorisation (Surprise) ───────────────────────────────────
reader = Reader(rating_scale=(1, 5))
data   = Dataset.load_from_df(ratings_df[["user_id","movie_id","rating"]], reader)
trainset, testset = train_test_split(data, test_size=0.2, random_state=SEED)
 
svd = SVD(n_factors=50, n_epochs=30, lr_all=0.005, reg_all=0.02, random_state=SEED)
svd.fit(trainset)
predictions = svd.test(testset)
rmse = accuracy.rmse(predictions, verbose=False)
mae  = accuracy.mae(predictions, verbose=False)
print(f"SVD  RMSE={rmse:.4f}  MAE={mae:.4f}")

# Cross-validate
cv = cross_validate(SVD(n_factors=50, n_epochs=20, random_state=SEED),
                    data, measures=["RMSE","MAE"], cv=3, verbose=False)
print(f"CV   RMSE={cv['test_rmse'].mean():.4f} ± {cv['test_rmse'].std():.4f}")

# Full trainset for serving
full_trainset = data.build_full_trainset()
svd_full = SVD(n_factors=50, n_epochs=30, lr_all=0.005, reg_all=0.02, random_state=SEED)
svd_full.fit(full_trainset)

# ── 6. Save artefacts ─────────────────────────────────────────────────────────
os.makedirs("models", exist_ok=True)
joblib.dump(svd_full,      "models/svd_model.pkl")
joblib.dump(content_sim,   "models/content_sim.pkl")
joblib.dump(user_sim_df,   "models/user_sim.pkl")
joblib.dump(user_item,     "models/user_item_matrix.pkl")

# Save lightweight lookup dicts (JSON-serialisable)
movies_lookup = movies_df.set_index("movie_id").to_dict(orient="index")
with open("models/movies_lookup.json", "w") as f:
    json.dump(movies_lookup, f)

movie_ids = list(movies_df["movie_id"])
with open("models/movie_ids.json", "w") as f:
    json.dump(movie_ids, f)

metrics = {
    "svd_rmse":  round(rmse, 4),
    "svd_mae":   round(mae, 4),
    "cv_rmse":   round(float(cv["test_rmse"].mean()), 4),
    "cv_rmse_std": round(float(cv["test_rmse"].std()), 4),
    "n_users":   N_USERS,
    "n_movies":  N_MOVIES,
    "n_ratings": len(ratings_df),
    "sparsity":  round(sparsity, 4),
    "svd_params": {"n_factors":50,"n_epochs":30,"lr_all":0.005,"reg_all":0.02},
}
with open("models/metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

print("All artefacts saved to models/")
