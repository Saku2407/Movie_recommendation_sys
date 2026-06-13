"""
app.py  —  Flask REST API for Movie Recommendation System
Run:  python backend/app.py
"""
 
import os, sys, json
import numpy as np
import pandas as pd
import joblib
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")
FRONTEND   = os.path.join(BASE_DIR, "frontend")

app = Flask(__name__, static_folder=FRONTEND)
CORS(app)

# ── Load artefacts ────────────────────────────────────────────────────────────
try:
    svd         = joblib.load(os.path.join(MODELS_DIR, "svd_model.pkl"))
    content_sim = joblib.load(os.path.join(MODELS_DIR, "content_sim.pkl"))
    user_sim    = joblib.load(os.path.join(MODELS_DIR, "user_sim.pkl"))
    user_item   = joblib.load(os.path.join(MODELS_DIR, "user_item_matrix.pkl"))

    with open(os.path.join(MODELS_DIR, "movies_lookup.json")) as f:
        movies_raw = json.load(f)
    movies_lookup = {int(k): v for k, v in movies_raw.items()}

    with open(os.path.join(MODELS_DIR, "movie_ids.json")) as f:
        movie_ids = json.load(f)

    with open(os.path.join(MODELS_DIR, "metrics.json")) as f:
        metrics = json.load(f)

    # Build a DataFrame from the lookup for content similarity indexing
    movies_df = pd.DataFrame.from_dict(movies_lookup, orient="index")
    movies_df.index = movies_df.index.astype(int)
    movies_df["movie_id"] = movies_df.index

    print(f"✓ Loaded {len(movies_lookup)} movies, {len(user_sim)} users")
except Exception as e:
    print(f"✗ Could not load models: {e}")
    print("  Run: python backend/train_model.py  first")
    sys.exit(1)


# ── Helpers ───────────────────────────────────────────────────────────────────
def movie_to_dict(mid):
    m = movies_lookup.get(int(mid), {})
    return {
        "id":           int(mid),
        "title":        m.get("title", "Unknown"),
        "genres":       m.get("genres", "").split("|"),
        "year":         m.get("year", 0),
        "director":     m.get("director", ""),
        "avg_rating":   m.get("avg_rating", 3.0),
        "n_ratings":    m.get("n_ratings", 0),
        "poster_color": m.get("poster_color", "#333"),
        "overview":     m.get("overview", ""),
    }


# ── API: all movies ───────────────────────────────────────────────────────────
@app.route("/api/movies", methods=["GET"])
def get_movies():
    genre  = request.args.get("genre", "")
    sort   = request.args.get("sort", "rating")   # rating | year | title
    limit  = int(request.args.get("limit", 50))
    offset = int(request.args.get("offset", 0))

    result = [movie_to_dict(mid) for mid in movie_ids]
    if genre:
        result = [m for m in result if genre in m["genres"]]
    if sort == "rating":
        result.sort(key=lambda x: x["avg_rating"], reverse=True)
    elif sort == "year":
        result.sort(key=lambda x: x["year"], reverse=True)
    elif sort == "title":
        result.sort(key=lambda x: x["title"])
    elif sort == "popular":
        result.sort(key=lambda x: x["n_ratings"], reverse=True)

    return jsonify({
        "movies": result[offset:offset+limit],
        "total":  len(result),
    })


# ── API: single movie ─────────────────────────────────────────────────────────
@app.route("/api/movies/<int:movie_id>", methods=["GET"])
def get_movie(movie_id):
    if movie_id not in movies_lookup:
        return jsonify({"error": "Movie not found"}), 404
    return jsonify(movie_to_dict(movie_id))


# ── API: content-based similar movies ────────────────────────────────────────
@app.route("/api/similar/<int:movie_id>", methods=["GET"])
def similar_movies(movie_id):
    n = int(request.args.get("n", 8))
    try:
        idx = movies_df.index.get_loc(movie_id)
        scores = list(enumerate(content_sim[idx]))
        scores.sort(key=lambda x: x[1], reverse=True)
        # Skip index 0 (itself)
        top = [i for i, _ in scores[1:n+1]]
        result = [movie_to_dict(movies_df.iloc[i]["movie_id"]) for i in top]
        return jsonify({"similar": result, "method": "content-based"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ── API: SVD personalised recommendations ────────────────────────────────────
@app.route("/api/recommend/svd/<int:user_id>", methods=["GET"])
def recommend_svd(user_id):
    n = int(request.args.get("n", 10))
    if user_id not in user_item.index:
        return jsonify({"error": f"User {user_id} not found. Valid range: 1–{len(user_sim)}"}), 404

    seen_movies = set(user_item.loc[user_id][user_item.loc[user_id] > 0].index)
    unseen = [mid for mid in movie_ids if mid not in seen_movies]

    preds = [(mid, svd.predict(user_id, mid).est) for mid in unseen]
    preds.sort(key=lambda x: x[1], reverse=True)

    result = []
    for mid, score in preds[:n]:
        m = movie_to_dict(mid)
        m["predicted_rating"] = round(score, 2)
        result.append(m)

    return jsonify({
        "recommendations": result,
        "method": "SVD Matrix Factorisation",
        "user_id": user_id,
        "n_seen": len(seen_movies),
    })


# ── API: user-based CF recommendations ───────────────────────────────────────
@app.route("/api/recommend/cf/<int:user_id>", methods=["GET"])
def recommend_cf(user_id):
    n = int(request.args.get("n", 10))
    k = int(request.args.get("k", 10))   # similar users to consider

    if user_id not in user_sim.index:
        return jsonify({"error": f"User {user_id} not found"}), 404

    # Get k most similar users
    sim_scores = user_sim[user_id].drop(user_id).nlargest(k)
    similar_users = list(sim_scores.index)

    seen = set(user_item.loc[user_id][user_item.loc[user_id] > 0].index)

    # Weighted average of similar users' ratings
    scores = {}
    weights = {}
    for su in similar_users:
        w = sim_scores[su]
        if w <= 0:
            continue
        rated = user_item.loc[su][user_item.loc[su] > 0]
        for mid, r in rated.items():
            if mid not in seen:
                scores[mid]  = scores.get(mid, 0)  + w * r
                weights[mid] = weights.get(mid, 0) + w

    pred = {mid: scores[mid] / weights[mid] for mid in scores if weights[mid] > 0}
    top = sorted(pred.items(), key=lambda x: x[1], reverse=True)[:n]

    result = []
    for mid, score in top:
        m = movie_to_dict(mid)
        m["predicted_rating"] = round(score, 2)
        result.append(m)

    return jsonify({
        "recommendations": result,
        "method": "User-Based Collaborative Filtering",
        "user_id": user_id,
        "similar_users_used": k,
    })


# ── API: top-rated / trending ─────────────────────────────────────────────────
@app.route("/api/trending", methods=["GET"])
def trending():
    n = int(request.args.get("n", 12))
    result = sorted(
        [movie_to_dict(mid) for mid in movie_ids],
        key=lambda x: x["avg_rating"] * np.log1p(x["n_ratings"]),
        reverse=True
    )[:n]
    return jsonify({"trending": result})


# ── API: genres list ──────────────────────────────────────────────────────────
@app.route("/api/genres", methods=["GET"])
def genres():
    all_genres = set()
    for m in movies_lookup.values():
        for g in m.get("genres","").split("|"):
            if g:
                all_genres.add(g)
    return jsonify({"genres": sorted(all_genres)})


# ── API: model info ───────────────────────────────────────────────────────────
@app.route("/api/model-info", methods=["GET"])
def model_info():
    return jsonify(metrics)


# ── Serve frontend ─────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return send_from_directory(FRONTEND, "index.html")

@app.route("/<path:path>")
def static_files(path):
    return send_from_directory(FRONTEND, path)


if __name__ == "__main__":
    print("\n🎬  Movie Recommendation System API")
    print("    Open → http://localhost:5000\n")
    app.run(debug=True, port=5000)
