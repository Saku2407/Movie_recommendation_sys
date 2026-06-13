/* main.js — Cinéma frontend */

const API = "";
let allGenres  = [];
let currentMethod = "svd";
let moviesCache = [];

// ── Navigation ────────────────────────────────────────────────────────────────
function showPage(id) {
  document.querySelectorAll(".page").forEach(p => p.classList.remove("active"));
  document.querySelectorAll(".nav-btn").forEach(b => b.classList.remove("active"));
  document.getElementById("page-" + id).classList.add("active");
  const btn = document.getElementById("nav-" + id);
  if (btn) btn.classList.add("active");
  window.scrollTo({ top: 0, behavior: "smooth" });

  if (id === "browse" && moviesCache.length === 0) loadBrowse();
  if (id === "model")  loadModelInfo();
}

function setMethod(m) {
  currentMethod = m;
  document.getElementById("method-svd").classList.toggle("active", m === "svd");
  document.getElementById("method-cf").classList.toggle("active",  m === "cf");
}

// ── Star rating ───────────────────────────────────────────────────────────────
function stars(r) {
  const full  = Math.floor(r);
  const half  = r % 1 >= 0.5 ? 1 : 0;
  return "★".repeat(full) + (half ? "½" : "") + " " + r.toFixed(1);
}

// ── Movie card HTML ───────────────────────────────────────────────────────────
function movieCardHTML(m, showPredicted = false) {
  const initial = (m.title || "?")[0];
  const genre   = m.genres && m.genres.length ? m.genres[0] : "";
  const pred    = showPredicted && m.predicted_rating
    ? `<div class="card-predicted-badge">★ ${m.predicted_rating}</div>` : "";

  return `
    <div class="movie-card" onclick="openModal(${m.id})">
      <div class="card-poster" style="background:${m.poster_color}">
        ${pred}
        <div class="card-year-badge">${m.year}</div>
        <div class="card-poster-initial">${initial}</div>
      </div>
      <div class="card-title">${m.title}</div>
      <div class="card-meta">
        <span class="card-star">★</span>
        <span>${m.avg_rating.toFixed(1)}</span>
        <span>·</span>
        <span class="card-genre">${genre}</span>
      </div>
    </div>`;
}

// ── Hero mosaic ───────────────────────────────────────────────────────────────
function buildMosaic(movies) {
  const mosaic = document.getElementById("hero-mosaic");
  if (!mosaic) return;
  const picks = movies.slice(0, 12);
  mosaic.innerHTML = picks.map(m => `
    <div class="mosaic-card" style="background:${m.poster_color}" onclick="openModal(${m.id})">
      <div class="mosaic-title">${m.title}</div>
    </div>`).join("");
}

// ── Trending ──────────────────────────────────────────────────────────────────
async function loadTrending() {
  try {
    const res  = await fetch(`${API}/api/trending?n=16`);
    const data = await res.json();
    const row  = document.getElementById("trending-row");
    row.innerHTML = data.trending.map(m => movieCardHTML(m)).join("");
    buildMosaic(data.trending);
  } catch (e) {
    console.warn("Trending:", e);
  }
}

// ── Genres ────────────────────────────────────────────────────────────────────
async function loadGenres() {
  try {
    const res  = await fetch(`${API}/api/genres`);
    const data = await res.json();
    allGenres  = data.genres;

    // Home genre chips
    const homeChips = document.getElementById("genre-chips");
    homeChips.innerHTML = allGenres.map(g =>
      `<button class="chip" onclick="browseGenre('${g}')">${g}</button>`
    ).join("");

    // Browse filter chips
    const filterRow = document.getElementById("genre-filter-row");
    filterRow.innerHTML = `<button class="chip active" data-genre="" onclick="filterGenre(this,'')">All</button>` +
      allGenres.map(g =>
        `<button class="chip" data-genre="${g}" onclick="filterGenre(this,'${g}')">${g}</button>`
      ).join("");
  } catch (e) {
    console.warn("Genres:", e);
  }
}

function browseGenre(genre) {
  showPage("browse");
  setTimeout(() => {
    const btn = document.querySelector(`.chip[data-genre="${genre}"]`);
    if (btn) filterGenre(btn, genre);
  }, 100);
}

// ── Browse ────────────────────────────────────────────────────────────────────
async function loadBrowse() {
  document.getElementById("browse-loading").style.display = "flex";
  document.getElementById("browse-grid").style.display   = "none";
  try {
    const res  = await fetch(`${API}/api/movies?limit=200`);
    const data = await res.json();
    moviesCache = data.movies;
    renderBrowse(moviesCache);
  } catch (e) {
    console.warn("Browse:", e);
  } finally {
    document.getElementById("browse-loading").style.display = "none";
  }
}

function renderBrowse(movies) {
  const grid  = document.getElementById("browse-grid");
  const empty = document.getElementById("browse-empty");
  const count = document.getElementById("browse-count");
  if (!movies.length) {
    grid.style.display   = "none";
    empty.style.display  = "block";
    count.textContent    = "0 films";
    return;
  }
  grid.innerHTML       = movies.map(m => movieCardHTML(m)).join("");
  grid.style.display   = "grid";
  empty.style.display  = "none";
  count.textContent    = `${movies.length} film${movies.length !== 1 ? "s" : ""}`;
}

function filterGenre(btn, genre) {
  document.querySelectorAll("#genre-filter-row .chip").forEach(c => c.classList.remove("active"));
  btn.classList.add("active");
  const sort = document.getElementById("sort-select").value;

  let filtered = genre ? moviesCache.filter(m => m.genres.includes(genre)) : [...moviesCache];
  if (sort === "rating")   filtered.sort((a,b) => b.avg_rating - a.avg_rating);
  if (sort === "popular")  filtered.sort((a,b) => b.n_ratings - a.n_ratings);
  if (sort === "year")     filtered.sort((a,b) => b.year - a.year);
  if (sort === "title")    filtered.sort((a,b) => a.title.localeCompare(b.title));

  renderBrowse(filtered);
}

// ── Personalised recommendations ─────────────────────────────────────────────
async function loadPersonalRecs() {
  const uid     = parseInt(document.getElementById("user-id-input").value);
  const loading = document.getElementById("recs-loading");
  const result  = document.getElementById("recs-result");
  const meta    = document.getElementById("recs-meta");
  const grid    = document.getElementById("recs-grid");

  if (!uid || uid < 1 || uid > 500) {
    alert("Please enter a User ID between 1 and 500.");
    return;
  }

  loading.style.display = "flex";
  result.style.display  = "none";

  try {
    const endpoint = currentMethod === "svd"
      ? `/api/recommend/svd/${uid}?n=12`
      : `/api/recommend/cf/${uid}?n=12`;
    const res  = await fetch(`${API}${endpoint}`);
    const data = await res.json();

    if (data.error) { alert(data.error); return; }

    const recs = data.recommendations || [];
    meta.innerHTML = `
      User <strong>${uid}</strong> · ${data.method}
      ${data.n_seen !== undefined ? ` · Has rated <strong>${data.n_seen}</strong> films` : ""}
      · Showing top <strong>${recs.length}</strong> picks`;

    grid.innerHTML = recs.map(m => movieCardHTML(m, true)).join("");
    result.style.display = "block";
  } catch (e) {
    alert("Could not reach the API. Make sure the Flask server is running.");
  } finally {
    loading.style.display = "none";
  }
}

// ── Modal ─────────────────────────────────────────────────────────────────────
async function openModal(movieId) {
  const overlay = document.getElementById("modal-overlay");
  overlay.style.display = "flex";
  document.body.style.overflow = "hidden";

  // Clear & show overlay immediately
  document.getElementById("modal-title").textContent   = "Loading…";
  document.getElementById("modal-meta").textContent    = "";
  document.getElementById("modal-genres").innerHTML    = "";
  document.getElementById("modal-overview").textContent = "";
  document.getElementById("modal-stats").innerHTML     = "";
  document.getElementById("modal-similar").innerHTML   = "";

  try {
    const [movieRes, simRes] = await Promise.all([
      fetch(`${API}/api/movies/${movieId}`),
      fetch(`${API}/api/similar/${movieId}?n=6`),
    ]);
    const m   = await movieRes.json();
    const sim = await simRes.json();

    document.getElementById("modal-poster").style.background = m.poster_color;
    document.getElementById("modal-poster").textContent      = m.title[0];
    document.getElementById("modal-meta").textContent        = `${m.year} · Dir. ${m.director}`;
    document.getElementById("modal-title").textContent       = m.title;
    document.getElementById("modal-genres").innerHTML        =
      m.genres.map(g => `<span class="chip" style="cursor:default">${g}</span>`).join("");
    document.getElementById("modal-overview").textContent    = m.overview;
    document.getElementById("modal-stats").innerHTML         = `
      <div class="modal-stat"><strong>★ ${m.avg_rating.toFixed(1)}</strong><br>avg rating</div>
      <div class="modal-stat"><strong>${m.n_ratings.toLocaleString()}</strong><br>ratings</div>
      <div class="modal-stat"><strong>${m.year}</strong><br>year</div>`;
    document.getElementById("modal-similar").innerHTML =
      (sim.similar || []).map(s => movieCardHTML(s)).join("");
  } catch (e) {
    document.getElementById("modal-title").textContent = "Could not load movie.";
  }
}

function closeModal(e) {
  if (e.target === document.getElementById("modal-overlay")) closeModalDirect();
}
function closeModalDirect() {
  document.getElementById("modal-overlay").style.display = "none";
  document.body.style.overflow = "";
}

// ── Model info ────────────────────────────────────────────────────────────────
async function loadModelInfo() {
  try {
    const res  = await fetch(`${API}/api/model-info`);
    const data = await res.json();
    document.getElementById("model-stats").innerHTML = `
      <div class="stat-card"><div class="stat-val">${data.n_movies}</div><div class="stat-lbl">Movies</div></div>
      <div class="stat-card"><div class="stat-val">${data.n_users}</div><div class="stat-lbl">Users</div></div>
      <div class="stat-card"><div class="stat-val">${(data.n_ratings/1000).toFixed(1)}K</div><div class="stat-lbl">Ratings</div></div>
      <div class="stat-card"><div class="stat-val">${(data.sparsity*100).toFixed(0)}%</div><div class="stat-lbl">Matrix sparsity</div></div>
      <div class="stat-card"><div class="stat-val">${data.svd_rmse}</div><div class="stat-lbl">SVD RMSE</div></div>
      <div class="stat-card"><div class="stat-val">${data.cv_rmse}</div><div class="stat-lbl">CV RMSE (3-fold)</div></div>
      <div class="stat-card"><div class="stat-val">${data.svd_params?.n_factors}</div><div class="stat-lbl">Latent factors</div></div>
      <div class="stat-card"><div class="stat-val">${data.svd_params?.n_epochs}</div><div class="stat-lbl">SVD epochs</div></div>
    `;
  } catch (e) {
    console.warn("Model info:", e);
  }
}

// ── Keyboard: close modal on Escape ──────────────────────────────────────────
document.addEventListener("keydown", e => {
  if (e.key === "Escape") closeModalDirect();
});

// ── Init ──────────────────────────────────────────────────────────────────────
loadTrending();
loadGenres();
