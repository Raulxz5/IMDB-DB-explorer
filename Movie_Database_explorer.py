
import argparse
import sqlite3
import sys
import time
from pathlib import Path

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.gridspec import GridSpec

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

# ── Paths ─────────────────────────────────────────────────────────────────────
OUT_DIR = Path("imdb_output")
DB_PATH = OUT_DIR / "imdb.db"

# ── Palette ───────────────────────────────────────────────────────────────────
PALETTE = [
    "#4361EE", "#F72585", "#4CC9F0", "#7209B7", "#F77F00",
    "#3A0CA3", "#560BAD", "#06D6A0", "#FCBF49", "#EF233C",
    "#2EC4B6", "#E9C46A", "#E76F51", "#264653", "#A8DADC",
]
BG      = "#0a0a14"
BG_AX   = "#0f0f1a"
FG      = "white"
MUTED   = "#888888"
GRID    = "#1e1e2e"


# ══════════════════════════════════════════════════════════════════════════════
#  1. DATABASE LOADING
# ══════════════════════════════════════════════════════════════════════════════

def _read_tsv_gz(path: Path, usecols=None, dtype=str, chunksize=None):
    """Read an IMDb .tsv.gz file, replacing \\N with NaN."""
    kwargs = dict(
        sep="\t",
        na_values=r"\N",
        low_memory=False,
        dtype=dtype,
    )
    if usecols:
        kwargs["usecols"] = usecols
    if chunksize:
        kwargs["chunksize"] = chunksize
    return pd.read_csv(path, **kwargs)


def _progress(label: str, iterable, total=None):
    if HAS_TQDM:
        return tqdm(iterable, desc=f"  {label}", total=total, unit=" chunks")
    print(f"  {label}...", end=" ", flush=True)
    return iterable


def load_title_basics(conn: sqlite3.Connection, data_dir: Path):
    """titles — filtered to movies and tvMovies only."""
    print("\n📂  Loading title.basics  (filtering to movies)...")
    path = data_dir / "title.basics.tsv.gz"
    CHUNK = 500_000
    reader = _read_tsv_gz(path, chunksize=CHUNK)
    first = True
    rows_written = 0
    for chunk in _progress("title.basics", reader):
        chunk = chunk[chunk["titleType"].isin(["movie", "tvMovie"])].copy()
        chunk = chunk[[
            "tconst", "titleType", "primaryTitle", "originalTitle",
            "startYear", "runtimeMinutes", "genres",
        ]]
        chunk["startYear"]       = pd.to_numeric(chunk["startYear"],       errors="coerce")
        chunk["runtimeMinutes"]  = pd.to_numeric(chunk["runtimeMinutes"],  errors="coerce")
        chunk.to_sql("titles", conn,
                     if_exists="replace" if first else "append",
                     index=False)
        first = False
        rows_written += len(chunk)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_titles_tconst ON titles(tconst)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_titles_year   ON titles(startYear)")
    conn.commit()
    if not HAS_TQDM:
        print("done")
    print(f"    ✓ {rows_written:,} movies written")


def load_ratings(conn: sqlite3.Connection, data_dir: Path):
    """ratings — small file, load in one shot."""
    print("\n📂  Loading title.ratings...")
    path = data_dir / "title.ratings.tsv.gz"
    df = _read_tsv_gz(path)
    df["averageRating"] = pd.to_numeric(df["averageRating"], errors="coerce")
    df["numVotes"]      = pd.to_numeric(df["numVotes"],      errors="coerce")
    df.to_sql("ratings", conn, if_exists="replace", index=False)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_ratings_tconst ON ratings(tconst)")
    conn.commit()
    print(f"    ✓ {len(df):,} ratings written")


def load_crew(conn: sqlite3.Connection, data_dir: Path):
    """crew — directors & writers per title."""
    print("\n📂  Loading title.crew...")
    path = data_dir / "title.crew.tsv.gz"
    CHUNK = 500_000
    reader = _read_tsv_gz(path, usecols=["tconst", "directors"], chunksize=CHUNK)
    first = True
    rows_written = 0
    for chunk in _progress("title.crew", reader):
        # Explode comma-separated director nconsts into one row each
        chunk = chunk.dropna(subset=["directors"])
        chunk = chunk.assign(
            directors=chunk["directors"].str.split(",")
        ).explode("directors")
        chunk.rename(columns={"directors": "nconst"}, inplace=True)
        chunk = chunk[chunk["nconst"].str.startswith("nm", na=False)]
        chunk.to_sql("crew", conn,
                     if_exists="replace" if first else "append",
                     index=False)
        first = False
        rows_written += len(chunk)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_crew_tconst ON crew(tconst)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_crew_nconst ON crew(nconst)")
    conn.commit()
    if not HAS_TQDM:
        print("done")
    print(f"    ✓ {rows_written:,} director–film links written")


def load_names(conn: sqlite3.Connection, data_dir: Path):
    """name.basics — only people who appear as directors in crew table."""
    print("\n📂  Loading name.basics  (directors only)...")
    path = data_dir / "name.basics.tsv.gz"
    CHUNK = 500_000

    # Fetch the set of director nconsts we actually need
    director_nconsts = set(
        pd.read_sql("SELECT DISTINCT nconst FROM crew", conn)["nconst"]
    )
    print(f"    (filtering to {len(director_nconsts):,} known directors)")

    reader = _read_tsv_gz(
        path,
        usecols=["nconst", "primaryName", "birthYear", "deathYear"],
        chunksize=CHUNK,
    )
    first = True
    rows_written = 0
    for chunk in _progress("name.basics", reader):
        chunk = chunk[chunk["nconst"].isin(director_nconsts)]
        chunk.to_sql("people", conn,
                     if_exists="replace" if first else "append",
                     index=False)
        first = False
        rows_written += len(chunk)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_people_nconst ON people(nconst)")
    conn.commit()
    if not HAS_TQDM:
        print("done")
    print(f"    ✓ {rows_written:,} people written")


def load_all(data_dir: Path):
    """Run all loaders and return an open connection."""
    OUT_DIR.mkdir(exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA cache_size=-64000")   # 64 MB cache

    t0 = time.time()
    load_title_basics(conn, data_dir)
    load_ratings(conn, data_dir)
    load_crew(conn, data_dir)
    load_names(conn, data_dir)
    print(f"\n✅  Database built in {time.time()-t0:.0f}s  →  {DB_PATH}")
    return conn


# ══════════════════════════════════════════════════════════════════════════════
#  2. SQL ANALYTICS  (all queries return DataFrames)
# ══════════════════════════════════════════════════════════════════════════════

QUERIES = {}

QUERIES["top_movies"] = """
    SELECT  t.primaryTitle            AS title,
            t.startYear               AS year,
            r.averageRating           AS rating,
            r.numVotes                AS votes,
            t.genres
    FROM    titles  t
    JOIN    ratings r ON t.tconst = r.tconst
    WHERE   r.numVotes  >= 50000
      AND   t.startYear IS NOT NULL
      AND   t.titleType = 'movie'
    ORDER BY r.averageRating DESC, r.numVotes DESC
    LIMIT   25
"""

QUERIES["top_genres"] = """
    WITH genre_exploded AS (
        SELECT  r.averageRating,
                r.numVotes,
                TRIM(value) AS genre
        FROM    titles  t
        JOIN    ratings r ON t.tconst = r.tconst,
                json_each('["' || REPLACE(t.genres, ',', '","') || '"]')
        WHERE   t.genres IS NOT NULL
          AND   r.numVotes >= 5000
          AND   t.titleType = 'movie'
    )
    SELECT  genre,
            ROUND(AVG(averageRating), 2)    AS avg_rating,
            COUNT(*)                         AS num_movies,
            ROUND(SUM(numVotes) / 1e6, 1)   AS total_votes_m
    FROM    genre_exploded
    WHERE   genre NOT IN ('\\N', 'Adult')
    GROUP   BY genre
    HAVING  num_movies >= 50
    ORDER   BY avg_rating DESC
"""

QUERIES["ratings_by_decade"] = """
    SELECT  (t.startYear / 10) * 10           AS decade,
            ROUND(AVG(r.averageRating), 2)     AS avg_rating,
            COUNT(*)                           AS num_movies,
            ROUND(AVG(r.numVotes),0)           AS avg_votes
    FROM    titles  t
    JOIN    ratings r ON t.tconst = r.tconst
    WHERE   t.startYear BETWEEN 1930 AND 2024
      AND   r.numVotes  >= 1000
      AND   t.titleType = 'movie'
    GROUP   BY decade
    ORDER   BY decade
"""

QUERIES["rating_distribution"] = """
    SELECT  ROUND(r.averageRating, 1)  AS rating_bucket,
            COUNT(*)                    AS num_movies
    FROM    titles  t
    JOIN    ratings r ON t.tconst = r.tconst
    WHERE   r.numVotes  >= 1000
      AND   t.titleType = 'movie'
    GROUP   BY rating_bucket
    ORDER   BY rating_bucket
"""

QUERIES["top_directors"] = """
    SELECT  p.primaryName              AS director,
            COUNT(DISTINCT c.tconst)   AS num_movies,
            ROUND(AVG(r.averageRating), 2) AS avg_rating,
            MAX(r.averageRating)       AS best_rating,
            MAX(r.numVotes)            AS max_votes
    FROM    crew    c
    JOIN    people  p ON c.nconst = p.nconst
    JOIN    titles  t ON c.tconst = t.tconst
    JOIN    ratings r ON c.tconst = r.tconst
    WHERE   r.numVotes  >= 10000
      AND   t.titleType = 'movie'
    GROUP   BY p.primaryName
    HAVING  num_movies >= 5
    ORDER   BY avg_rating DESC, num_movies DESC
    LIMIT   20
"""

QUERIES["genre_by_decade"] = """
    WITH ge AS (
        SELECT  (t.startYear / 10) * 10 AS decade,
                TRIM(value)              AS genre,
                r.averageRating
        FROM    titles  t
        JOIN    ratings r ON t.tconst = r.tconst,
                json_each('["' || REPLACE(t.genres, ',', '","') || '"]')
        WHERE   t.startYear BETWEEN 1950 AND 2024
          AND   t.genres IS NOT NULL
          AND   r.numVotes >= 1000
          AND   t.titleType = 'movie'
    )
    SELECT  decade,
            genre,
            COUNT(*) AS num_movies
    FROM    ge
    WHERE   genre NOT IN ('\\N', 'Adult')
    GROUP   BY decade, genre
    ORDER   BY decade, num_movies DESC
"""

QUERIES["votes_by_year"] = """
    SELECT  t.startYear                    AS year,
            COUNT(*)                        AS num_movies,
            ROUND(AVG(r.averageRating), 2)  AS avg_rating
    FROM    titles  t
    JOIN    ratings r ON t.tconst = r.tconst
    WHERE   t.startYear BETWEEN 1950 AND 2024
      AND   r.numVotes  >= 500
      AND   t.titleType = 'movie'
    GROUP   BY t.startYear
    ORDER   BY t.startYear
"""


def run_analytics(conn: sqlite3.Connection) -> dict:
    print("\n🔍  Running SQL analytics...")
    results = {}
    for name, sql in QUERIES.items():
        try:
            results[name] = pd.read_sql(sql, conn)
            print(f"    ✓ {name:25s}  ({len(results[name])} rows)")
        except Exception as e:
            print(f"    ✗ {name:25s}  ERROR: {e}")
            results[name] = pd.DataFrame()
    return results


def export_csvs(results: dict):
    """Export each analytics result to CSV."""
    csv_dir = OUT_DIR / "csv"
    csv_dir.mkdir(exist_ok=True)
    for name, df in results.items():
        if not df.empty:
            path = csv_dir / f"{name}.csv"
            df.to_csv(path, index=False)
    print(f"    ✓ CSVs exported → {csv_dir}/")


# ══════════════════════════════════════════════════════════════════════════════
#  3. VISUALISATIONS
# ══════════════════════════════════════════════════════════════════════════════

def _style(ax, title="", xlabel="", ylabel=""):
    ax.set_facecolor(BG_AX)
    if title:
        ax.set_title(title, color=FG, fontsize=10, fontweight="bold", pad=8)
    if xlabel:
        ax.set_xlabel(xlabel, color=MUTED, fontsize=8)
    if ylabel:
        ax.set_ylabel(ylabel, color=MUTED, fontsize=8)
    ax.tick_params(colors=MUTED, labelsize=7.5)
    for spine in ax.spines.values():
        spine.set_edgecolor(GRID)
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, color=GRID, linewidth=0.5)


def plot_dashboard(results: dict):
    fig = plt.figure(figsize=(18, 12), facecolor=BG)
    gs  = GridSpec(2, 3, figure=fig,
                   hspace=0.48, wspace=0.35,
                   left=0.06, right=0.97,
                   top=0.91, bottom=0.07)

    fig.text(0.5, 0.965, "IMDb Movie Database Explorer",
             ha="center", fontsize=20, fontweight="bold", color=FG)
    fig.text(0.5, 0.937, "Analysis of IMDb public dataset  ·  movies with ≥ 1,000 votes",
             ha="center", fontsize=9, color=MUTED)

    # ── [0,0]  Rating distribution histogram ──────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    _style(ax1, "Rating Distribution", "Average Rating", "Number of Movies")
    rd = results.get("rating_distribution", pd.DataFrame())
    if not rd.empty:
        ax1.bar(rd["rating_bucket"], rd["num_movies"],
                width=0.09, color=PALETTE[0], edgecolor=BG, linewidth=0.5)
        ax1.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f"))
        ax1.yaxis.set_major_formatter(
            mticker.FuncFormatter(lambda x, _: f"{x/1000:.0f}k"))

    # ── [0,1]  Average rating by decade ───────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    _style(ax2, "Avg Rating & Movies Released by Decade",
           "Decade", "Avg IMDb Rating")
    rd2 = results.get("ratings_by_decade", pd.DataFrame())
    if not rd2.empty:
        ax2_r = ax2.twinx()
        ax2_r.set_facecolor(BG_AX)
        ax2_r.bar(rd2["decade"], rd2["num_movies"], width=7,
                  color=PALETTE[2], alpha=0.3, label="# Movies")
        ax2_r.yaxis.set_major_formatter(
            mticker.FuncFormatter(lambda x, _: f"{x/1000:.0f}k"))
        ax2_r.tick_params(colors=MUTED, labelsize=7.5)
        ax2_r.set_ylabel("Movies Released", color=MUTED, fontsize=8)
        for spine in ax2_r.spines.values():
            spine.set_edgecolor(GRID)

        ax2.plot(rd2["decade"], rd2["avg_rating"],
                 color=PALETTE[1], linewidth=2.5, marker="o",
                 markersize=5, zorder=5, label="Avg Rating")
        ax2.set_ylim(5.5, 8.0)
        ax2.set_ylabel("Avg Rating", color=MUTED, fontsize=8)

        lines1, labels1 = ax2.get_legend_handles_labels()
        lines2, labels2 = ax2_r.get_legend_handles_labels()
        ax2.legend(lines1 + lines2, labels1 + labels2,
                   fontsize=7, framealpha=0, labelcolor=FG, loc="upper left")

    # ── [0,2]  Top genres by avg rating ───────────────────────────────────
    ax3 = fig.add_subplot(gs[0, 2])
    _style(ax3, "Top Genres by Avg Rating", "", "Avg Rating")
    tg = results.get("top_genres", pd.DataFrame())
    if not tg.empty:
        top = tg.head(12).sort_values("avg_rating")
        bars = ax3.barh(top["genre"], top["avg_rating"],
                        color=PALETTE[:len(top)][::-1],
                        edgecolor=BG, linewidth=0.5)
        for bar, val in zip(bars, top["avg_rating"]):
            ax3.text(bar.get_width() + 0.01,
                     bar.get_y() + bar.get_height() / 2,
                     f"{val:.2f}", va="center", color=FG, fontsize=7.5)
        ax3.set_xlim(tg["avg_rating"].min() - 0.3,
                     tg["avg_rating"].max() + 0.25)

    # ── [1,0]  Movies per year line ────────────────────────────────────────
    ax4 = fig.add_subplot(gs[1, 0])
    _style(ax4, "Movies Released per Year (≥500 votes)",
           "Year", "Number of Movies")
    vy = results.get("votes_by_year", pd.DataFrame())
    if not vy.empty:
        ax4.fill_between(vy["year"], vy["num_movies"],
                         alpha=0.25, color=PALETTE[3])
        ax4.plot(vy["year"], vy["num_movies"],
                 color=PALETTE[3], linewidth=1.8)

    # ── [1,1]  Top directors ───────────────────────────────────────────────
    ax5 = fig.add_subplot(gs[1, 1])
    _style(ax5, "Top Directors by Avg Rating (≥5 films, ≥10k votes)",
           "Avg Rating", "")
    td = results.get("top_directors", pd.DataFrame())
    if not td.empty:
        top_d = td.head(12).sort_values("avg_rating")
        colours = [PALETTE[i % len(PALETTE)] for i in range(len(top_d))]
        bars = ax5.barh(top_d["director"], top_d["avg_rating"],
                        color=colours[::-1], edgecolor=BG, linewidth=0.5)
        for bar, row in zip(bars, top_d.itertuples()):
            ax5.text(bar.get_width() + 0.01,
                     bar.get_y() + bar.get_height() / 2,
                     f"{row.avg_rating:.2f}  ({row.num_movies} films)",
                     va="center", color=FG, fontsize=6.5)
        ax5.set_xlim(top_d["avg_rating"].min() - 0.3,
                     top_d["avg_rating"].max() + 0.6)

    # ── [1,2]  Genre popularity over decades (stacked area) ───────────────
    ax6 = fig.add_subplot(gs[1, 2])
    _style(ax6, "Genre Popularity Over Decades", "Decade", "Movies Released")
    gd = results.get("genre_by_decade", pd.DataFrame())
    if not gd.empty:
        pivot = gd.pivot_table(
            index="decade", columns="genre",
            values="num_movies", fill_value=0
        )
        # Keep only top 8 genres by total
        top_genres = pivot.sum().nlargest(8).index
        pivot = pivot[top_genres]
        pivot.plot.area(ax=ax6, color=PALETTE[:len(top_genres)],
                        alpha=0.85, linewidth=0)
        ax6.legend(fontsize=6.5, framealpha=0, labelcolor=FG,
                   ncol=2, loc="upper left")
        ax6.yaxis.set_major_formatter(
            mticker.FuncFormatter(lambda x, _: f"{x:.0f}"))

    out = OUT_DIR / "dashboard.png"
    plt.savefig(out, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"  ✓ Dashboard  → {out}")


def plot_top_movies(results: dict):
    """Separate chart: top 25 highest-rated movies."""
    tm = results.get("top_movies", pd.DataFrame())
    if tm.empty:
        return

    top = tm.head(20)
    fig, ax = plt.subplots(figsize=(12, 8), facecolor=BG)
    _style(ax, "Top 20 Highest-Rated Movies  (≥50,000 votes)", "", "IMDb Rating")
    ax.set_facecolor(BG_AX)

    labels = [f"{r['title']} ({int(r['year'])})" for _, r in top.iterrows()]
    colours = [PALETTE[i % len(PALETTE)] for i in range(len(top))]
    bars = ax.barh(labels[::-1], top["rating"][::-1],
                   color=colours, edgecolor=BG, linewidth=0.5)

    for bar, val in zip(bars, top["rating"][::-1]):
        ax.text(bar.get_width() + 0.01,
                bar.get_y() + bar.get_height() / 2,
                f"★ {val:.1f}", va="center", color=FG, fontsize=8)

    ax.set_xlim(top["rating"].min() - 0.3, top["rating"].max() + 0.35)
    fig.tight_layout()

    out = OUT_DIR / "top_movies.png"
    plt.savefig(out, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"  ✓ Top movies → {out}")


# ══════════════════════════════════════════════════════════════════════════════
#  4. TEXT REPORT
# ══════════════════════════════════════════════════════════════════════════════

def print_report(results: dict):
    sep = "─" * 60
    print(f"\n{'═'*60}")
    print("  IMDb MOVIE DATABASE REPORT")
    print(f"{'═'*60}")

    tm = results.get("top_movies")
    if tm is not None and not tm.empty:
        print(f"\n  TOP 10 MOVIES  (≥50,000 votes)")
        print(sep)
        print(f"  {'TITLE':<38} {'YEAR':>4}  {'RATING':>6}  {'VOTES':>9}")
        print(sep)
        for _, r in tm.head(10).iterrows():
            print(f"  {r['title']:<38} {int(r['year']):>4}  "
                  f"  {r['rating']:>5.1f}  {int(r['votes']):>9,}")

    tg = results.get("top_genres")
    if tg is not None and not tg.empty:
        print(f"\n  TOP GENRES BY AVG RATING")
        print(sep)
        print(f"  {'GENRE':<18} {'AVG RATING':>10}  {'MOVIES':>7}")
        print(sep)
        for _, r in tg.head(10).iterrows():
            print(f"  {r['genre']:<18} {r['avg_rating']:>10.2f}  "
                  f"{int(r['num_movies']):>7,}")

    td = results.get("top_directors")
    if td is not None and not td.empty:
        print(f"\n  TOP DIRECTORS  (≥5 films, ≥10k votes)")
        print(sep)
        print(f"  {'DIRECTOR':<28} {'FILMS':>5}  {'AVG':>5}  {'BEST':>5}")
        print(sep)
        for _, r in td.head(10).iterrows():
            print(f"  {r['director']:<28} {int(r['num_movies']):>5}  "
                  f"{r['avg_rating']:>5.2f}  {r['best_rating']:>5.1f}")

    print(f"\n{'═'*60}\n")


# ══════════════════════════════════════════════════════════════════════════════
#  5. MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="IMDb Movie Database Explorer")
    parser.add_argument("--data",      default="data",
                        help="Directory containing IMDb .tsv.gz files (default: ./data)")
    parser.add_argument("--skip-load", action="store_true",
                        help="Skip loading step (use existing DB)")
    parser.add_argument("--db",        default=None,
                        help="Override DB path")
    args = parser.parse_args()

    global DB_PATH
    if args.db:
        DB_PATH = Path(args.db)

    OUT_DIR.mkdir(exist_ok=True)
    data_dir = Path(args.data)

    if not args.skip_load:
        # Validate files exist
        required = [
            "title.basics.tsv.gz", "title.ratings.tsv.gz",
            "title.crew.tsv.gz",   "name.basics.tsv.gz",
        ]
        missing = [f for f in required if not (data_dir / f).exists()]
        if missing:
            print(f"\n❌  Missing files in '{data_dir}':")
            for f in missing:
                print(f"       {f}")
            print(f"\n   Download from: https://datasets.imdbws.com/")
            print(f"   Place files in: {data_dir.resolve()}/\n")
            sys.exit(1)

        conn = load_all(data_dir)
    else:
        if not DB_PATH.exists():
            print(f"\n❌  Database not found: {DB_PATH}")
            sys.exit(1)
        conn = sqlite3.connect(DB_PATH)
        print(f"\n🗄   Using existing database: {DB_PATH}")

    results = run_analytics(conn)
    export_csvs(results)

    print("\n📊  Generating visualisations...")
    plot_dashboard(results)
    plot_top_movies(results)

    print_report(results)
    conn.close()
    print(f"✅  All outputs saved to: {OUT_DIR.resolve()}/\n")


if __name__ == "__main__":
    main()