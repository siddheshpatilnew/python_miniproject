"""
Music Analysis and Mood-Based Recommendation System
Reconstructed from the project's presentation and the supplied tracks.csv dataset.

Requirements:
    pip install pandas matplotlib seaborn

Files:
    Keep this script and tracks.csv in the same folder.

Run:
    python music_analysis_project.py
"""

from pathlib import Path
import ast
import re

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


DATA_FILE = Path(__file__).resolve().parent / "tracks.csv"
AUDIO_FEATURES = [
    "popularity",
    "danceability",
    "energy",
    "valence",
    "loudness",
]


def clean_title(title):
    """Remove text inside parentheses or square brackets from a song title."""
    title = str(title)
    title = re.sub(r"\([^)]*\)", "", title)
    title = re.sub(r"\[[^\]]*\]", "", title)
    return title.strip()


def load_and_clean_data(file_path=DATA_FILE):
    """Load the CSV, remove rows missing essential data, and prepare columns."""
    if not file_path.exists():
        raise FileNotFoundError(
            f"Could not find '{file_path.name}'. Put it in the same folder as this script."
        )

    df = pd.read_csv(file_path)

    required_columns = {
        "name", "artists", "popularity", "release_date",
        "danceability", "energy", "valence", "loudness"
    }
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        raise ValueError(
            "The dataset is missing these required columns: "
            + ", ".join(sorted(missing_columns))
        )

    # The supplied dataset already has a year column, but deriving it from
    # release_date also follows the method shown in the project presentation.
    df["year"] = pd.to_numeric(
        df["release_date"].astype(str).str[:4], errors="coerce"
    )

    df.dropna(subset=["name", "artists", "popularity", "year"], inplace=True)
    df["year"] = df["year"].astype(int)
    df["popularity"] = pd.to_numeric(df["popularity"], errors="coerce")

    for column in AUDIO_FEATURES:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")

    df.dropna(subset=["popularity"], inplace=True)
    df["clean_name"] = df["name"].apply(clean_title)

    print(f"Dataset loaded: {len(df):,} usable tracks")
    return df


def filter_by_start_year(df):
    """Keep songs released in or after a user-provided year."""
    while True:
        try:
            start_year = int(input("Enter starting year (e.g., 2017): ").strip())
            if 1900 <= start_year <= 2100:
                break
            print("Enter a year between 1900 and 2100.")
        except ValueError:
            print("Invalid input. Enter a year such as 2017.")

    recent_df = df[df["year"] >= start_year].copy()
    print(f"Rows used after year filtering: {len(recent_df):,}")
    if recent_df.empty:
        print("No tracks were found for that year or later.")
    return recent_df


def show_correlation_heatmap(recent_df):
    """Display a heatmap showing correlations among numeric music features."""
    columns = [column for column in AUDIO_FEATURES if column in recent_df.columns]
    numeric_data = recent_df[columns].dropna()

    if len(columns) < 2 or numeric_data.empty:
        print("Not enough numeric features to generate a correlation heatmap.")
        return

    correlation_matrix = numeric_data.corr()

    plt.figure(figsize=(9, 6))
    sns.heatmap(
        correlation_matrix,
        annot=True,
        cmap="coolwarm",
        fmt=".2f",
        vmin=-1,
        vmax=1,
    )
    plt.title(f"Correlation Between Song Features (Since {recent_df['year'].min()})")
    plt.tight_layout()
    plt.show()


def show_top_artists(recent_df):
    """Display a bar chart of artists with the highest mean popularity."""
    if recent_df.empty:
        return

    while True:
        try:
            top_n = int(input("Enter number of top artists to display (e.g., 20): "))
            if top_n > 0:
                break
            print("Enter a positive number.")
        except ValueError:
            print("Invalid input. Enter a positive whole number.")

    # In this dataset, artists are stored as strings such as "['Artist A']".
    # Grouping by that column follows the approach shown in the original PPT.
    artist_stats = (
        recent_df.groupby("artists")["popularity"]
        .mean()
        .sort_values(ascending=False)
        .head(top_n)
    )

    if artist_stats.empty:
        print("No artist records are available.")
        return

    plt.figure(figsize=(11, max(5, len(artist_stats) * 0.35)))
    sns.barplot(
        x=artist_stats.values,
        y=artist_stats.index,
        hue=artist_stats.index,
        palette="viridis",
        legend=False,
    )
    plt.title(f"Top {len(artist_stats)} Artists by Average Popularity")
    plt.xlabel("Average Popularity Score")
    plt.ylabel("Artists")
    plt.tight_layout()
    plt.show()


def recommend_music(recent_df):
    """Filter songs by the selected mood and show the five most popular matches."""
    print("\nChoose your mood:")
    print("1. Happy")
    print("2. Sad")
    print("3. Chill")

    try:
        choice = int(input("Enter mood (1, 2, or 3): ").strip())
    except ValueError:
        print("Invalid input. Please enter 1, 2, or 3.")
        return

    if choice == 1:
        mood = "Happy"
        condition = (
            (recent_df["valence"] > 0.6)
            & (recent_df["danceability"] > 0.6)
        )
    elif choice == 2:
        mood = "Sad"
        condition = recent_df["valence"] < 0.3
    elif choice == 3:
        mood = "Chill"
        condition = recent_df["energy"] < 0.5
    else:
        print("Invalid choice. Please select 1, 2, or 3.")
        return

    recommendations = recent_df.loc[condition].sort_values(
        by="popularity", ascending=False
    ).head(5)

    print(f"\nTop {len(recommendations)} {mood} song recommendations:")
    if recommendations.empty:
        print("No songs matched this mood in the selected year range.")
        return

    result_columns = ["clean_name", "artists", "popularity"]
    print(recommendations[result_columns].to_string(index=False))


def main():
    print("=" * 58)
    print(" MUSIC ANALYSIS AND MOOD-BASED RECOMMENDATION SYSTEM")
    print("=" * 58)

    try:
        df = load_and_clean_data()
    except (FileNotFoundError, ValueError, pd.errors.ParserError) as error:
        print(f"Error: {error}")
        return

    recent_df = filter_by_start_year(df)
    if recent_df.empty:
        return

    print("\nGenerating correlation heatmap...")
    show_correlation_heatmap(recent_df)

    print("\nGenerating top-artists popularity chart...")
    show_top_artists(recent_df)

    print("\nGenerating mood-based recommendations...")
    recommend_music(recent_df)


if __name__ == "__main__":
    main()
