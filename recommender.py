# ============================================================
# PROJECT 3: AI Recommendation Logic
# ============================================================
# Intern      : [Write Your Full Name Here]
# Organization: DecodeLabs
# Batch       : 2026
# File        : recommender.py
# ============================================================
# OBJECTIVE   : Build a Content-Based Tech Stack Recommender
# ARCHITECTURE: IPO Model (Input → Process → Output)
# ALGORITHM   : TF-IDF Vectorization + Cosine Similarity
# DATASET     : raw_skills.csv (20 tech job roles)
# ============================================================
# PIPELINE:
#   Step 1: Load Dataset (raw_skills.csv)
#   Step 2: Build TF-IDF Vector Space
#   Step 3: Accept User Skill Input
#   Step 4: Score via Cosine Similarity
#   Step 5: Sort and Filter (Top-N)
#   Step 6: Display Recommendations
# ============================================================

# --- CORE LIBRARIES ---
import numpy as np
import pandas as pd

# --- ML TOOLS (already installed from Project 2) ---
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# --- VISUALIZATION ---
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

# --- BUILT-IN ---
import warnings
warnings.filterwarnings('ignore')

# --- DISPLAY SETTINGS ---
pd.set_option('display.max_colwidth', 100)

print("=" * 60)
print("  PROJECT 3: AI Recommendation Logic")
print("  Algorithm : TF-IDF + Cosine Similarity")
print("  Domain    : Tech Stack Recommender")
print("  Powered by: DecodeLabs | Batch 2026")
print("=" * 60)
print()

# ============================================================
# STEP 1: LOAD AND EXPLORE THE DATASET
# ============================================================

print("[ STEP 1 ] Loading Tech Jobs Dataset...")
print("-" * 60)

# --- Load the CSV file ---
# This is our "item catalog" — like Netflix's movie database
# but for tech job roles
try:
    df = pd.read_csv('raw_skills.csv')
    print(f"✅ Dataset loaded successfully!")
except FileNotFoundError:
    print("❌ ERROR: raw_skills.csv not found!")
    print("   Make sure raw_skills.csv is in the same")
    print("   folder as recommender.py")
    exit()

print()

# --- Basic exploration ---
print(f"Total Job Roles    : {len(df)}")
print(f"Columns            : {list(df.columns)}")
print()

# --- Show all job roles available ---
print("Available Job Roles in our Catalog:")
print("-" * 40)
for i, role in enumerate(df['job_role'], 1):
    print(f"  {i:2}. {role}")
print()

# --- Show sample entries with their skills ---
print("Sample Dataset Entries (first 3 roles):")
print("-" * 40)
for i in range(3):
    role   = df['job_role'][i]
    skills = df['skills'][i]
    skill_list = skills.split()
    print(f"\n  Role   : {role}")
    print(f"  Skills : {skill_list[:5]}... "
          f"({len(skill_list)} skills total)")

print()
print("✅ Dataset exploration complete!")
print()

# ============================================================
# STEP 2: BUILD THE TF-IDF VECTOR SPACE
# ============================================================

print("[ STEP 2 ] Building TF-IDF Vector Space...")
print("-" * 60)

# --- Create the TF-IDF Vectorizer ---
# This is our "translation machine" — text → numbers
# It will learn the vocabulary from ALL job role skills
tfidf = TfidfVectorizer(
    analyzer='word',      # Analyze at word level
    ngram_range=(1, 1),   # Single words only (unigrams)
    min_df=1,             # Include all terms
    stop_words=None       # Keep all words (skills are all important)
)

# --- Fit and transform the skills column ---
# fit()      = learn the vocabulary from all job skills
# transform()= convert each job's skills into a TF-IDF vector
# fit_transform() does both at once on the same data
job_vectors = tfidf.fit_transform(df['skills'])

# --- Explore the vector space ---
vocabulary    = tfidf.get_feature_names_out()
vocab_size    = len(vocabulary)
matrix_shape  = job_vectors.shape

print(f"Vocabulary Size    : {vocab_size} unique skills")
print(f"Matrix Shape       : {matrix_shape}")
print(f"  → {matrix_shape[0]} job roles")
print(f"  → {matrix_shape[1]} skill dimensions")
print()

# --- Show vocabulary sample ---
print(f"Sample Skills in Vocabulary (first 15):")
print(f"  {list(vocabulary[:15])}")
print()

# --- Show TF-IDF weights for first job role ---
first_role        = df['job_role'][0]
first_vector      = job_vectors[0].toarray()[0]
nonzero_indices   = np.where(first_vector > 0)[0]

print(f"TF-IDF Weights for '{first_role}':")
print(f"  {'Skill':<25} {'TF-IDF Weight'}")
print(f"  {'-'*40}")

# Sort by weight (highest first)
skill_weights = [(vocabulary[i], first_vector[i])
                 for i in nonzero_indices]
skill_weights.sort(key=lambda x: x[1], reverse=True)

for skill, weight in skill_weights:
    bar = "█" * int(weight * 30)
    print(f"  {skill:<25} {weight:.4f}  {bar}")

print()
print("✅ TF-IDF Vector Space built successfully!")
print()

# ============================================================
# STEP 3: BUILD THE RECOMMENDATION ENGINE FUNCTION
# ============================================================

print("[ STEP 3 ] Building Recommendation Engine...")
print("-" * 60)

def get_recommendations(user_skills, top_n=3,
                        min_score=0.0, verbose=True):
    """
    Generate job role recommendations based on user skills.

    Parameters:
        user_skills : list → User's skills
                      e.g. ["python", "machine_learning", "sql"]
        top_n       : int  → Number of recommendations (default 3)
        min_score   : float→ Minimum similarity threshold (default 0)
        verbose     : bool → Print detailed output (default True)

    Returns:
        DataFrame with ranked recommendations and scores
    """

    # --- COLD START PROTECTION ---
    # Minimum 3 skills required for meaningful recommendations
    if len(user_skills) < 3:
        print("⚠️  WARNING: Please provide at least 3 skills")
        print("   More skills = more accurate recommendations!")
        print("   Current input has only "
              f"{len(user_skills)} skill(s)")
        return None

    # --- PIPELINE STEP 1: INGESTION ---
    # Normalize: lowercase and strip whitespace
    # Then join into single string for vectorizer
    normalized_skills = [s.lower().strip().replace(' ', '_')
                         for s in user_skills]
    user_profile_text = ' '.join(normalized_skills)

    if verbose:
        print(f"  User Skills      : {user_skills}")
        print(f"  Normalized       : {normalized_skills}")
        print(f"  Profile Text     : '{user_profile_text}'")
        print()

    # --- PIPELINE STEP 2: SCORING ---
    # Transform user profile using SAME fitted vectorizer
    # Critical: transform() NOT fit_transform()
    user_vector = tfidf.transform([user_profile_text])

    # Calculate cosine similarity between user and ALL job roles
    # Returns array of shape (1, 20) → one score per job
    similarity_scores = cosine_similarity(
        user_vector, job_vectors
    ).flatten()   # flatten() converts [[s1,s2...]] to [s1,s2...]

    # --- PIPELINE STEP 3: SORTING ---
    # Get indices that would sort scores highest to lowest
    sorted_indices = np.argsort(similarity_scores)[::-1]

    # --- PIPELINE STEP 4: FILTERING ---
    # Build results DataFrame with top matches
    results = []
    rank = 1

    for idx in sorted_indices:
        score    = similarity_scores[idx]
        job_role = df['job_role'][idx]
        skills   = df['skills'][idx].split()

        # Skip if below minimum threshold
        if score < min_score:
            continue

        # Find matching skills between user and job role
        matching = [s for s in normalized_skills
                    if s in skills]
        missing  = [s for s in skills
                    if s not in normalized_skills][:3]

        results.append({
            'rank'           : rank,
            'job_role'       : job_role,
            'match_score'    : round(score, 4),
            'match_percent'  : f"{score*100:.1f}%",
            'matching_skills': matching,
            'skills_to_learn': missing
        })

        rank += 1
        if rank > top_n:
            break

    # Convert to DataFrame for clean display
    results_df = pd.DataFrame(results)
    return results_df


def display_recommendations(results_df, user_skills):
    """
    Display recommendations in a clean formatted way.

    Parameters:
        results_df  : DataFrame → Output from get_recommendations()
        user_skills : list      → Original user skills for context
    """

    if results_df is None or len(results_df) == 0:
        print("❌ No recommendations found!")
        print("   Try adding more relevant tech skills.")
        return

    print()
    print("=" * 60)
    print("  🎯 YOUR PERSONALIZED JOB RECOMMENDATIONS")
    print("=" * 60)
    print(f"  Based on your skills: {user_skills}")
    print("=" * 60)
    print()

    for _, row in results_df.iterrows():
        rank    = row['rank']
        role    = row['job_role']
        score   = row['match_score']
        percent = row['match_percent']
        matched = row['matching_skills']
        missing = row['skills_to_learn']

        # Medal emoji based on rank
        medal = {1: "🥇", 2: "🥈", 3: "🥉"}.get(rank, "🏅")

        # Build visual score bar
        bar_length = int(score * 30)
        bar = "█" * bar_length + "░" * (30 - bar_length)

        print(f"  {medal} RANK #{rank}: {role}")
        print(f"  {'─'*50}")
        print(f"  Match Score    : {percent}")
        print(f"  [{bar}]")

        if matched:
            print(f"  ✅ Your skills that match:")
            for skill in matched:
                print(f"      → {skill}")

        if missing:
            print(f"  📚 Top skills to learn for this role:")
            for skill in missing:
                print(f"      → {skill}")

        print()

    print("=" * 60)
    print()


print("✅ Recommendation engine functions built!")
print()

# ============================================================
# STEP 4: TEST THE RECOMMENDATION ENGINE
# ============================================================

print("[ STEP 4 ] Testing Recommendation Engine...")
print("-" * 60)

# ── TEST 1: Data Science Profile ──────────────────────────
print("TEST 1: Data Science Profile")
print("─" * 60)

skills_test1 = [
    "python", "machine_learning",
    "statistics", "sql", "data_analysis"
]

results1 = get_recommendations(
    user_skills=skills_test1,
    top_n=3,
    verbose=True
)
display_recommendations(results1, skills_test1)


# ── TEST 2: Web Development Profile ───────────────────────
print("TEST 2: Web Development Profile")
print("─" * 60)

skills_test2 = [
    "javascript", "react",
    "nodejs", "html", "css"
]

results2 = get_recommendations(
    user_skills=skills_test2,
    top_n=3,
    verbose=True
)
display_recommendations(results2, skills_test2)


# ── TEST 3: Cloud/DevOps Profile ──────────────────────────
print("TEST 3: Cloud/DevOps Profile")
print("─" * 60)

skills_test3 = [
    "aws", "docker",
    "kubernetes", "linux", "terraform"
]

results3 = get_recommendations(
    user_skills=skills_test3,
    top_n=3,
    verbose=True
)
display_recommendations(results3, skills_test3)


# ── TEST 4: Cold Start Test (less than 3 skills) ──────────
print("TEST 4: Cold Start Protection Test")
print("─" * 60)

skills_test4 = ["python"]   # Only 1 skill — should warn!

results4 = get_recommendations(
    user_skills=skills_test4,
    top_n=3,
    verbose=False
)
print()

# ============================================================
# ENHANCEMENT 1: INTERACTIVE MODE
# ============================================================

print()
print("[ ENHANCEMENT 1 ] Interactive Recommendation Mode")
print("-" * 60)

def run_interactive_mode():
    """
    Run the recommender in interactive terminal mode.
    User types their own skills and gets live recommendations.
    """

    print()
    print("=" * 60)
    print("  🤖 INTERACTIVE TECH STACK RECOMMENDER")
    print("  Type your skills one by one.")
    print("  Minimum 3 skills required.")
    print("  Type 'done' when finished entering skills.")
    print("  Type 'exit' to quit the program.")
    print("=" * 60)
    print()
    print("  Available skill examples:")
    print("  python, javascript, sql, docker, aws,")
    print("  machine_learning, react, tensorflow,")
    print("  kubernetes, nodejs, git, tensorflow...")
    print()

    # --- Outer loop: allows multiple recommendation sessions ---
    while True:

        user_skills = []
        print("─" * 60)
        print("  Enter your skills (one per line):")
        print("─" * 60)

        # --- Inner loop: collect skills one by one ---
        skill_count = 1
        while True:

            # Get skill input from user
            skill = input(f"  Skill {skill_count}: ").strip()

            # Check for exit command
            if skill.lower() == 'exit':
                print()
                print("  👋 Thanks for using the Tech Stack Recommender!")
                print("  Goodbye!")
                print("=" * 60)
                return   # Exit the entire function

            # Check for done command
            if skill.lower() == 'done':
                if len(user_skills) < 3:
                    print(f"  ⚠️  You only entered {len(user_skills)}"
                          f" skill(s). Please add at least"
                          f" {3 - len(user_skills)} more.")
                    continue
                else:
                    break   # Exit inner loop, process skills

            # Validate skill is not empty
            if not skill:
                print("  ⚠️  Empty input. Please type a skill.")
                continue

            # Normalize the skill
            normalized = skill.lower().replace(' ', '_')

            # Check if skill exists in our vocabulary
            vocab_list = tfidf.get_feature_names_out()
            if normalized not in vocab_list:
                print(f"  ⚠️  '{normalized}' not found in our"
                      f" dataset vocabulary.")
                print(f"      Adding it anyway — it may reduce"
                      f" accuracy slightly.")

            # Add to skills list
            user_skills.append(normalized)
            print(f"  ✅ Added: {normalized}"
                  f"  (Total: {len(user_skills)} skills)")
            skill_count += 1

            # Remind user they can type 'done' after 3 skills
            if len(user_skills) == 3:
                print()
                print("  💡 You have 3 skills! Type 'done' to get")
                print("     recommendations or keep adding more.")
                print()

        # --- Generate and display recommendations ---
        print()
        print("  Analyzing your profile...")
        print("  Calculating similarity scores...")
        print()

        results = get_recommendations(
            user_skills=user_skills,
            top_n=3,
            verbose=False   # Clean output for interactive mode
        )
        display_recommendations(results, user_skills)

        # --- Ask if user wants to try again ---
        print("─" * 60)
        again = input("  Try with different skills? (yes/no): "
                     ).strip().lower()
        print()

        if again not in ['yes', 'y']:
            print("  👋 Thanks for using the Tech Stack Recommender!")
            print("=" * 60)
            break


# ============================================================
# ENHANCEMENT 2: SIMILARITY VISUALIZATION
# ============================================================

print()
print("[ ENHANCEMENT 2 ] Generating Similarity Charts...")
print("-" * 60)

def visualize_recommendations(user_skills,
                               title_suffix=""):
    """
    Generate a comprehensive visualization of similarity
    scores across all job roles for given user skills.

    Parameters:
        user_skills   : list → User's skill set
        title_suffix  : str  → Extra label for chart title
    """

    # --- Calculate scores for ALL job roles ---
    normalized = [s.lower().strip().replace(' ', '_')
                  for s in user_skills]
    profile_text = ' '.join(normalized)
    user_vector  = tfidf.transform([profile_text])
    scores       = cosine_similarity(
        user_vector, job_vectors
    ).flatten()

    # --- Build sorted DataFrame ---
    viz_df = pd.DataFrame({
        'job_role' : df['job_role'],
        'score'    : scores,
        'percent'  : scores * 100
    }).sort_values('score', ascending=True)

    # --- Color coding by score level ---
    colors = []
    for score in viz_df['score']:
        if score >= 0.50:
            colors.append('#2ecc71')    # Green  - high match
        elif score >= 0.25:
            colors.append('#f39c12')    # Orange - medium match
        else:
            colors.append('#e74c3c')    # Red    - low match

    # --- Create the figure ---
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    # ── Chart 1: Horizontal Bar Chart (All Roles) ─────────
    bars = axes[0].barh(
        viz_df['job_role'],
        viz_df['percent'],
        color=colors,
        edgecolor='white',
        linewidth=0.5,
        height=0.7
    )

    # Add percentage labels on bars
    for bar, score in zip(bars, viz_df['percent']):
        width = bar.get_width()
        axes[0].text(
            width + 0.5,
            bar.get_y() + bar.get_height()/2,
            f'{width:.1f}%',
            va='center', ha='left',
            fontsize=8, fontweight='bold'
        )

    # Reference lines
    axes[0].axvline(x=50, color='green',
                    linestyle='--', alpha=0.5,
                    linewidth=1, label='50% threshold')
    axes[0].axvline(x=25, color='orange',
                    linestyle='--', alpha=0.5,
                    linewidth=1, label='25% threshold')

    axes[0].set_xlabel('Match Score (%)', fontsize=11)
    axes[0].set_title(
        f'All Job Role Similarity Scores\n'
        f'Skills: {", ".join(normalized[:4])}...',
        fontweight='bold', fontsize=11
    )
    axes[0].set_xlim(0, 110)
    axes[0].legend(loc='lower right', fontsize=9)

    # Legend for colors
    high_patch   = mpatches.Patch(
        color='#2ecc71', label='High Match (≥50%)')
    medium_patch = mpatches.Patch(
        color='#f39c12', label='Medium Match (25-50%)')
    low_patch    = mpatches.Patch(
        color='#e74c3c', label='Low Match (<25%)')
    axes[0].legend(
        handles=[high_patch, medium_patch, low_patch],
        loc='lower right', fontsize=9
    )

    # ── Chart 2: Top 5 Roles — Detailed Comparison ────────
    top5 = viz_df.nlargest(5, 'score')
    top5_colors = ['#2ecc71' if s >= 0.5
                   else '#f39c12' if s >= 0.25
                   else '#e74c3c'
                   for s in top5['score']]

    bars2 = axes[1].bar(
        range(len(top5)),
        top5['percent'],
        color=top5_colors,
        edgecolor='black',
        linewidth=0.8,
        width=0.6
    )

    # Add value labels on top of bars
    for bar in bars2:
        height = bar.get_height()
        axes[1].text(
            bar.get_x() + bar.get_width()/2,
            height + 0.5,
            f'{height:.1f}%',
            ha='center', va='bottom',
            fontsize=11, fontweight='bold'
        )

    axes[1].set_xticks(range(len(top5)))
    axes[1].set_xticklabels(
        top5['job_role'],
        rotation=15,
        ha='right',
        fontsize=9
    )
    axes[1].set_ylabel('Match Score (%)', fontsize=11)
    axes[1].set_ylim(0, 110)
    axes[1].set_title(
        f'Top 5 Job Role Matches\n'
        f'Skills: {", ".join(normalized[:4])}...',
        fontweight='bold', fontsize=11
    )
    axes[1].grid(axis='y', alpha=0.3)

    # Overall figure title
    plt.suptitle(
        f'Tech Stack Recommender — Similarity Analysis\n'
        f'DecodeLabs Project 3 | {title_suffix}',
        fontsize=13, fontweight='bold'
    )

    plt.tight_layout()

   # Save with descriptive filename
    
    clean_suffix = title_suffix.lower().replace(' ', '_')
    filename = (f"similarity_{clean_suffix}.png"
                if title_suffix
                else "similarity_chart.png")
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"✅ Chart saved as: {filename}")
    print()
    return filename

# --- Generate charts for our test profiles ---
print("Generating chart for Data Science profile...")
visualize_recommendations(
    ["python", "machine_learning", "statistics",
     "sql", "data_analysis"],
    title_suffix="Data Science Profile"
)

print("Generating chart for Web Dev profile...")
visualize_recommendations(
    ["javascript", "react", "nodejs", "html", "css"],
    title_suffix="Web Dev Profile"
)

print("✅ All visualization charts generated!")
print()

# ============================================================
# ENHANCEMENT 3: MULTI-PROFILE COMPARISON
# ============================================================

print("[ ENHANCEMENT 3 ] Multi-Profile Comparison...")
print("-" * 60)

# --- Define 4 different career profiles ---
profiles = {
    "AI Engineer" : [
        "python", "tensorflow", "deep_learning",
        "machine_learning", "pytorch"
    ],
    "Cloud DevOps" : [
        "aws", "docker", "kubernetes",
        "terraform", "linux"
    ],
    "Web Developer" : [
        "javascript", "react", "nodejs",
        "html", "css"
    ],
    "Data Engineer" : [
        "python", "sql", "spark",
        "etl", "airflow"
    ]
}

# --- Get top recommendation for each profile ---
print(f"  {'Profile':<20} {'#1 Match':<30} {'Score'}")
print(f"  {'─'*65}")

profile_results = {}

for profile_name, skills in profiles.items():

    results = get_recommendations(
        user_skills=skills,
        top_n=1,
        verbose=False
    )

    if results is not None and len(results) > 0:
        top_role    = results.iloc[0]['job_role']
        top_score   = results.iloc[0]['match_percent']
        top_numeric = results.iloc[0]['match_score']

        print(f"  {profile_name:<20} {top_role:<30} {top_score}")
        profile_results[profile_name] = {
            'skills'     : skills,
            'top_role'   : top_role,
            'score'      : top_numeric
        }

print()

# --- Comparison Chart ---
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()

profile_colors = ['#3498db', '#e74c3c', '#2ecc71', '#9b59b6']

for idx, (profile_name, skills) in enumerate(
        profiles.items()):

    # Get all scores for this profile
    normalized   = [s.lower().strip() for s in skills]
    profile_text = ' '.join(normalized)
    user_vector  = tfidf.transform([profile_text])
    scores       = cosine_similarity(
        user_vector, job_vectors
    ).flatten()

    # Get top 5 for this profile
    top_indices = np.argsort(scores)[::-1][:5]
    top_roles   = [df['job_role'][i] for i in top_indices]
    top_scores  = [scores[i] * 100 for i in top_indices]

    # Plot
    bars = axes[idx].barh(
        top_roles[::-1],
        top_scores[::-1],
        color=profile_colors[idx],
        alpha=0.8,
        edgecolor='white'
    )

    # Labels
    for bar, score in zip(bars, top_scores[::-1]):
        axes[idx].text(
            bar.get_width() + 0.5,
            bar.get_y() + bar.get_height()/2,
            f'{score:.1f}%',
            va='center', fontsize=9,
            fontweight='bold'
        )

    axes[idx].set_title(
        f'{profile_name}\n'
        f'Skills: {", ".join(skills[:3])}...',
        fontweight='bold',
        color=profile_colors[idx],
        fontsize=10
    )
    axes[idx].set_xlim(0, 110)
    axes[idx].set_xlabel('Match Score (%)')
    axes[idx].grid(axis='x', alpha=0.3)

plt.suptitle(
    'Multi-Profile Career Comparison\n'
    'DecodeLabs Project 3 — Tech Stack Recommender',
    fontsize=13, fontweight='bold'
)
plt.tight_layout()
plt.savefig('multi_profile_comparison.png',
            dpi=150, bbox_inches='tight')
plt.show()

print("✅ Multi-profile comparison saved as:")
print("   multi_profile_comparison.png")
print()

# --- Run interactive mode ---
run_interactive_mode()

