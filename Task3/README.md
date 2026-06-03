# Project 3: AI Recommendation Logic 🎯

## 👤 Intern Details
- **Name:** Ayesha Nazish
- **Organization:** DecodeLabs
- **Batch:** 2026
- **Project:** Project 3 — AI Recommendation Logic

---

## 🎯 Project Objective
Build a Content-Based Tech Stack Recommender that maps
a user's raw skills and career goals to the most relevant
job roles using TF-IDF vectorization and Cosine Similarity —
moving beyond classification into active prediction.

---

## 🔗 Connection to Previous Projects

| Project | Type | Core Logic |
|---|---|---|
| Project 1 — Chatbot | Rule-Based | Dictionary lookup |
| Project 2 — Classifier | Supervised ML | KNN distance |
| Project 3 — Recommender | Content-Based | Cosine similarity |

**The progression:** Rules → Classification → Prediction

---

## 🧠 Architecture
INPUT              PROCESS                    OUTPUT
─────              ───────                    ──────
User Skills   →    TF-IDF Vectorization   →   Top-N Job Roles
(3+ skills)        Cosine Similarity           with Match %
4-Step Pipeline             Skills to Learn

### The 4-Step Ranking Pipeline

INGESTION  → Capture and normalize user skills
SCORING    → Calculate cosine similarity vs all roles
SORTING    → Rank by highest similarity score
FILTERING  → Return only Top-N results


---

## ✅ Features Implemented

| Feature | Status |
|---|---|
| Dataset loading (20 job roles) | ✅ Done |
| TF-IDF vector space (85 dimensions) | ✅ Done |
| Cosine similarity scoring | ✅ Done |
| Top-N recommendations with scores | ✅ Done |
| Matching skills display | ✅ Done |
| Skills to learn display | ✅ Done |
| Cold start protection (min 3 skills) | ✅ Done |
| Interactive terminal mode | ✅ Done |
| Similarity visualization charts | ✅ Done |
| Multi-profile career comparison | ✅ Done |

---

## 📁 Project Structure
Task3/
├── recommender.py              ← Main recommendation engine
├── raw_skills.csv              ← Dataset (20 job roles)
├── similarity_data_science_profile.png
├── similarity_web_dev_profile.png
├── multi_profile_comparison.png
└── README.md          

---

## 🚀 How to Run

### Requirements
- Python 3.7 or above
- Libraries: numpy, pandas, scikit-learn, matplotlib, seaborn

### Install Dependencies
```bash
pip install numpy pandas scikit-learn matplotlib seaborn
```

### Run the Recommender
```bash
cd e:\Decodelab_AI\Task3
python recommender.py
```

### Using Interactive Mode
When the interactive prompt appears:
Skill 1: python
Skill 2: machine_learning
Skill 3: sql
Skill 4: done
Type `exit` anytime to quit.

---

## 📊 Dataset Information

| Property | Value |
|---|---|
| File | raw_skills.csv |
| Total Job Roles | 20 |
| Total Unique Skills | ~85 |
| Format | job_role, skills (space-separated tags) |
| Skill Format | underscore_separated (e.g. machine_learning) |

### Sample Job Roles
Data Scientist, ML Engineer, Backend Developer,
Frontend Developer, DevOps Engineer, Cloud Architect,
NLP Engineer, Blockchain Developer, and 12 more.

---

## 🤖 Algorithm Details

### Why Content-Based Filtering?
- Works without historical user data
- Only needs item catalog (job role tags)
- Immediately effective for new users

### Why TF-IDF over Binary Vectors?
- Binary: treats "python" same as "kubernetes"
- TF-IDF: rare/specific skills get higher weight
- More accurate similarity calculations

### Why Cosine Similarity over Euclidean?
- Euclidean: sensitive to vector magnitude/length
- Cosine: only measures directional alignment
- Works correctly regardless of description length

---

## 📈 Sample Results

### Input: python, machine_learning, sql, statistics
#1 Data Scientist          → 78.3% match
#2 Machine Learning Eng    → 52.1% match
#3 AI Research Scientist   → 48.7% match

### Input: aws, docker, kubernetes, terraform
#1 DevOps Engineer         → 91.2% match
#2 Cloud Architect         → 87.4% match
#3 Data Engineer           → 31.2% match

---

## 🔑 Key Concepts Demonstrated

- **Content-Based Filtering** — item attribute matching
- **TF-IDF Vectorization** — weighted skill importance
- **Cosine Similarity** — direction-based matching
- **Vector Space Model** — mathematical skill representation
- **Cold Start Problem** — handled with minimum input rule
- **Top-N Filtering** — prevents choice overload
- **4-Step Pipeline** — Ingestion → Score → Sort → Filter

---

## 💡 What I Learned

- How recommendation engines work at Netflix/Amazon level
- Why TF-IDF is superior to simple binary encoding
- How cosine similarity handles variable-length profiles
- The Cold Start problem and how to solve it
- How to build a complete content-based filtering pipeline
- How mathematical similarity enables intelligent suggestions

---

## 📊 Output Files Generated

| File | Description |
|---|---|
| `similarity_data_science_profile.png` | All 20 roles scored for DS profile |
| `similarity_web_dev_profile.png` | All 20 roles scored for Web profile |
| `multi_profile_comparison.png` | 4 career profiles compared side by side |

---

*Built with ❤️ at DecodeLabs AI Industrial Training — Batch 2026*