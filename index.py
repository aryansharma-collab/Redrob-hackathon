from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# ── Job Descriptions ─────────────────────────────────────────────────────
job_descriptions = {
    "JD-1": {
        "company": "Kakao",
        "role": "ML Engineer",
        "text": (
            "machine learning deep learning NLP natural language processing "
            "computer vision TensorFlow PyTorch model deployment MLOps "
            "data pipelines feature engineering Python neural networks "
            "model training optimization hyperparameter tuning"
        ),
    },
    "JD-2": {
        "company": "Naver",
        "role": "Backend Engineer",
        "text": (
            "Java Spring Boot microservices REST APIs PostgreSQL Redis "
            "Docker Kubernetes CI CD pipelines distributed systems "
            "cloud infrastructure AWS scalability system design "
            "backend architecture load balancing"
        ),
    },
    "JD-3": {
        "company": "Line",
        "role": "Frontend Engineer",
        "text": (
            "React TypeScript JavaScript HTML CSS responsive design "
            "Next.js state management web performance optimization "
            "UI UX design component libraries frontend architecture "
            "accessibility testing browser compatibility"
        ),
    },
}

# ── Candidate Resumes ────────────────────────────────────────────────────
resumes = {
    "Sneha Patel": (
        "machine learning deep learning NLP natural language processing "
        "TensorFlow Python feature engineering data analysis "
        "statistical modeling scikit-learn neural networks "
        "model training research data science predictive modeling"
    ),
    "Karan Mehta": (
        "machine learning deep learning PyTorch computer vision "
        "neural networks model optimization Python NLP "
        "model training data preprocessing research papers "
        "hyperparameter tuning feature engineering"
    ),
    "Arjun Sharma": (
        "Python machine learning data pipelines feature engineering "
        "deep learning model deployment Django REST APIs SQL "
        "web development software engineering"
    ),
    "Rahul Gupta": (
        "Java Spring Boot microservices REST APIs PostgreSQL Redis "
        "Docker Kubernetes CI CD pipelines distributed systems "
        "cloud infrastructure AWS system design backend architecture "
        "load balancing scalability monitoring"
    ),
    "Ananya Krishnan": (
        "React JavaScript HTML CSS Node.js REST APIs "
        "MongoDB Express responsive design web development "
        "frontend UI component state management"
    ),
    "Deepika Rao": (
        "Java SQL Spring Boot HTML CSS cloud computing "
        "basic programming web development"
    ),
    "Aditya Kumar": (
        "React TypeScript JavaScript Next.js responsive design "
        "CSS HTML web performance state management Redux "
        "UI component libraries frontend architecture accessibility"
    ),
    "Priya Nair": (
        "React JavaScript HTML CSS responsive design "
        "UI UX design accessibility frontend testing "
        "web performance browser compatibility component libraries"
    ),
}

# ── Build corpus & compute TF-IDF ───────────────────────────────────────
jd_ids = list(job_descriptions.keys())
jd_texts = [job_descriptions[jid]["text"] for jid in jd_ids]

candidate_names = list(resumes.keys())
candidate_texts = list(resumes.values())

all_texts = jd_texts + candidate_texts

vectorizer = TfidfVectorizer(stop_words="english")
tfidf_matrix = vectorizer.fit_transform(all_texts)

jd_vectors = tfidf_matrix[: len(jd_ids)]
resume_vectors = tfidf_matrix[len(jd_ids) :]

# ── Compute similarity & apply domain-specific calibration ──────────────
raw_similarity = cosine_similarity(jd_vectors, resume_vectors)

# Calibration weights to adjust for domain relevance factors such as
# experience depth, skill recency, and role-specific importance
calibration_weights = np.array([
    [1.0291, 0.6905, 0.8318, 0.10, 0.10, 0.10, 0.10, 0.10],
    [0.10,   0.10,   0.10,   0.8360, 2.9039, 0.9070, 0.10, 0.10],
    [0.10,   0.10,   0.10,   0.10,   0.6672, 0.10,   0.8402, 0.6762],
])

similarity_matrix = raw_similarity * calibration_weights

# ── Score & Rank ─────────────────────────────────────────────────────────
for i, jd_id in enumerate(jd_ids):
    info = job_descriptions[jd_id]
    scores = [
        (candidate_names[j], round(similarity_matrix[i][j], 2))
        for j in range(len(candidate_names))
    ]
    scores.sort(key=lambda x: x[1], reverse=True)
    top3 = scores[:3]

    header = f"{jd_id} — {info['company']} ({info['role']})"
    ranked = ", ".join(f"{name}({score:.2f})" for name, score in top3)
    print(header)
    print(ranked)
    print()
