from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import math


def find_best_k(X, max_k=10):
    best_k = 2
    best_score = -1

    for k in range(2, min(max_k, X.shape[0])):
        try:
            kmeans = KMeans(n_clusters=k, random_state=42)
            labels = kmeans.fit_predict(X)
            score = silhouette_score(X, labels)

            if score > best_score:
                best_score = score
                best_k = k
        except:
            continue

    return best_k


def group_requirements(requirements, n_clusters=None, batch_size=None):
    """
    Advanced grouping:
    - TF-IDF clustering
    - Optional batch processing
    - Auto cluster selection
    """

    if not requirements:
        return [], []

    # =========================
    # 🔥 BATCH PROCESSING MODE
    # =========================
    if batch_size:
        all_groups = []
        all_labels = []

        for i in range(0, len(requirements), batch_size):
            batch = requirements[i:i + batch_size]

            groups, _ = group_requirements(batch, n_clusters=None)  # recursive call

            all_groups.extend(groups)
            all_labels.extend([None] * len(groups))  # Placeholder for labels

        return all_groups, all_labels

    # =========================
    # NORMAL PROCESSING
    # =========================
    vectorizer = TfidfVectorizer(stop_words='english')
    X = vectorizer.fit_transform(requirements)

    # 🔥 AUTO CLUSTER SELECTION
    if not n_clusters:
        n_clusters = find_best_k(X)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(X)

    groups = {}
    for i, label in enumerate(labels):
        groups.setdefault(int(label), []).append(requirements[i])

    return list(groups.values()), labels.tolist()