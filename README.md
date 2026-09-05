# Book Recommendation Engine (KNN)

A machine learning project that recommends books similar to a given title, built using **K-Nearest Neighbors (KNN)** and **cosine similarity**, trained on the [Book-Crossings dataset](https://www.kaggle.com/datasets/somnambwl/bookcrossing-dataset), 1.1 million ratings across 270,000+ books from 90,000+ users. Completed as part of [freeCodeCamp's Machine Learning with Python certification](https://www.freecodecamp.org/learn/machine-learning-with-python/).

---

## What it does

Given a book title, the model returns the **5 most similar books**, along with a distance score indicating how closely related each recommendation is.

```python
get_recommends("Where the Heart Is (Oprah's Book Club (Paperback))")
```

**Output:**
```python
[
  "Where the Heart Is (Oprah's Book Club (Paperback))",
  [
    ["I'll Be Seeing You", 0.80],
    ['The Weight of Water', 0.77],
    ['The Surgeon', 0.77],
    ['I Know This Much Is True', 0.77],
    ['The Lovely Bones: A Novel', 0.72]
  ]
]
```

Lower distance = more similar. A distance of `0` would mean an identical book.

---

## How it works

1. **Filter the data**: Remove users with fewer than 200 ratings and books with fewer than 100 ratings, to reduce noise and keep only statistically meaningful data.
2. **Reshape the data**: Pivot the ratings into a book × user matrix, where each row is a book and each column is a user's rating for that book.
3. **Convert to a sparse matrix**: Since most users only rate a small fraction of books, the matrix is mostly zeros. A sparse matrix (`scipy.sparse.csr_matrix`) stores this efficiently.
4. **Train a KNN model**: Using `sklearn.neighbors.NearestNeighbors` with **cosine distance**, which measures similarity based on rating patterns rather than raw values.
5. **Recommend**: For any given book, find its 5 nearest neighbors in this "rating space" and return them with their distances.

---

## Tech Stack

- **Python**
- **Pandas**: data cleaning, filtering, and reshaping
- **NumPy**
- **SciPy**: sparse matrix representation
- **Scikit-learn**: K-Nearest Neighbors model

---

## Dataset

The [Book-Crossings dataset](https://www.kaggle.com/datasets/somnambwl/bookcrossing-dataset) contains:
- ~1.1 million ratings (scale: 1–10)
- ~270,000 books
- ~90,000 users

After filtering for statistical significance, the working dataset is reduced to **673 books** and **888 users**.

---

## Running the project

This project runs in Google Colab. Open the notebook, run all cells in order, and call:

```python
get_recommends("Any book title from the dataset")
```

---

## Result

Passed the freeCodeCamp automated test suite, correctly identifying similar books within the expected distance tolerance.
