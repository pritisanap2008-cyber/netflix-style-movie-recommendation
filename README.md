# Netflix-Style Movie Recommendation System

Collaborative filtering recommender: predicts movie ratings and builds top-10 lists, using the MovieLens data.

## Data
- MovieLens ml-latest-small (GroupLens, University of Minnesota): 100,836 ratings, 9,724 rated movies, 610 users
- The notebook downloads the data by itself, so the data files are not included here
- Used instead of the Netflix Prize data, which needs a Kaggle login and is too big for free Colab
- Very sparse: only 1.7% of the user-movie table is filled, and about 35% of movies have a single rating

## Method
- Split by time for each user: oldest 80% of ratings for training (80,896), newest 20% for testing (19,940)
- Baselines: global average, movie average, user average, user + movie bias model
- SVD (matrix factorization) with 5, 10, 20 and 40 factors
- Item-based neighbours (cosine similarity, 30 neighbours)
- Metrics: RMSE and MAE for ratings, precision@10 for top-10 lists

## Results

| Method | RMSE | MAE |
|---|---|---|
| Global average | 1.0688 | 0.8360 |
| Movie average | 1.0227 | 0.7867 |
| User average | 0.9648 | 0.7486 |
| Baseline (user + movie bias) | 0.8990 | 0.6926 |
| SVD, 20 factors | 0.8939 | 0.6881 |
| Item-based neighbours | 0.8807 | 0.6730 |

Top-10 lists (591 users): most popular movies 0.056, item-based model 0.044 (precision@10).

## Findings
- The item-based model is best for rating prediction (about 2% lower RMSE than the baseline)
- Most of the gain comes from user and movie biases, because the data is sparse
- For top-10 lists, simply recommending the most popular movies beat the model

## Limitations
- MovieLens small, not the real Netflix data
- Number of SVD factors chosen using the test set
- Precision@10 counts only rated movies, so scores are low
- Recommendations are dominated by movie bias, not personal taste

## Tools
Python, pandas, numpy, scipy, scikit-learn, Google Colab
