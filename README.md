Recommender-systems using matrix-factorization
================

## Matrix factorization

The goal for this project is to produce recommendation using
collaborative filtering (matrix factorization & cosine distance).

Project is made from three parts:

1.  cross-validation, wich contains the tunning grid needed for ALS and
    calculates the accuracy using NDCG

2.  making\_reco, generates recommendation user based (ALS) and item
    based (cosine distance)

3.  api\_reco, which will deliver recommendation for a user using an API

The library used for recommendation is
<https://github.com/rexyai/rsparse> fallowing this excellent post
<http://dsnotes.com/post/2017-05-28-matrix-factorization-for-recommender-systems/>.

The model runs extremly fast, it can generate recommendation in ~ 300
secs for 138286 users for ALS and in ~ 200 secs for 8546 using cosine
distance.