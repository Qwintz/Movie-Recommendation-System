# Movie Recommendation System



## Introduction

The Movie Recommendation System based on Collaborative Filtering is a project developed as part of the "Artificial Intelligence" course. It is written in Python and utilizes datasets from MovieLens. The system leverages collaborative filtering algorithms to analyze user ratings and preferences and generate personalized movie recommendations. By studying patterns and relationships among users, it can suggest movies that are likely to align with a user's interests. Users have the ability to create profiles, rate movies, and receive tailored recommendations based on their individual tastes. The system aims to enhance the movie-watching experience by providing users with relevant and enjoyable movie suggestions.



## How it works

The implemented algorithm of joint filtering of movie recommendations works as follows:

Collect data from users who have rated movies. This includes the user’s ratings for each movie and the titles of the movies they have seen.

Calculate the cosine similarity between each user and each movie.
![Cosine similarity formula](https://i.postimg.cc/tJkQ9whn/cosine.png)
The cosine similarity measures the similarity between two vectors, in this case, the user’s rating vector and the movie’s vector.

Perform SVD on the user-movie rating matrix.
![SVD](https://i.postimg.cc/VsWc9r3s/svd.png)
SVD decomposes a matrix into three matrices: a left singular matrix, a right singular matrix, and a diagonal matrix of singular values.

Use the left singular matrix to create a user-user matrix. Each row of this matrix represents a user, and each column represents another user. The value in each cell represents the cosine similarity between the corresponding users.

Apply the k-nearest neighbor (KNN) algorithm to the user-user matrix to generate recommendations. The KNN algorithm takes into account the similarity between users to make recommendations.

Finally, rank the recommendations based on the cosine similarity scores between users and movies and present the top recommendations to the user.

## Example
![Example](https://i.postimg.cc/fTWQ8WNc/example.png)
