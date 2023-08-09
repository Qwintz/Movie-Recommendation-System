import pandas as pd # отбработка таблиц с данными
import numpy as np # обработка матриц
from scipy.sparse.linalg import svds # метод разложения матриц
from sklearn.metrics import mean_absolute_error, mean_squared_error # метрики для оценки точности
from tabulate import tabulate # красивый вывод в консоль

users = pd.read_csv('datasets/users.dat', sep = '::', names = ['userId', 'gender', 'age', 'occupation', 'zip'], engine = 'python')
movies = pd.read_csv('datasets/movies.dat', sep = '::', names = ['movieId', 'name', 'genres'], engine = 'python')
reviews = pd.read_csv('datasets/ratings.dat', sep = '::', names = ['userId', 'movieId', 'rating', 'time'], engine = 'python')

ages = {
    1: 'Under 18',
    18: '18 - 24',
    25: '25 - 34',
    35: '35 - 44',
    45: '45 - 49',
    50: '50 - 55',
    56: '56 +'
}
occupations = {
    0: 'Not specified',
    1: 'Academic / Educator',
    2: 'Artist',
    3: 'Clerical / Admin',
    4: 'College / Grad Student',
    5: 'Customer Service',
    6: 'Doctor / Health Care',
    7: 'Executive / Managerial',
    8: 'Farmer',
    9: 'Homemaker',
    10: 'Student',
    11: 'Lawyer',
    12: 'Programmer',
    13: 'Retired',
    14: 'Sales / Marketing',
    15: 'Scientist',
    16: 'Self-Employed',
    17: 'Technician / Engineer',
    18: 'Tradesman / Craftsman',
    19: 'Unemployed',
    20: 'Writer'
}
genders = {
    'M': 'Male',
    'F': 'Female'
}

def main():
    # Очистка данных
    movies['release_year'] = movies['name'].str.extract(r'(?:\((\d{4})\))?\s*$', expand = False) # Достаем год из названия фильма
    users.drop(['zip'], axis = 1, inplace = True)
    reviews.drop(['time'], axis = 1, inplace = True)
    users['age'] = users['age'].map(ages)
    users['occupation'] = users['occupation'].map(occupations)
    users['gender'] = users['gender'].map(genders)

    # Подготовка общего набора
    final_df = reviews.merge(movies, on = 'movieId', how = 'left').merge(users, on = 'userId', how = 'left')

    # Создание матрицы оценок
    final_df_matrix = final_df.pivot(index = 'userId', columns = 'movieId', values = 'rating').fillna(0)
    user_ratings_mean = np.mean(final_df_matrix.values, axis = 1)
    ratings_demeaned = final_df_matrix.values - user_ratings_mean.reshape(-1, 1)
    U, sigma, Vt = svds(ratings_demeaned, k = 50)
    sigma = np.diag(sigma)
    all_user_predicted_ratings = np.dot(np.dot(U, sigma), Vt) + user_ratings_mean.reshape(-1, 1)
    predictions = pd.DataFrame(all_user_predicted_ratings, columns = final_df_matrix.columns)

    # Оценка точности
    print(f'MAE = {mean_absolute_error(final_df_matrix, predictions)}')
    print(f'RMSE = {np.sqrt(mean_squared_error(final_df_matrix, predictions))}')

    # Вывод отчетов
    while True:
        uid = int(input('User id? '))
        user = users[users.userId == uid]
        already_rated, recomendations = recomend_movies(predictions, uid, movies, reviews, 10)
        already_rated.drop(['userId'], axis = 1, inplace = True)
        
        print('User information:', tabulate(user, headers='keys', tablefmt='psql'), sep = '\n')
        print('Already reted movies:', tabulate(already_rated, headers='keys', tablefmt='psql'), sep = '\n')
        print('Recomended movies:', tabulate(recomendations, headers='keys', tablefmt='psql'), sep = '\n')

# Получить набор рекомендаций
def recomend_movies(predictions, userId, movies, reviews, count):
    sorted_user_predictions = predictions.iloc[userId - 1].sort_values(ascending = False)
    user_data = reviews[reviews.userId == userId]
    user_full = user_data.merge(movies, on = 'movieId', how = 'left').sort_values(['rating'], ascending = False)
    recommendations = (
        movies[~movies['movieId'].isin(user_full['movieId'])]
        .merge(pd.DataFrame(sorted_user_predictions).reset_index(), left_on = 'movieId', right_on = 'movieId', how = 'left')
        .rename(columns = {userId - 1: 'Predictions'})
        .sort_values('Predictions', ascending = False)
        .iloc[:count, :-1]
    )
    return user_full.head(10), recommendations.sort_values('release_year', ascending = False)

if __name__ == '__main__':
    main()