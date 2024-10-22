import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error

# Завантаження набору даних Iris
np.random.seed(2021)
iris = load_iris()
X, y, labels, feature_names  = iris.data, iris.target, iris.target_names, iris['feature_names']
df_iris= pd.DataFrame(X, columns=feature_names) 
df_iris['label'] =  y
features_dict = {k: v for k, v in enumerate(labels)}
df_iris['label_names'] = df_iris.label.apply(lambda x: features_dict[x])

# Розділяємо на тренувальні та тестові вибірки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Нормалізуємо дані
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Змінні для збереження найкращого K і найкращого результату
k_best = None
score_best = float('inf')

# Пошук найкращого значення K
for k in range(1, 21):
    knn_regressor = KNeighborsRegressor(n_neighbors=k)
    knn_regressor.fit(X_train, y_train)
    y_pred = knn_regressor.predict(X_test)
    score = mean_squared_error(y_test, y_pred)
    
    print(f'K={k}, MSE={score}')
    
    if score < score_best:
        score_best = score
        k_best = k

print('The best k = {} , score = {}'.format(k_best, score_best))

import matplotlib.pyplot as plt

k_values = list(range(1, 21))
mse_values = []

for k in k_values:
    knn_regressor = KNeighborsRegressor(n_neighbors=k)
    knn_regressor.fit(X_train, y_train)
    y_pred = knn_regressor.predict(X_test)
    score = mean_squared_error(y_test, y_pred)
    mse_values.append(score)

plt.plot(k_values, mse_values, marker='o')
plt.xlabel('K')
plt.ylabel('Mean Squared Error')
plt.title('KNN Regressor Performance')
plt.show()

from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
cmap_bold = ListedColormap(['blue','#FFFF00','black','green'])

np.random.seed= 2021
X_D2, y_D2 = make_blobs(n_samples = 300, n_features = 2, centers = 8,
                       cluster_std = 1.3, random_state = 4)
y_D2 = y_D2 % 2
plt.figure()
plt.title('Sample binary classification problem with non-linearly separable classes')
plt.scatter(X_D2[:,0], X_D2[:,1], c=y_D2,
           marker= 'o', s=30, cmap=cmap_bold)
plt.show()