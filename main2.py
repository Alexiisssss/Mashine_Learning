# Задача 2. Обнаружение болезни паркинсона с помощью XGBoost
# C помощью Data Science предсказать заболевание паркинсона на ранней стадии, используя алгоритм машинного обучения XGBoost и библиотеку sklearn для нормализации признаков.
#
# Использовать датасет UCI ML Parkinsons. Описание признаков и меток датасета представлены здесь. Требуется помимо создания самой модели получить ее точность на тестовой выборке. Выборки делить в соотношении 80% обучающая, 20% - тестовая.
#
# Дополнительные баллы можно получить, если получить точность более 95%.

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

# Загрузка данных
data = pd.read_csv("dataset/parkinsons.data")

# Разделение на признаки и метки
X = data.drop(['name', 'status'], axis=1)
y = data['status']

# Разделение на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Нормализация признаков
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Обучение модели XGBoost
model = XGBClassifier()
model.fit(X_train_scaled, y_train)

# Предсказание на тестовой выборке
y_pred = model.predict(X_test_scaled)

# Оценка качества модели
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)


# На данной модели точность 94.8 %. Пробовал разными путями, но к сожалению больше 94.8 не получается, только меньше. Так что, дерзайте!)