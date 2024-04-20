# Задача 1. Обнаружение фальшивых новостей
# Фальшивые новости — это ложная информация, распространяемая через социальные сети и другие сетевые СМИ для достижения политических или идеологических целей.
#
# Используя библиотеку sklearn построить модель классического машинного обучения, которая может с высокой точностью более 90% определять, является ли новость реальной (REAL） или фальшивой（FAKE).
#
# Самостоятельно изучить и применить к задаче TfidfVectorizer для извлечения признаков из текстовых данных и PassiveAggressiveClassifier.
#
# Использовать датасет для обучения (fake_news.csv).
#
# Построить матрицу ошибок (confusion matrix).


import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Загрузка данных из файла
df = pd.read_csv('dataset/fake_news.csv')

# Подготовка данных
df['text'] = df['text'].str.lower()  # Приведение текста к нижнему регистру

# Разделение данных на признаки (X) и целевую переменную (y)
X = df['text']
y = df['label']

# Разделение данных на тренировочный и тестовый наборы
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Инициализация TfidfVectorizer и преобразование текстовых данных в числовые признаки
tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
tfidf_train = tfidf_vectorizer.fit_transform(X_train)
tfidf_test = tfidf_vectorizer.transform(X_test)

# Инициализация PassiveAggressiveClassifier и обучение модели
pac = PassiveAggressiveClassifier(max_iter=50)
pac.fit(tfidf_train, y_train)

# Оценка модели
accuracy = pac.score(tfidf_test, y_test)
print(f'Accuracy: {accuracy:.2f}')

# Построение матрицы ошибок
y_pred = pac.predict(tfidf_test)
conf_matrix = confusion_matrix(y_test, y_pred)

# Визуализация матрицы ошибок
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['FAKE', 'REAL'], yticklabels=['FAKE', 'REAL'])
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show()
