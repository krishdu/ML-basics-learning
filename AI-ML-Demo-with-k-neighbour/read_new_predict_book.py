# using KNeighborsClassifier 
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier
df = pd.read_csv('books_4.csv')
df['language_code'].fillna('eng', inplace=True)

features = ['language_code', 'Author', 'genre']

train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

# Define the features

train_df.fillna('eng', inplace=True)
test_df.fillna('eng', inplace=True)
#print(test_df[features].iloc[0])

# Vectorize the text features using TF-IDF
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(train_df[features].apply(lambda x: ' '.join(x), axis=1))

# Fit the KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors=5)
classifier.fit(X, train_df['Book Name'])


test_book = test_df.iloc[0][features]
user_vector = vectorizer.transform([' '.join(test_book)])

#print(f"Your Choose ---- \n {test_df.iloc[0]} \n Recommendation -- ")
#print(classifier.predict(user_vector))
#print(test_df[features])
y_pred = classifier.predict(vectorizer.transform(test_df[features].apply(lambda x: ' '.join(x), axis=1)))



# distances, indices = classifier.kneighbors(user_vector)

# for i, index in enumerate(indices[0]):
#     book = df.iloc[index]
#     print(f"{i+1}. Title: {book['Book Name']}, Author: {book['Author']}, Genre: {book['genre']}")
