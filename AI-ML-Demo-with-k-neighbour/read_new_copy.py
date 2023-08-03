import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors

df = pd.read_csv('books_4.csv')


encoder = LabelEncoder()
df['Author'] = encoder.fit_transform(df['Author'])
df['genre'] = encoder.fit_transform(df['genre'])

train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)


neigh = NearestNeighbors(n_neighbors=5)
neigh.fit(train_df[['Book_average_rating', 'Author', 'genre']])


test_book = test_df.iloc[0][['Book_average_rating', 'Author', 'genre']]
distances, indices = neigh.kneighbors([test_book])


for i, index in enumerate(indices[0]):
    book = df.iloc[index]
    print(f"{i+1}. Title: {book['Book Name']}, Author: {book['Author']}, Genre: {encoder.inverse_transform([book['genre']])}")
