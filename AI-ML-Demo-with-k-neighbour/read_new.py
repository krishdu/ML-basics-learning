from scipy.sparse import csr_matrix
import pickle
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors


books = pd.read_csv('books_4.csv', sep=",", error_bad_lines=True)
#print(books.shape)
# print(books.head())

# ratings_with_books = books.merge(books, on='Publishing Year')
# print(ratings_with_books.head())
# ratings_with_books.to_csv('file1.csv')

# number_rating = books.groupby('Book Name')['Book_average_rating'].count().reset_index()
# print(number_rating)


book_pivot = books.pivot_table(columns='Author', index='Book Name', values= 'Publishing Year')
#print(book_pivot.shape)
book_pivot.fillna(0, inplace=True)

book_sparse = csr_matrix(book_pivot)
type(book_sparse)
#print(book_sparse.shape)



model = NearestNeighbors(algorithm= 'brute')
model.fit(book_sparse)

pickle.dump(book_pivot,open('artifacts/book_pivot.pkl','wb'))
pickle.dump(model,open('artifacts/model.pkl','wb'))

# FINDING SUGGESTIONS USING EXACT NUMBER
# distance, suggestion = model.kneighbors(book_pivot.iloc[237,:].values.reshape(1,-1), n_neighbors=3 )
# print(distance)
# print(suggestion)

print(book_pivot)
# for i in range(len(suggestion)):
#     print(book_pivot.index[suggestion[i]])

book_name = "Voyager"

# # FINDING SUGGESTION USING BOOK NAME
# # IT WILL FIND THE BOOK OF SAME AUTHOR
# # def recommend_book(book_name):
book_id = np.where(book_pivot.index == book_name)[0][0]
#print(book_pivot.iloc[book_id,:].values.reshape(1,-1))
# distance, suggestion = model.kneighbors(book_pivot.iloc[book_id,:].values.reshape(1,-1), n_neighbors=8 )

# for i in range(len(suggestion)):
#         books = book_pivot.index[suggestion[i]]
#         for j in books:
#             if j == book_name:
#                 print(f"You searched '{book_name}'\n")
#                 print("The suggestion books are: \n")
#             else:
#                 print(j)