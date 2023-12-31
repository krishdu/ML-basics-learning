import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from keras.layers import Input, Embedding, Flatten, Dot, Dense
from keras.models import Model, load_model

dataset = pd.read_csv('ratings.csv')
train, test = train_test_split(dataset, test_size=0.2, random_state=42)
n_users = len(dataset.user_id.unique())
n_books = len(dataset.book_id.unique())

print(n_users)

# book_input = Input(shape=[1], name="Book-Input")
# book_embedding = Embedding(n_books+1, 5, name="Book-Embedding")(book_input)
# book_vec = Flatten(name="Flatten-Books")(book_embedding)

# user_input = Input(shape=[1], name="User-Input")
# user_embedding = Embedding(n_users+1, 5, name="User-Embedding")(user_input)
# user_vec = Flatten(name="Flatten-Users")(user_embedding)

# prod = Dot(name="Dot-Product", axes=1)([book_vec, user_vec])
# model = Model([user_input, book_input], prod)
# model.compile('adam', 'mean_squared_error')

# history = model.fit([train.user_id, train.book_id], train.rating, epochs=10, verbose=1)
# model.save('regression_model.h5')

# # Extract embeddings
# book_em = model.get_layer('Book-Embedding')
# book_em_weights = book_em.get_weights()[0]


model = load_model('regression_model.h5')
# Creating dataset for making recommendations for the first user
book_data = np.array(list(set(dataset.book_id)))
user = np.array([1 for i in range(len(book_data))])
predictions = model.predict([user, book_data])
predictions = np.array([a[0] for a in predictions])
recommended_book_ids = (-predictions).argsort()[:5]
print(recommended_book_ids)
print(predictions[recommended_book_ids])

#Get books details 
books = pd.read_csv('books.csv')
#books.head()
print(books[books['id'].isin(recommended_book_ids)])