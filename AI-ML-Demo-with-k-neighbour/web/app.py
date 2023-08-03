from flask import Flask,render_template,request
import pickle
import numpy as np

book_pivot = pickle.load(open('../artifacts/book_pivot.pkl','rb'))
model = pickle.load(open('../artifacts/model.pkl','rb'))

app = Flask(__name__, template_folder='templates')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/recommend')
def recommend_ui():
    return render_template('recommend.html')

@app.route('/recommend_books',methods=['post'])
def recommend():
    book_name = request.form.get('user_input')
    book_id = np.where(book_pivot.index == book_name)[0][0]
    distance, suggestion = model.kneighbors(book_pivot.iloc[book_id,:].values.reshape(1,-1), n_neighbors=6 )

    data = []
    for i in  range(len(suggestion)):
        books = book_pivot.index[suggestion[i]]
        for j in books:
            if j != book_name:
                data.append(j)
        print(books)
    # print(data)

    return render_template('recommend.html', data=data)

if __name__ == '__main__':
    app.run(debug=True)