from flask import Flask, request, render_template
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import pickle

app = Flask(__name__)
cv = pickle.load(open('count_vector.pkl', 'rb'))
feature_names = pickle.load(open('feature_names.pkl', 'rb'))
tfidf = pickle.load(open('tfidf.pkl', 'rb'))

# creating a list of custom stopwords
stop_words = set(stopwords.words('english'))
new_words = ["fig", "figure", "image", "sample", "using",
             "show", "result", "large", "also", "one", "two",
             "three", "four", "five", "six", "seven", "eight", "nine"
            ]
stop_words = list(stop_words.union(new_words))

# Custom Functions

def preprocessing_text(txt):
    txt = txt.lower()
    txt = re.sub(r'<.*?>',' ' ,txt)  # replace < > . * ? with space in txt
    txt = re.sub(r'[^a-zA-Z]', ' ', txt) # save only  characters
    txt = nltk.word_tokenize(txt) # Tokenization
    txt = [word for word in txt if word not in stop_words] # removing stop words
    txt = [word for word in txt if len(word)>=3]
    stemming = PorterStemmer()
    txt = [stemming.stem(word) for word in txt]
    return ' '.join(txt)


def get_keywords(docs, topN=10): # user doc
    # getting words count and importance

    # Word Counts for User docs
    docs_wc = tfidf.transform(cv.transform([docs]))  # finding word count

    # Sorting Sparse Matrix Coordinates
    docs_wc = docs_wc.tocoo()  # coordinates; col name and data
    tuples = zip(docs_wc.col, docs_wc.data)
    sorted_items = sorted(tuples, key=lambda x:(x[1], x[0], reversed==True))

    # Extract top 10 keywords
    sorted_items = sorted_items[:topN]
    score_vals = []
    features_val = []
    for idx, score in sorted_items:
        score_vals.append(round(score, 3))
        features_val.append(feature_names[idx])

    # final result
    results = {}
    for idx in range(len(features_val)):
        results[features_val[idx]] = score_vals[idx]
    return results


# Routes
@app.route('/')
def index():
    return render_template('index.html')

# Extract Keywords
@app.route('/extract_keywords', methods=['POST', 'GET'])
def extract_keywords():
    file = request.files['file']
    if file.filename == '':
        return render_template('index.html', error = 'no file selected')
    if file:
        file = file.read().decode('utf-8', errors= 'ignore')
        cleaned_file = preprocessing_text(file)
        keywords = get_keywords(cleaned_file, 20)
        return render_template('keywords.html', keywords=keywords)

# Search Keywords
@app.route('/search_keywords', methods=['POST', 'GET'])
def search_keywords():
    search_keyword = request.form['search']
    if search_keyword:
        keywords = []
        for keyword in feature_names:
            if search_keyword.lower()in keyword.lower():
                keywords.append(keyword)
                if len(keywords) ==20:  # limit to 20 keywords
                    break
        print(keywords)
        return render_template('keywords_list.html', keywords=keywords)
    return render_template('index.html')




if __name__ == "__main__":
    app.run(debug=True)
