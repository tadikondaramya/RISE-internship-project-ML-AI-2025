import pandas as pd
import nltk
import string
import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.utils import resample

# Download NLTK data
nltk.download('stopwords')
nltk.download('punkt')

# Step 1: Load and clean dataset
url = "https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv"
data = pd.read_csv(url, sep='\t', names=['result', 'emails'])
data.drop_duplicates(inplace=True)

# Step 2: Balance the dataset
ham = data[data['result'] == 'ham']
spam = data[data['result'] == 'spam']
spam_upsampled = resample(spam, replace=True, n_samples=len(ham), random_state=42)
data_balanced = pd.concat([ham, spam_upsampled])

# Step 3: Preprocessing
stop_words = set(stopwords.words('english'))
ps = PorterStemmer()

def preprocess(text):
    text = text.lower()
    text = re.sub(r"http\S+", " link ", text)  # Replace URLs with the token "link"
    tokens = word_tokenize(text)
    clean_tokens = [ps.stem(word) for word in tokens 
                    if word not in stop_words and word not in string.punctuation]
    return ' '.join(clean_tokens)

data_balanced['transform_text'] = data_balanced['emails'].apply(preprocess)

# Step 4: Encode and Vectorize
encoder = LabelEncoder()
data_balanced['result'] = encoder.fit_transform(data_balanced['result'])  # spam = 1, ham = 0
tfidf = TfidfVectorizer(max_features=3000)
X = tfidf.fit_transform(data_balanced['transform_text']).toarray()
y = data_balanced['result']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 5: Train Model
model = MultinomialNB()
model.fit(X_train, y_train)

# Step 6: Evaluate
y_pred = model.predict(X_test)
print("Model Accuracy: {:.2f}".format(accuracy_score(y_test, y_pred)))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Precision Score: {:.10f}".format(precision_score(y_test, y_pred)))

# Step 7: Predict on user input
def predict_message(msg):
    processed = preprocess(msg)
    msg_vector = tfidf.transform([processed]).toarray()
    pred = model.predict(msg_vector)
    return "Spam" if pred[0] == 1 else "Ham"

# === User input for prediction ===
print("\n=== Email Spam Classifier ===")
for _ in range(2):
    user_input = input("Enter an email message to classify: ")
    prediction = predict_message(user_input)
    print("The email is predicted as:", prediction)