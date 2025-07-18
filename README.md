# RISE-internship-project-ML-AI-2025
📨 Project 1: Email Spam Detection

Description:
The Email Spam Detection project aims to classify email messages as either spam or ham (not spam) using supervised machine learning techniques. Spam emails reduce productivity and may pose cybersecurity risks. By training a machine learning model on labeled email data, we can automate the detection process and build an intelligent filtering system.

Logic of the Code:
The project starts by importing the dataset of emails labeled as spam or ham. The emails go through text preprocessing steps: converting to lowercase, removing punctuation, stopwords, and performing tokenization and stemming. The cleaned text is transformed into numerical form using TF-IDF Vectorization, which reflects word importance.

A train-test split is performed to separate data into training and testing sets. Then, models like Naive Bayes or SVM (Support Vector Machine) are trained on the data. After training, the model predicts labels on the test set. The output is evaluated using accuracy, precision, recall, and confusion matrix, confirming the model's ability to detect spam with over 90% accuracy.


💳 Project 3: Loan Eligibility Predictor

Description:
The Loan Eligibility Predictor project focuses on building a predictive model to determine whether a loan applicant is eligible for loan approval. Banks and financial institutions use such predictive tools to assess risk before granting loans. The project uses applicant features such as age, income, credit score, education, employment, and marital status.

Logic of the Code:
The dataset is loaded and preprocessed. Missing values are handled, and categorical variables like gender or education are encoded using techniques like Label Encoding or One-Hot Encoding. Numerical features are scaled if needed.

Next, the dataset is split into training and testing sets. Machine learning models such as Logistic Regression and Random Forest Classifier are trained to learn patterns that distinguish approved from rejected applications. The model is evaluated using performance metrics like confusion matrix, ROC curve, accuracy, and F1-score. The result is an interactive or backend-integrated prediction model capable of simulating real-world loan approval systems.


📈 Project 6: Stock Price Prediction (LSTM)

Description:
This project predicts future stock prices using LSTM (Long Short-Term Memory) neural networks, a special type of Recurrent Neural Network (RNN) suited for time-series forecasting. The model is trained on historical data of a stock (e.g., Apple - AAPL) to learn past trends and make future predictions. It's a real-world application of AI in finance and investment analysis.

Logic of the Code:
The process begins with downloading historical stock price data using the yfinance library. Only the ‘Close’ prices are selected for prediction. The data is normalized using MinMaxScaler to bring all values between 0 and 1 — crucial for neural networks.

To prepare for LSTM input, a sliding window technique is used to create sequences (e.g., 60 past days to predict the 61st). These sequences become the input X, and the next value becomes the label y. After reshaping the data into 3D (samples, time steps, features), the dataset is split into training and testing sets.

An LSTM model is built using TensorFlow/Keras, typically with two LSTM layers and one Dense output layer. After training for several epochs, the model predicts stock prices on the test set. The predictions and actual prices are inverse-transformed back to real stock price values. Finally, a graph is plotted showing actual vs predicted prices, providing a visual comparison of model performance.
