import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import f1_score, classification_report
import re
import string
import spacy
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import time

nlp = spacy.load("en_core_web_sm")
nltk.download('stopwords')
nltk.download('wordnet')

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def get_clean_dataframe(dataframe: pd.DataFrame):

    #replacing the value of Labour (Co-op) to Labour in the party column
    dataframe['party'] = dataframe['party'].replace('Labour (Co-op)','Labour')

    #finding the 4 most common values in the party column
    #excluding the speaker value in the party column to not include it in the top 4
    dataframe = dataframe[dataframe['party']!='Speaker']
    most_common_parties = dataframe['party'].value_counts().head(4).keys().to_numpy()

    #only keeping the rows where the value in the party column is in the 4 most common parties
    dataframe = dataframe[dataframe['party'].isin(most_common_parties)]

    #only keeping the rows where speech class is  speech
    dataframe = dataframe[dataframe['speech_class']=='Speech']

    #only keeping the rows where the length of the speech column in greater than 1500 characters
    dataframe = dataframe[dataframe['speech'].str.len() >= 1500]

    return dataframe

def get_data_for_training(dataframe: pd.DataFrame,ngrams=(1,1), custom_tokenizer=None, custom_stop_words = "english"):
        
        #to have an array of the party values as the target variable
        y = dataframe['party'].to_numpy()
        x = dataframe['speech'].to_numpy()

        #first splitting the data into train and test
        x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.3, stratify=y,random_state=99)

        #initialising the vectorizer
        vectorizer = TfidfVectorizer(stop_words=custom_stop_words, max_features=4000, ngram_range=ngrams, tokenizer=custom_tokenizer)

        #extracting features from the train and test dataset using the same vectorizer
        x_train_vectors = vectorizer.fit_transform(x_train)
        x_test_vectors = vectorizer.transform(x_test)

        return x_train_vectors, y_train, x_test_vectors,y_test

def train_random_forest_classifier(x_train,y_train,x_test,y_test,class_weight = 'balanced'):
    classifier = RandomForestClassifier(n_estimators=400, class_weight=class_weight)
    classifier.fit(x_train, y_train)

    y_predictions = classifier.predict(x_test)
     
    print('Random Forest Classification Report')
    print(classification_report(y_test,y_predictions))
    print(f"Macro-Average F1 Score: {f1_score(y_test, y_predictions, average='macro')}\n")
    
def train_SVM_classifier(x_train,y_train,x_test,y_test):
    classifier = SVC(kernel='linear')
    classifier.fit(x_train, y_train)

    y_predictions = classifier.predict(x_test)
     
    print('SVM Classification Report')
    print(classification_report(y_test,y_predictions))
    print(f"Macro-Average F1 Score: {f1_score(y_test, y_predictions, average='macro')}\n")

def custom_tokenizer(text):
    #using spacy's nlp to parse the text and then return tokens
    doc = nlp(text)
    tokens = []

    #preserve named-entities
    for ent in doc.ents:
        tokens.append(ent.text)
    for token in doc:
        tokens.append(token.text)

    #convert all to lower case
    tokens = [token.lower() for token in tokens]

    # #remove punctuation
    # punc_pattern = re.compile('[%s]' % re.escape(string.punctuation))
    # # replaces all punctuation with nothing
    # tokens = [punc_pattern.sub('', token) for token in tokens]

    #keeping only alphanumeric characters
    tokens = [token for token in tokens if token.isalnum()]

    #lemmatize the tokens
    tokens = [lemmatizer.lemmatize(token) for token in tokens if token not in stop_words]

    return tokens



if __name__ == "__main__":
    dataframe = pd.read_csv('p2-texts/hansard40000.csv')
    
    #Part a
    dataframe = get_clean_dataframe(dataframe)
    print(f"Shape of the dataframe: {dataframe.shape}")

    #ADDITIONAL, defining class_weights to improve the performance of the classifiers
    party_counts = dataframe.value_counts('party')
    class_weights = {}
    total = dataframe.shape[0]

    for index,value in party_counts.items():
        class_weights[index] = total/value


    # #Part b
    x_train, y_train, x_test, y_test = get_data_for_training(dataframe)

    #Part c
    start = time.perf_counter()
    train_random_forest_classifier(x_train,y_train,x_test,y_test,class_weights)
    train_SVM_classifier(x_train,y_train,x_test,y_test)
    end = time.perf_counter()
    print(f'Time took to train the Random forest and SVM classifier: {end - start} second(s)\n\n')

    #Part d
    start = time.perf_counter()
    x_train_with_grams, y_train_with_grams, x_test_with_grams, y_test_with_grams = get_data_for_training(dataframe,(1,3))
    train_random_forest_classifier(x_train_with_grams, y_train_with_grams, x_test_with_grams, y_test_with_grams,class_weights)
    train_SVM_classifier(x_train_with_grams, y_train_with_grams, x_test_with_grams, y_test_with_grams)
    end = time.perf_counter()
    print(f'Time took to train the Random forest and SVM classifier with bigrams and trigrams: {end - start} second(s)\n\n')

    # Part e
    # this is done to convert the stop words into the same pattern as our tokens to avoid the "Your stop_words may be inconsistent with your preprocessing" warning
    processes_stop_words = list(set(custom_tokenizer(' '.join(stop_words))))

    start = time.perf_counter()
    x_train_with_tokenizer, y_train_with_tokenizer, x_test_with_tokenizer, y_test_with_tokenizer = get_data_for_training(dataframe,(1,3), custom_tokenizer, processes_stop_words)
    train_random_forest_classifier(x_train_with_tokenizer, y_train_with_tokenizer, x_test_with_tokenizer, y_test_with_tokenizer,class_weights)
    train_SVM_classifier(x_train_with_tokenizer, y_train_with_tokenizer, x_test_with_tokenizer, y_test_with_tokenizer)
    end = time.perf_counter()
    print(f'Time took to train the Random forest and SVM classifier with custom tokenizer: {end - start} second(s)\n\n')