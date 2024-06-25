import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import f1_score, classification_report

def get_clean_dataframe(dataframe: pd.DataFrame):

    #replacing the value of Labour (Co-op) to Labour in the party column
    dataframe['party'] = dataframe['party'].replace('Labour (Co-op)','Labour')

    #finding the 4 most common values in the party column
    #excluding the speaker value in the party column to not include in the top 4
    dataframe = dataframe[dataframe['party']!='Speaker']
    most_common_parties = dataframe['party'].value_counts().head(4).keys().to_numpy()

    #only keeping the rows where the value in the party column is in the 4 most common parties
    dataframe = dataframe[dataframe['party'].isin(most_common_parties)]

    #only keeping the rows where speech class is  speech
    dataframe = dataframe[dataframe['speech_class']=='Speech']

    #only keeping the rows where the length of the speech column in greater than 1500 characters
    dataframe = dataframe[dataframe['speech'].str.len() >= 1500]

    return dataframe

def get_data_for_training(dataframe: pd.DataFrame,ngrams=(1,1)):
        
        #to have an array of the party values as the target variable
        y = dataframe['party'].to_numpy()
        x = dataframe['speech'].to_numpy()

        #first splitting the data into train and test
        x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.3, stratify=y,random_state=99)

        #initialising the vectorizer
        vectorizer = TfidfVectorizer(stop_words="english", max_features=4000, ngram_range=ngrams)

        #extracting features from the train and test dataset using the same vectorizer
        x_train_vectors = vectorizer.fit_transform(x_train)
        x_test_vectors = vectorizer.transform(x_test)

        return x_train_vectors, y_train, x_test_vectors,y_test

def train_random_forest_classifier(x_train,y_train,x_test,y_test):
    classifier = RandomForestClassifier(n_estimators=400, class_weight='balanced')
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

if __name__ == "__main__":
    dataframe = pd.read_csv('p2-texts/hansard40000.csv')
    
    #Part a
    dataframe = get_clean_dataframe(dataframe)
    print(f"Shape of the dataframe: {dataframe.shape}")

    #Part b
    x_train, y_train, x_test, y_test = get_data_for_training(dataframe)

    #Part c
    train_random_forest_classifier(x_train,y_train,x_test,y_test)
    train_SVM_classifier(x_train,y_train,x_test,y_test)

    #Part d
    x_train_with_grams, y_train_with_grams, x_test_with_grams, y_test_with_grams = get_data_for_training(dataframe,(1,3))
    train_random_forest_classifier(x_train_with_grams, y_train_with_grams, x_test_with_grams, y_test_with_grams)
    train_SVM_classifier(x_train_with_grams, y_train_with_grams, x_test_with_grams, y_test_with_grams)
