import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer


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


if __name__ == "__main__":
    dataframe = pd.read_csv('p2-texts/hansard40000.csv')
    
    #Part a
    dataframe = get_clean_dataframe(dataframe)
    
    print(f"Shape of the dataframe: {dataframe.shape}")



