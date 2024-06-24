import nltk
import spacy
from pathlib import Path
import os
import pandas as pd
import re
import string
import pickle
import collections
import spacy.tokens


nlp = spacy.load("en_core_web_sm")
nlp.max_length = 2000000



def fk_level(text: str, dictionary:dict) -> float:
    """Returns the Flesch-Kincaid Grade Level of a text (higher grade is more difficult).
    Requires a dictionary of syllables per word.

    Args:
        text (str): The text to analyze.
        d (dict): A dictionary of syllables per word.

    Returns:
        float: The Flesch-Kincaid Grade Level of the text. (higher grade is more difficult)
    """
    total_syllables = sum([count_syl(word_in_text,dictionary) for word_in_text in text])
    
    total_words = len(nltk.word_tokenize(text))
    total_sentences = len(nltk.sent_tokenize(text))

    flesch_kincaid_grade = 0.39 * (total_words/total_sentences) + 11.8 * (total_syllables/total_words) - 15.59

    return flesch_kincaid_grade

def count_syl(word: str, dictionary: dict) -> int:
    """Counts the number of syllables in a word given a dictionary of syllables per word.
    if the word is not in the dictionary, syllables are estimated by counting vowel clusters

    Args:
        word (str): The word to count syllables for.
        d (dict): A dictionary of syllables per word.

    Returns:
        int: The number of syllables in the word.
    """
    if word in dictionary.keys():
        return len([phenome for phenome in dictionary[word][0] if phenome[-1].isdigit()])
    else:
        syllables = 0
        vowels = 'aeiou'

        if word[0] in vowels:
            syllables+=1

        for index in range(1,len(word)):
            if word[index] in vowels and word[index-1] not in vowels:
                syllables+=1
        
        if word.endswith('e'):
            syllables=-1
        
        if syllables == 0:
            syllables = 1
        
        return syllables

def read_novels(path: Path = Path.cwd() / "p1-texts" / "novels") -> pd.DataFrame:
    """Reads texts from a directory of .txt files and returns a DataFrame with the text, title,
    author, and year"""
    data_list=[]

    for filename in os.listdir(path):

        #extracting the title, author and year from the file title
        title,author,year = filename.split('-')
        title = ' '.join(title.split('_'))
        year = int(year.rstrip('.txt'))

        with open(os.path.join(path, filename)) as file:
            text = file.read()
        
        #creating a for each novel
        item_list = [text,title,author,year]

        data_list.append(item_list)

    #creating a dataframe from list of lists.
    data_dataframe = pd.DataFrame(data=data_list, columns=['text','title','author','year'])
    #returning the sorted dataframe
    return data_dataframe.sort_values(by='year', ignore_index=True)

def parse(df, store_path=Path.cwd() / "pickles", out_name="parsed.pickle") -> pd.DataFrame:
    """Parses the text of a DataFrame using spaCy, stores the parsed docs as a column and writes 
    the resulting  DataFrame to a pickle file"""

    parsed_texts=[]
    for _, row in df.iterrows():
        text = row['text']
        #dividing the file into chunks if it exceeds nlp.max_length
        if len(text) > nlp.max_length:
            chunks = [text[starting_point: starting_point + nlp.max_length] for starting_point in range(0,len(text),nlp.max_length)]
            docs = [nlp(chunk) for chunk in chunks]
        else: 
            docs = nlp(text)
        
        parsed_texts.append(docs)
    
    #adding the parsed text as a column
    df['parsed_text'] = parsed_texts

    #writing out to the pickle file
    pickle_path = os.path.join(store_path, out_name)
    with open(pickle_path, 'wb') as file:
        pickle.dump(df,file)
        
    return df


def nltk_ttr(text: str) -> float:
    """Calculates the type-token ratio of a text. Text is tokenized using nltk.word_tokenize."""
    tokens = nltk.word_tokenize(text)

    #remove punctuations from the tokens
    remove_punc = re.compile('[%s]' % re.escape(string.punctuation))
    tokens = [remove_punc.sub('', token) for token in tokens]
    
    #calculating type token ratio for the text
    total_no_of_tokens = len(tokens)
    total_no_of_types = len(set(tokens))

    #type-token ratio
    ttr = total_no_of_tokens/total_no_of_types

    return ttr
    

def get_ttrs(df):
    """helper function to add ttr to a dataframe"""
    results = {}
    for _, row in df.iterrows():
        results[row["title"]] = nltk_ttr(row["text"])
    return results


def get_fks(df: pd.DataFrame) -> dict:
    """helper function to add fk scores to a dataframe"""
    results = {}
    cmudict = nltk.corpus.cmudict.dict()
    for _, row in df.iterrows():
        results[row["title"]] = round(fk_level(row["text"], cmudict), 4)
    return results


def subjects_by_verb_pmi(doc, target_verb):
    """Extracts the most common subjects of a given verb in a parsed document. Returns a list."""
    pass



def subjects_by_verb_count(doc, verb):
    """Extracts the most common subjects of a given verb in a parsed document. Returns a list."""
    pass


def get_subjects(df: pd.DataFrame):
    """helper function to get subjects for each novel in the dataframe"""
    for _, row in df.iterrows():
        print(row["title"])
        print(subject_counts(row["parsed_text"]))


def subject_counts(doc: spacy.tokens.doc.Doc) -> list:
    """Extracts the most common subjects in a parsed document. Returns a list of tuples."""
    subjects = []

    for token in doc:
        if token.dep_ in ('nsubj','nsubjpass'):
            subjects.append(token.text)

    subject_frequency_dict = collections.Counter(subjects)
    most_common_subjects = subject_frequency_dict.most_common(10)
    return most_common_subjects


if __name__ == "__main__":
    """
    uncomment the following lines to run the functions once you have completed them
    """
    # path = Path.cwd() / "p1-texts" / "novels"
    # df = read_novels(path) # this line will fail until you have completed the read_novels function above.
    # print(df.head())
    # nltk.download("cmudict")
    # parse(df)
    # print(df.head())
    # print(get_ttrs(df))
    # print(get_fks(df))
    df = pd.read_pickle(Path.cwd() / "pickles" /"parsed.pickle")
    get_subjects(df)
    """ 
    for i, row in df.iterrows():
        print(row["title"])
        print(subjects_by_verb_count(row["parsed"], "say"))
        print("\n")

    for i, row in df.iterrows():
        print(row["title"])
        print(subjects_by_verb_pmi(row["parsed"], "say"))
        print("\n")
    """

