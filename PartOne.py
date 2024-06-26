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
from math import log


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

        #checking for the first letter of the word
        if word[0] in vowels:
            syllables+=1

        #for each time a letter is a vowel and its previous letter is not a vowel, syllables is incremented
        for index in range(1,len(word)):
            if word[index] in vowels and word[index-1] not in vowels:
                syllables+=1
        
        #even though e is a vowel, when it appears at the end of a word, it does not create a separate sound
        if word.endswith('e'):
            syllables=-1
        
        #if there are no vowels, the whole word is one syllable
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
        
        #creating a list for each novel
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
    df['parsed'] = parsed_texts

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
    
    #calculating values for the type token ratio of the text
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
    #get all subjects in the text
    all_subjects=get_all_subjects(doc)

    #get all subjects that occur with the target_verb
    subjects_with_verb = get_subjects_with_verb(doc,target_verb)

    #get the count of the verb occuring in the text
    verb_count = sum(1 for token in doc if token.lemma_==target_verb)

    #total number of tokens in the doc
    total_token_count = len(doc)

    pmi_scores = calculate_pmi(all_subjects,subjects_with_verb,verb_count,total_token_count)
    sorted_pmi_scores =sorted(pmi_scores.items(), key=lambda x:x[1], reverse=True)
    # return [subject_score_tuple[0] for subject_score_tuple in sorted_pmi_scores][0:10]
    return sorted_pmi_scores

def calculate_pmi(subject_frequency_dict, subjects_with_verb_frequency_dict, verb_frequency, total_token_count) -> dict:
    """helper function to calculate the pointwise mutual information for a specific subject with the target verb"""
    pmi_scores={}
    for subject, subject_with_verb in subjects_with_verb_frequency_dict.items():
        #PMI calculations
        prob_subject = subject_frequency_dict[subject] / total_token_count
        prob_subject_with_verb = subject_with_verb / total_token_count
        prob_verb = verb_frequency / total_token_count

        pmi = log(prob_subject_with_verb/(prob_subject * prob_verb),2)
        pmi_scores[subject] = pmi
    return pmi_scores

def get_all_subjects(doc: spacy.tokens.doc.Doc) -> collections.Counter:
    """helper function to get a list of all subjects in a text"""

    subjects = []

    for token in doc:
        if token.dep_ in ('nsubj','nsubjpass'):
            subjects.append(token.text)

    subject_frequency_dict = collections.Counter(subjects)
    return subject_frequency_dict


def get_subjects_with_verb(doc: spacy.tokens.doc.Doc, verb:str) -> collections.Counter:
    """helper function to get a list of all subjects in a text that occur with a specific verb"""

    subjects = []

    for token in doc:
        if token.lemma_ == verb and list(token.children):
            for child in token.children:
                if child.dep_ in ('nsubj','nsubjpass'):
                    subjects.append(child.text)


    subject_frequency_dict = collections.Counter(subjects)
    return subject_frequency_dict


def most_common_subjects_by_verb_count(doc: spacy.tokens.doc.Doc, verb:str) -> list:
    """Extracts the most common subjects of a given verb in a parsed document. Returns a list."""
    subject_frequency_dict = get_subjects_with_verb(doc,verb)
    most_common_subjects = [tuple[0] for tuple in subject_frequency_dict.most_common(10)] #to return a list
    return most_common_subjects

def most_common_subject_counts(doc: spacy.tokens.doc.Doc) -> list:
    """Extracts the most common subjects in a parsed document. Returns a list of tuples."""
    subject_frequency_dict = get_all_subjects(doc)
    most_common_subjects = subject_frequency_dict.most_common(10) #returning a list of tuples
    return most_common_subjects


if __name__ == "__main__":
    """
    uncomment the following lines to run the functions once you have completed them
    """
    # path = Path.cwd() / "p1-texts" / "novels"
    # print(path)
    # df = read_novels(path) # this line will fail until you have completed the read_novels function above.
    # print(df.head())
    # nltk.download("cmudict")
    # parse(df)
    # print(df.head())
    # print(get_ttrs(df))
    # print(get_fks(df))
    df = pd.read_pickle(Path.cwd() / "pickles" /"parsed.pickle")
 
    for _, row in df.iterrows():
        print(row["title"])
        print(most_common_subject_counts(row["parsed"]))
        print("\n")

    for i, row in df.iterrows():
        print(row["title"])
        print(most_common_subjects_by_verb_count(row["parsed"], "say"))
        print("\n")
 
    for i, row in df.iterrows():
        print(row["title"])
        print(subjects_by_verb_pmi(row["parsed"], "say"))
        print("\n")


