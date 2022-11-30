import numpy as np
import pandas as pd
import pdb
import ast
import os

from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from nltk.tokenize import word_tokenize
import nltk
nltk.download('punkt')

def make_dirs(directory):
    '''
    make directories, if not exist
    '''
    if not os.path.exists(directory):
        os.makedirs(directory)
        print (f'Directory created: {directory}')
    else:
        print ('Directory already exists!')
    return

def flatten(list_of_lists, level=2):
    if level == 2:
        flattened_list = [
            j for i in list_of_lists 
                for j in i
        ]
    elif level == 3:
        flattened_list = [
            k for i in list_of_lists
                for j in i
                    for k in j
        ]
    return flattened_list

def firsts(list_l):
    return [i[0] for i in list_l]

def lasts(list_l):
    return [i[-1] for i in list_l]

def preprocess_counts_by_year(counts_by_year, from_publication_year, to_current_year):
    '''
    Creates a list tuples (year_i, citations_accumulated_in_year_i)
    from `from_publication_year` to `to_current_year`
    '''
    
    years = list(range(from_publication_year, to_current_year + 1))
    citations = [0]*len(years)
    for year_i, cc_i in counts_by_year:
        if year_i >= from_publication_year:
            citations[year_i-from_publication_year] = cc_i
    return list(zip(years, citations))    

def preprocess_works(works, from_year, to_year): 
    '''
    work related preprocessing
    '''
    # remove duplicate works (based on index 'id')
    if works.index.nunique() < len(works):
        print ('works: removing duplicates')
        works = works.reset_index().drop_duplicates(['id']).set_index('id')

    # if reading from csv, uncomment following code:
    # # convert string to dicts/lists
    # for col in ['authorships', 'concepts', 'referenced_works', 'counts_by_year']:
    #     works[col] = works[col].map(ast.literal_eval)

    works['counts_by_year'] = works['counts_by_year'].map(
        lambda x: preprocess_counts_by_year(x, from_year, to_year)
    )
    
    # remove works where sum of citation in individual year
    # does not sum to total citations (global field)
    works = works[works['counts_by_year'].map(
        lambda x: np.sum(lasts(x))) == works['cited_by_count']]

    return works


def preprocess_authors(authors): 
    '''
    author related preprocessing
    '''
    # remove duplicate authors (based on index 'id')
    if authors.index.nunique() < len(authors):
        print ('authors: removing duplicates')
        authors = authors.reset_index().drop_duplicates(['id']).set_index('id')

    # if reading from csv, uncomment following code:
    # # convert string to dicts/lists
    # for col in ['concepts', 'counts_by_year']:
    #     authors[col] = authors[col].map(ast.literal_eval)

    return authors


def preprocess_venues(venues): 
    '''
    venues related preprocessing
    '''
    # remove duplicate venues (based on index 'id')
    if venues.index.nunique() < len(venues):
        print ('venues: removing duplicates')
        venues = venues.reset_index().drop_duplicates(['id']).set_index('id')

    # if reading from csv, uncomment following code:
    # # convert string to dicts/lists
    # for col in ['concepts', 'counts_by_year']:
    #     venues[col] = venues[col].map(ast.literal_eval)

    return venues


def preprocess_insts(insts): 
    '''
    institution related preprocessing
    '''
    # remove duplicate insts (based on index 'id')
    if insts.index.nunique() < len(insts):
        print ('insts: removing duplicates')
        insts = insts.reset_index().drop_duplicates(['id']).set_index('id')

    # if reading from csv, uncomment following code:
    # # convert string to dicts/lists
    # for col in ['associated_institutions', 'concepts', 'counts_by_year']:
    #     insts[col] = insts[col].map(ast.literal_eval)

    return insts


def print_metric(metric_name, metric_list):
    mean, std = np.mean(metric_list), np.std(metric_list)
    print (f"  - {metric_name}: {np.round(mean, 3)} +/- {np.round(std, 3)}")
    return


def train_doc2vec(docs, dims=16, saved_model_name='d2v.model'):
    '''
    Training Doc2Vec on custom docs and saving model file
    # ref: https://medium.com/@mishra.thedeepak/doc2vec-simple-implementation-example-df2afbbfbad5
    '''
    # tag docs
    tagged_data = [TaggedDocument(words=word_tokenize(_d.lower()), tags=[str(i)]) 
                   for i, _d in tqdm(enumerate(docs), 'Tagging')]
    
    # model init
    model = Doc2Vec(vector_size=dims,
                window=3,
                alpha=alpha, 
                min_alpha=0.00025,
                min_count=1,
                workers=-1)
    
    # building vocab
    model.build_vocab(tagged_data)
    
    # model train
    max_epochs = 10
    for epoch in tqdm(range(max_epochs), 'Training'):
        model.train(tagged_data,
                    total_examples=model.corpus_count,
                    epochs=model.epochs)
        
    model.save(saved_model_name)
    print(f"Model Saved: {saved_model_name}")
    return


def abstract_feats(df, works, dims=16, saved_model='d2v.model'):
    # TODO: needs to be tested
    # ref: https://medium.com/@mishra.thedeepak/doc2vec-simple-implementation-example-df2afbbfbad5
    # TEXTUAL Features
    train_doc2vec(works['abstract'].tolist(), dims)
    model= Doc2Vec.load(saved_model)
    
    df[[f'abs_{i}' for i in range(dims)]] = np.zeros((len(df), dims))
    for idx, i in tqdm(enumerate(works.index[13000:]), 'Inferencing'):
        df.loc[i,[f'abs_{i}' for i in range(dims)]] = model.infer_vector(
            word_tokenize(works.loc[i, 'abstract'].lower())
        )

    return df