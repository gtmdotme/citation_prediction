# basic
import sys
import os
import json
import ast
import time
import requests
from tqdm import tqdm
from collections import Counter, defaultdict, namedtuple

# debug
import pdb
from loguru import logger

import numpy as np
import pandas as pd

# sklearn
from sklearn import metrics
from sklearn.mixture import GaussianMixture

# custom
from utils import *

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


def preprocess_data(works, from_year, to_year, authors, venues, insts):
    '''
    preprocessing data
    '''
    
    # if reading from csv, uncomment following code:
    # # convert string to dicts/lists
    # WORKS: for col in ['authorships', 'concepts', 'referenced_works', 'counts_by_year']:
    # AUTHORS: for col in ['concepts', 'counts_by_year']:
    # VENUES: for col in ['concepts', 'counts_by_year']:
    # INSTS: for col in ['associated_institutions', 'concepts', 'counts_by_year']:
    #     authors[col] = authors[col].map(ast.literal_eval)
    
    # remove duplicate works (based on index 'id')
    if works.index.nunique() < len(works):
        print ('works: removing duplicates')
        works = works.reset_index().drop_duplicates(['id']).set_index('id')
        
    # remove duplicate authors (based on index 'id')
    if authors.index.nunique() < len(authors):
        print ('authors: removing duplicates')
        authors = authors.reset_index().drop_duplicates(['id']).set_index('id')
        
    # remove duplicate venues (based on index 'id')
    if venues.index.nunique() < len(venues):
        print ('venues: removing duplicates')
        venues = venues.reset_index().drop_duplicates(['id']).set_index('id')
        
    if insts.index.nunique() < len(insts):
        print ('insts: removing duplicates')
        insts = insts.reset_index().drop_duplicates(['id']).set_index('id')

    # filter `counts_by_year` in given time period
    works['counts_by_year'] = works['counts_by_year'].map(
        lambda x: preprocess_counts_by_year(x, from_year, to_year)
    )
    
    # remove works where sum of citation in individual year
    # does not sum to total citations (global field)
    works = works[works['counts_by_year'].map(
        lambda x: np.sum(lasts(x))) == works['cited_by_count']]
    
    # remove works where author not in data
    tbr = works['authorships'].map(
        lambda x: len([a for a in firsts(x) if a not in authors.index]) > 0)
    works = works[~tbr]
    
    # remove works where insts not in data
    tbr = works['authorships'].map(
        lambda x: len([i for i in flatten(lasts(x)) if i not in insts.index]) > 0)
    works = works[~tbr]
    
    # remove works where venue not in data
    tbr = works['host_venue'].map(lambda x: x not in venues.index)
    works = works[~tbr]
    
    # remove works where author's inst is unknown
    tbr = works['authorships'].map(lambda x: len([len(i) for i in lasts(x) if len(i) == 0]) > 0)
    works = works[~tbr]

    return works, authors, venues, insts

## feature engineering related utils
def bin_citations(x, n_clusters, seed):
    x = x.to_numpy().reshape(-1, 1)
    gmm = GaussianMixture(n_components=n_clusters, max_iter=1000, random_state=seed)
    labels = gmm.fit(x).predict(x)
    return labels

def count_prior_citations(counts_by_year, YEAR):
    '''
    `counts_by_year` is a list of tuple (year, #works, #citations)
    '''
    counts = [i[2] for i in counts_by_year if i[0]<YEAR]
    if len(counts):
        return np.sum(counts)
    return 0

def count_prior_works(counts_by_year, YEAR):
    '''
    `counts_by_year` is a list of tuple (year, #works, #citations)
    '''
    counts = [i[1] for i in counts_by_year if i[0]<YEAR]
    if len(counts):
        return np.sum(counts)
    return 0

def get_features(works, authors, venues, insts, YEAR, N_CLASSES, seed):
    '''
    Feature generate given the data for both regression and classification
    '''
    df = pd.DataFrame()

    # `WORK`: Paper meta-data features
    df['no_of_authors'] = works['authorships'].map(lambda x: len(x))
    df['no_of_referenced_works'] = works['referenced_works'].map(len)
    df['open_access_is_oa'] = works['open_access_is_oa']
    df['publication_month'] = works['publication_date'].map(lambda x: int(x.split('-')[1]))

    
    # `AUTHOR`: Author-specific Features
    authors['prior_citations'] = authors['counts_by_year'].map(lambda x: count_prior_citations(x, YEAR))
    a_prior_citations_dict = defaultdict(lambda: 0, authors['prior_citations'].to_dict())
    threshold = authors['prior_citations'].mean()
    df['author_prominency'] = works['authorships'].map(
        lambda x: 1 if max([a_prior_citations_dict[i[0]] for i in x]) >= threshold else 0)
    df['authors_mean_citations'] = works['authorships'].map(lambda x: np.mean([a_prior_citations_dict[i[0]] for i in x]))
    
    authors['prior_works'] = authors['counts_by_year'].map(lambda x: count_prior_works(x, YEAR))
    a_prior_works_dict = defaultdict(lambda: 0, authors['prior_works'].to_dict())
    df['authors_mean_works'] = works['authorships'].map(lambda x: np.mean([a_prior_works_dict[i[0]] for i in x]))

    
    # `VENUE`: Journal and Publisher relevant features
    venues['prior_citations'] = venues['counts_by_year'].map(lambda x: count_prior_citations(x, YEAR))
    v_prior_citations_dict = defaultdict(lambda: 0, venues['prior_citations'].to_dict())
    df['venue_citations'] = works['host_venue'].map(lambda x: v_prior_citations_dict[x])
    
    venues['prior_works'] = venues['counts_by_year'].map(lambda x: count_prior_works(x, YEAR))
    v_prior_works_dict = defaultdict(lambda: 0, venues['prior_works'].to_dict())
    df['venue_works'] = works['host_venue'].map(lambda x: v_prior_works_dict[x])
    
    df['venue_significance'] = works['host_venue'].map(
        lambda x: v_prior_citations_dict[x]/v_prior_works_dict[x] if v_prior_works_dict[x] else 0)
    

    # `INSTITUTION`: Insti-specific Features
    insts['prior_citations'] = insts['counts_by_year'].map(lambda x: count_prior_citations(x, YEAR))
    i_prior_citations_dict = defaultdict(lambda: 0, insts['prior_citations'].to_dict())
    df['insts_mean_citations'] = works['authorships'].map(
        lambda x: np.mean([i_prior_citations_dict[i[1][0]] for i in x if len(i[1])>0]))
    
    insts['prior_works'] = insts['counts_by_year'].map(lambda x: count_prior_works(x, YEAR))
    i_prior_works_dict = defaultdict(lambda: 0, insts['works_count'].to_dict())
    df['insts_mean_works'] = works['authorships'].map(lambda x: np.mean([i_prior_works_dict[i[1][0]] for i in x if len(i[1])>0]))    
    
    # TODO: needs to be corrected
    df.fillna(0, inplace=True)
    
    # target variable - regression
    ## cumulative citation count
    df['y_reg'] = works['counts_by_year'].map(lambda x: np.sum([v for k, v in x if k in [YEAR, YEAR+1, YEAR+2]]))
    # normalizing target variable
    elapsed_months = 2*12 + 12-df['publication_month']
    df['y_reg_norm'] = df['y_reg']/elapsed_months
    
    # target variable - classification
    df['y_clf'] = bin_citations(df['y_reg'], n_clusters=N_CLASSES, seed=seed)
    
    return df


def print_metric(metric_name, metric_list):
    mean, std = np.mean(metric_list), np.std(metric_list)
    print (f"  - {metric_name}: {np.round(mean, 3)} +/- {np.round(std, 3)}")
    return


## doc2vec related utils
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
                window=4,
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


def infer_doc2vec(works, col, indices, dims=16, saved_model='d2v.model'):
    '''
    Infering from Doc2Vec on custom indices of data
    # ref: https://medium.com/@mishra.thedeepak/doc2vec-simple-implementation-example-df2afbbfbad5
    '''
    df = pd.DataFrame(index=indices)
    model= Doc2Vec.load(saved_model)

    df[[f'{col}_{i}' for i in range(dims)]] = np.zeros((len(df), dims))
    for i in tqdm(works.loc[indices].index, 'Inferencing'):
        df.loc[i,[f'{col}_{i}' for i in range(dims)]] = model.infer_vector(
            word_tokenize(works.loc[i, col].lower())
        )

    return df