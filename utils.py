import numpy as np
import pandas as pd
import pdb
import ast
import os

# make dir if not exist
def make_dirs(directory):
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
    years = list(range(from_publication_year, to_current_year + 1))
    citations = [0]*len(years)
    for year_i, cc_i in counts_by_year:
        if year_i >= from_publication_year:
            citations[year_i-from_publication_year] = cc_i
    return list(zip(years, citations))    

def preprocess_works(works, from_year, to_year):    
    # work related preprocessing
    for col in ['authorships', 'concepts', 'referenced_works', 'counts_by_year']:
        works[col] = works[col].map(ast.literal_eval)

    # removing discrepancy in `works`
    works[f'counts_by_year'] = works['counts_by_year'].map(
        lambda x: preprocess_counts_by_year(x, from_year, to_year)
    )
    works = works[works['counts_by_year'].map(
        lambda x: np.sum(lasts(x))) == works['cited_by_count']].reset_index(drop=True)

    return works