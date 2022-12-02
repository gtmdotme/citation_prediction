# # basic
# import sys
# import os
# import json
# import requests
# from tqdm import tqdm

# # debug
# import pdb
# from loguru import logger

# parser for `Concepts`
def concept_parser(authors):
    processed_authors = []
    for author in authors:
        processed_authors.append({
            'id': get_id(author['id']),
            'wikidata': author['wikidata'],
            'level': author['level'],
            'description': author['description'],
            'display_name': author['display_name'],
            'works_count': author['works_count'],
            'cited_by_count': author['cited_by_count'],
            'ancestors': [get_id(i['id']) for i in author['ancestors']],
            'related_concepts': parse_concepts(author['related_concepts'], 10),
            'counts_by_year': parse_author_counts_by_year(author['counts_by_year']),
            'created_date': author['created_date'],
        })
    return processed_authors

# parser for `Venues`
def venue_parser(authors):
    processed_authors = []
    for author in authors:
        processed_authors.append({
            'id': get_id(author['id']),
            'display_name': author['display_name'],
            'works_count': author['works_count'],
            'cited_by_count': author['cited_by_count'],
            'is_oa': author['is_oa'],
            'type': author['type'],
            'created_date': author['created_date'],
            'concepts': parse_concepts(author['x_concepts'], 10),
            'counts_by_year': parse_author_counts_by_year(author['counts_by_year']),
        })
    return processed_authors

# parser for `Institutions`
def institution_parser(authors):
    processed_authors = []
    for author in authors:
        processed_authors.append({
            'id': get_id(author['id']),
            'display_name': author['display_name'],
            'country_code': author['country_code'],
            'type': author['type'],
            'homepage_url': author['homepage_url'],
            'works_count': author['works_count'],
            'cited_by_count': author['cited_by_count'],
            'associated_institutions': parse_associated_institutions(author['associated_institutions']),
            'concepts': parse_concepts(author['x_concepts'], 10),
            'counts_by_year': parse_author_counts_by_year(author['counts_by_year']),
            'created_date': author['created_date'],
        })
    return processed_authors

# parser for `Authors`
def author_parser(authors):
    processed_authors = []
    for author in authors:
        processed_authors.append({
            'id': get_id(author['id']),
            'orcid': author['orcid'],
            'display_name': author['display_name'],
            'works_count': author['works_count'],
            'cited_by_count': author['cited_by_count'],
            'created_date': author['created_date'],
            'concepts': parse_concepts(author['x_concepts']),
            'counts_by_year': parse_author_counts_by_year(author['counts_by_year']),
        })
    return processed_authors

# parser for `Works`
def work_parser(works):
    processed_works = []
    for work in works:
        if work['id'] is None:
            continue
        if work['host_venue']['id'] is None:
            continue
        if None in [
            auth['author']['id'] 
            for auth in work['authorships']
        ]:
            continue
        if None in [
            inst['id'] 
            for auth in work['authorships'] 
                for inst in auth['institutions']
        ]:
            continue
        if None in [
            con['id'] 
            for con in work['concepts']
        ]:
            continue

        processed_works.append({
            'id': get_id(work['id']),
            'doi': work['doi'],
            'title': work['title'],
            'type': work['type'],
            'publication_date': work['publication_date'],
            'host_venue': get_id(work['host_venue']['id']),
            'open_access_is_oa': work['open_access']['is_oa'],
            'open_access_oa_status': work['open_access']['oa_status'],
            'authorships': parse_authorships(work['authorships']),
            'page_count': parse_biblio(work['biblio']),
            'cited_by_count': work['cited_by_count'],
            'concepts': parse_concepts(work['concepts']),
            'referenced_works': [get_id(ref) for ref in work['referenced_works']],
            'abstract': parse_abstract(work['abstract_inverted_index']),
            'counts_by_year': parse_work_counts_by_year(work['counts_by_year']),
        })
    return processed_works

# parsing utility functions
def get_id(url):
    return url.split('https://openalex.org/')[1]

def get_url(id_):
    return f'https://openalex.org/{id_}'

def get_cited_by_url(work_id_):
    return f'https://api.openalex.org/works?filter=cites:{work_id_}'


# `Works` related parsing utils
def int_author_position(pos):
    if pos.lower() == 'first':
        return 1
    elif pos.lower() == 'middle':
        return 2
    elif pos.lower() == 'last':
        return 3
    
def parse_authorships(authorships):
    return [
        [
            get_id(auth['author']['id']), 
            [get_id(inst['id']) if inst['id'] else None for inst in auth['institutions']]
        ]
        for auth in sorted(authorships, key=lambda x: int_author_position(x['author_position']))
    ]

def parse_concepts(concepts, top_k=5):
    return [
        [
            get_id(con['id']), 
            con['score']
        ]
        for con in sorted(concepts, key=lambda x: x['score'], reverse=True)
    ][:top_k]
    
def parse_abstract(abstract_inverted_index):
    max_len = max([j for _, i in abstract_inverted_index.items() for j in i]) + 1
    abstract = ['' for _ in range(max_len)]
    for key, value in abstract_inverted_index.items():
        for idx in value:
            abstract[idx] = key
    return ' '.join(abstract)

def parse_work_counts_by_year(counts_by_year):
    return [
        [
            count['year'],
            count['cited_by_count']
        ]
        for count in counts_by_year
    ]

def parse_biblio(biblio):
    if biblio['first_page']:
        if biblio['last_page']:
            return int(biblio['last_page']) - int(biblio['first_page']) + 1,
    return -1

# `Author` related parsing utils
def parse_author_counts_by_year(counts_by_year):
    return [
        [
            count['year'],
            count['works_count'],
            count['cited_by_count'],
        ]
        for count in counts_by_year
    ]


def batchify(arr, batch_size):
    return [arr[i:i + batch_size] for i in range(0, len(arr), batch_size)]


# `Institution` related parsing utils
def parse_associated_institutions(associated_institutions):
    return [
        [
            get_id(ref['id']), 
            ref['relationship']
        ] 
        for ref in associated_institutions
    ]