# basic
import sys
import os
import json
import requests
from tqdm import tqdm

# debug
import pdb
from loguru import logger

from parser import batchify


# `Works` scraper for OpenAlex API
def oa_work_scraper(
    data_url,
    parser = None,
    email = 'armaan.1997@gmail.com',
    LOG_PATH='log.out.txt', 
    DATA_PATH = 'data.out.txt', 
    PER_PAGE = 25,
    INIT_PAGE = 0,
    INIT_CURSOR = '*',
    dry_run = True,
):
    
    try:
        expected_results = requests.get(data_url).json()['meta']
        print ('dry run:', expected_results)
        if expected_results['count'] == 0:
            print (requests.get(data_url).json())
    except Exception as err:
        print (f"Got {str(response.json())[:100]}... from OpenAlex")
        print (f"Unexpected {err}, {type(err)}")
        return
        
    if not dry_run :
        # instantiating the logger
        logger.remove()
        logger.add(LOG_PATH, backtrace=True, diagnose=True, catch=True, 
                   filter=lambda record: record["level"].name == "INFO")
        logger.add(DATA_PATH, rotation='500 MB', 
                   filter=lambda record: record["level"].name == "SUCCESS",
                   format="{message}")

        # creating header for URL requests
        headers = {
            "Accept": "application/json",
            "User-Agent": f"mailto:{email}"
        }
        
        # scraping loop
        with tqdm(total=expected_results['count']) as pbar:
            pbar.update(PER_PAGE*INIT_PAGE)
            page = INIT_PAGE
            cursor = INIT_CURSOR
            results_remain = (PER_PAGE * INIT_PAGE <= expected_results['count'])
            while results_remain:
                hit_url = data_url + f'&per_page={PER_PAGE}&cursor={cursor}'
                response = requests.get(hit_url, headers=headers) # scrape data, API call

                try:
                    assert (response.status_code == 200), 'oops'
                    results = response.json()['results'] # parse data
                    page += 1

                    processed_results = parser(results)
                    logger.success (processed_results) # data logging
                    logger.info ('page: {}, results:{}, parser: {}, processed results: {}, hit url: {}'
                                 .format(page, len(results), parser, len(processed_results), hit_url)) # scrape logging
                    pbar.update(len(results))

                    cursor = response.json()['meta']['next_cursor'] # parse next pointer to data
                    results_remain = (PER_PAGE * page <= expected_results['count'])
                except Exception as err:
                    print (f"Got {str(response.json())[:100]}... from OpenAlex")
                    print (f"Unexpected {err}, {type(err)}")
                    break


        print (f'Last log: \n  - page: {page}, \n  - cursor: {cursor}, \n  - hit_url: {hit_url}')
        print (f'Logs saved to file: {LOG_PATH}')
        print (f'Data saved to file: {DATA_PATH}')

    return

# `Author, Venues, Institutions` scraper for OpenAlex API
def oa_author_scraper(
    data_url,
    author_ids,
    parser,
    email,
    LOG_PATH='log.out.txt', 
    DATA_PATH = 'data.out.txt'
):

    # instantiating the logger
    logger.remove()
    logger.add(LOG_PATH, backtrace=True, diagnose=True, catch=True, 
               filter=lambda record: record["level"].name == "INFO")
    logger.add(DATA_PATH, rotation='500 MB', 
               filter=lambda record: record["level"].name == "SUCCESS",
               format="{message}")

    # creating header for URL requests
    headers = {
        "Accept": "application/json",
        "User-Agent": f"mailto:{email}"
    }
        
    # scraping loop
    expected_results = len(author_ids)
    with tqdm(total=expected_results) as pbar:
        for page, batch_authors in enumerate(batchify(author_ids, 50)):
            try:
                hit_url = data_url + '|'.join(batch_authors) + '&per_page=50'
                response = requests.get(hit_url, headers=headers) # scrape data, API call
                assert (response.status_code == 200), 'oops'
                results = parser(response.json()['results'])
                logger.success (results) # data logging
                logger.info ('page: {}, results:{}, hit url: {}'.format(page, len(results), hit_url)) # scrape logging
                pbar.update(len(results))
                # assert len(results) == len(batch_authors), 'whoops, less results' ## due to merging author ids of same author
            except Exception as err:
                print (f"Got {str(response.json())[:100]}... from OpenAlex")
                print (f"Unexpected {err}, {type(err)}")
                break


    print (f'Last log: \n  - page: {page}, \n  - hit_url: {hit_url}')
    print (f'Logs saved to file: {LOG_PATH}')
    print (f'Data saved to file: {DATA_PATH}')

    return