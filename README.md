# citation_prediction
Course Project on 'Predicting the Impact of a Scientific Paper'


## Directory Structure

```
.
├── data/                       # data directory (needs to be downloaded)
├── data-scrape.ipynb           # code for scraping data from OpenAlex
├── data-eda.ipynb              # code for exploratory data analysis
├── data-modeling.ipynb         # code for modeling `citation prediction`
├── parser.py                   # helper functions for parsing various data entities
├── scraper.py                  # helper functions for scraping data from OpenAlexAPI
├── .gitignore
├── LICENSE
└── README.md
```


## Dataset
The dataset files can be downloaded from here: [OneDrive](https://purdue0-my.sharepoint.com/:f:/g/personal/gchoudha_purdue_edu/EvnracBaGV9BjcjJvHZ6Go8BjWz7VFjKKOo7OuiSXQ4Pqw).


## Getting Started
Install `conda` distribution for managing python packages. Create an environment:
```bash
$ conda create -n myenv python=3.9
```
Then install the following dependencies:
* Install common packages: 
```bash
$ conda install -c anaconda pandas numpy scikit-learn xgboost seaborn tqdm requests pyopenssl idna
```
* Install logger: 
```bash
$ conda install -c conda-forge loguru
```