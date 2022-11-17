# citation_prediction
Course Project on 'Predicting the Impact of a Scientific Paper'


## Directory Structure

```
.
├── data-scrape.ipynb           # code for scraping data from OpenAlex
├── sample-dir/         		# sample description
└── README.md
```

## Dataset
The dataset files can be downloaded from here: [OneDrive](https://purdue0-my.sharepoint.com/:f:/g/personal/gchoudha_purdue_edu/EvnracBaGV9BjcjJvHZ6Go8BjWz7VFjKKOo7OuiSXQ4Pqw?e=I5uZxb).

## Getting Started
Install `conda` distribution for managing python packages. Create an environment:
```bash
$ conda create -n myenv python=3.9
```
Then install the following dependencies:
* Install common packages: `$ conda install -c anaconda pandas numpy scikit-learn xgboost seaborn tqdm requests pyopenssl idna`,
* Install sklearn: `$ conda install -c conda-forge loguru`,