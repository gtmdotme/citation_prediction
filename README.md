# citation_prediction
Course Project on 'Predicting the Impact of a Scientific Paper'


## Directory Structure

```
.
├── data-scrape.ipynb           # code for scraping data from OpenAlex
├── sample-dir/         		# sample description
└── README.md
```

## Getting Started
Install `conda` distribution for managing python packages. Create an environment:
```bash
$ conda create -n myenv python=3.9
```
Then install the following dependencies:
* Install common packages: `$ conda install -c anaconda pandas numpy seaborn tqdm requests pyopenssl idna`,
* Install sklearn: `$ conda install -c conda-forge loguru`,