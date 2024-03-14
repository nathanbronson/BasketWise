<p align="center"><img src="https://github.com/nathanbronson/BasketWise/blob/main/logo.jpg?raw=true" alt="logo" width="200"/></p>

_____
# BasketWise
a collection of NCAA Basketball data science experiments

## About
BasketWise is a collection of experiments on NCAA Basketball data. It includes regressions, matching algorithms, and neural networks meant to predict the outcomes of basketball games. Applications included in this repository are those that predict specific games, create league rankings, and forecast tournament outcomes. This endeavor was exploratory and informational, so no user-facing applications are included in this repository. In sentiment, this repository is a set of utility functions for sports data analysis. This repository also includes a lot of junk.

Data files are stored separately and are not included in this repository. Active development of this repository and its antecedents took place between 2017 and 2023. This repository is no longer actively maintained.

## Structure
BasketWise contains many experiments. The main experiments can be found at the following places in the repository.
```
BasketWise
└───Rankings/
│   │   ncaam_rankings.py           code to create rankings
│   │   
│
└───2021 Experiments/               
│   │   marchlogisticfull.py        code to forecast 2021 NCAA Tournament
│   │   _final_output.txt           final forecast of 2021 NCAA Tournament
│   │   
│
└───2022 Experiments/               
│   └───Other Models/               code for various neural networks and KNN classifiers
│   │   │
│   │
│   │   marchlogisticfull_MAIN.py   more efficient tournament forecasting script
│   │
│
└───2023 Experiments/               
│   └───Weekly Rankings/            rankings calculated each week of the 2022-23 season
│   │   │       
│   │   
│   │   bracketformer.py            transformer to predict game outcomes from past play-by-play data
│   │   optimized_winformer.py      transformer to estimate play-by-play win probability
│   │   win_percentage.py           real time win probability graph based on winformer estimates
│   │   playerformer.py             transformer to predict game outcomes based on player lineup
│   │   rankings_analysis.py        analyze timewise coherence of rankings metric
│   │   linearity.py                analyze correlation between rankings metric and point differentials
│   │   
│
```

## License
See `LICENSE`.
