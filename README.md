# ML_Bias

SDIC Individual Research Project 


## Project Title

Algorithmic Bias in Machine Learning for Healthcare


## Environment

```
Python 3.8.8
Numpy 1.21.5
Pandas 1.4.1
scikit-learn 1.0.2
XGBoost 1.6.1
Matplotlib 3.3.4
```


## Dataset

The data used in this paper is [National (Nationwide) Inpatient Sample (NIS)](https://www.hcup-us.ahrq.gov/nisoverview.jsp) from Healthcare Cost and Utilization Project  (HCUP). According to Data Use Restrictions, I will not redistribute HCUP data by posting on any website or publishing in any other publicly accessible online repository. Interested parties may contact HCUP for more information on accessing HCUP data at this 
[link](https://www.hcup-us.ahrq.gov/).


## Project build instructions

```
├── bias
│    ├── __init__.py   
│    ├── algorithms.py                        
│    ├── main.py                
│    └── utils.py       
├── example                    
│    └── example.ipynb
├── HCUP    // Cannot be made publicly available
│    ├── NIS_2012_Core.csv
│    └── NIS_2014_Core.csv                            
├── output
│    ├── income60.png
│    ├── income75.png
│    ├── income90.png
│    ├── race60.png
│    ├── race75.png
│    └── race90.png
├── __init__.py
└── README.md
```