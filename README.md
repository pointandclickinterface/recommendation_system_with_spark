# Recommendation System with PySpark and Yelp data
This is a recommendation system using PySpark for the large yelp dataset. It uses the weighted average of two boosted trees from catboost. The first tree trains on a limited set of categorie and accounts for cases where the data might be sparce and the second tree trains on a larger set of categories and is more accurate when the data is not sparce. 

## Requirements

- Python 3.7 or above
- The requirements in requirements.txt

## Getting started

Unzip the data.zip file

After you have installed the requirements with 
```
pip3 requirements.txt
```
run 
```
python3 recommendation_example.py
```
