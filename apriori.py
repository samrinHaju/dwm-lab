#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 23 08:21:40 2019

@author: aiktc17DCO68
"""
#importing the labraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importing the dataset
dataset=pd.read_csv('Market_Basket_Optimisation.csv',header = None)
transactions = []
for i in range(0,7501):
    transactions.append([str(dataset.values[i,j])for j in range(0,20)])
 #training Apriori on the dataset
from apyori import apriori
rules = apriori(transactions,min_support=0.003,min_confidance)
 

