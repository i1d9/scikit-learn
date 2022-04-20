from re import S
import pandas as pd
import numpy as np
#from apyori import apriori

from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules


df = pd.read_csv("final.csv")

print(df.shape[0])
frequent_itemsets_plus = apriori(df, min_support=0.03, use_colnames=True).sort_values("support", ascending= False).reset_index(drop=True)
frequent_itemsets_plus['length'] = frequent_itemsets_plus['itemsets'].apply(lambda x: len(x))

print(frequent_itemsets_plus)
#Frequently bought items are 24

associations = association_rules(frequent_itemsets_plus, metric='lift', min_threshold=1).sort_values('lift', ascending=False).reset_index(drop=True)
print(associations)

#Out of the total transactions, coffie and medialuna were bought together 47.5% which is 4525
#Coffee and Medialuna were both together 47.5% times amoung the provided dataset
#0.475016 * 9526 = 4525

print("\nSorted according to Confidence Levels\n")
confidence = association_rules(frequent_itemsets_plus, metric='lift', min_threshold=1).sort_values('confidence', ascending=False).reset_index(drop=True)
print(confidence)
