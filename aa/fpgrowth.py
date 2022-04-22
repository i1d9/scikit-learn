import pandas as pd
import numpy as np
#from apyori import apriori

from mlxtend.frequent_patterns import apriori, association_rules, fpgrowth


df = pd.read_csv("final.csv")
frequently_recurring_elemets = fpgrowth(df, min_support=0.05, use_colnames=True)

print("Frequently Recurring Items\n")
#Displays Frequently recurring elements
print(frequently_recurring_elemets)


print("")
#Creating association rules
result = association_rules(frequently_recurring_elemets, metric="lift", min_threshold=1).sort_values('confidence', ascending=False)

print(result)