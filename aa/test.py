import pandas as pd
import xlsxwriter

#Load the dataset from th csv
df = pd.read_csv('cafe.csv')

cafe_items = pd.DataFrame(df,columns=['Item'])
transaction_items = pd.DataFrame(df,columns=['Transaction']).drop_duplicates()
unique_items = cafe_items.drop_duplicates()

rows = df.shape[0]
columns = df.shape[1]

final = {}
for row in range(rows):
    date = df.loc[row, 'Date']
    time = df.loc[row, 'Time']
    transaction = df.loc[row, 'Transaction']
    item = df.loc[row, 'Item']

    info = {}
    if transaction in final:
        info = final[transaction]
        info[item] = True
        final[transaction] = info
    else:
        info = {'Date': date, 'Time': time}
        info[item] = True
        final[transaction] = info
        

for key in final.keys():
    #print(key)
    print(final[key])


output_columns = {'Transaction': final.keys()}

for unique_item in unique_items['Item']:
    target_values = []
    for key in final.keys():
        if unique_item in final[key]:
            target_values.append(1)
        else:
            target_values.append(0)
    output_columns[unique_item] = target_values


output = pd.DataFrame(output_columns)
output.to_csv("formatted_dataset.csv")
"""

# Create a Pandas Excel writer using XlsxWriter as the engine.

writer = pd.ExcelWriter('formatted.xlsx', engine='xlsxwriter')
# Convert the dataframe to an XlsxWriter Excel object.

output.to_excel(writer, sheet_name='Sheet1', index=False)
# Close the Pandas Excel writer and output the Excel file.
writer.save()
"""
