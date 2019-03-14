import pandas as pd
import os

#print(os.path.dirname(os.path.abspath(__file__)))
df = pd.read_csv(os.path.dirname(os.path.abspath(__file__))+"\Test 3\Test_ID_DocumentID_Word-Table 1.csv")
df = df.dropna()

formattedDf = pd.DataFrame(columns = ["FreeText"])

for i in df["DocumentID"].unique():
    sentence = " ".join(df.loc[df["DocumentID"]==i, "Word"].values)
    newRow = pd.DataFrame({"FreeText": [sentence]}, index = [i])
    formattedDf = formattedDf.append(newRow)

formattedDf.to_csv("Test3Formatted.csv")