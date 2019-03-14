import pandas as pd
import os
from lxml import etree as et

test1Location = os.path.dirname(os.path.abspath(__file__))+"\\Test 1\\metamapped\\"
test2Location = os.path.dirname(os.path.abspath(__file__))+"\\Test 2\\metamapped\\"

formattedDf = pd.DataFrame (columns = ["FreeText"]) 

ind = 0

for path in [test1Location, test2Location]:
    for subdir, dirs, fileNames in os.walk(path):
        for file in fileNames:
            parsedXML = et.parse(path + file)
        
            root = parsedXML.getroot()
	    
            freeText = root.findall(".//UttText")
        
            fullString = ""
            for i in freeText:
                fullString += " " + i.text.rstrip()
        
            newRow = pd.DataFrame ({"FreeText": [fullString]}, index = [ind])
            formattedDf = formattedDf.append(newRow)
        
            ind +=1
        
formattedDf.to_csv("Test1+2Formatted.csv")