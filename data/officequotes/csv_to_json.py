import csv
import json
import pdb
 
# Function to convert a CSV to JSON
# Takes the file paths as arguments
def make_json(csvFilePath, jsonFilePath):
     
    # create a dictionary
    data = []
     
    # Open a csv reader called DictReader
    with open(csvFilePath, encoding='utf-8') as csvf:
        csvReader = csv.DictReader(csvf, delimiter=',')

        with open(jsonFilePath, 'w', encoding='utf-8') as jsonf:
            # Convert each row into a dictionary
            # and add it to data
            for rows in csvReader:
                # rows["ask"] = rows["parent"]
                # rows["ask_id"] = rows["parent_id"]

                rows["title"] = rows["parent"]
                rows["body"] = rows["reply"]

                del rows["parent"]
                del rows["parent_id"]
                del rows["reply"]
                # pdb.set_trace()
                
                jsonf.write(json.dumps(rows))
                jsonf.write('\n')

    # Open a json writer, and use the json.dumps()
    # function to dump data

         
# Driver Code
 
# Decide the two file paths according to your
# computer system
csvFilePath = r'parent_reply.csv'
jsonFilePath = r'data.json'
 
# Call the make_json function
make_json(csvFilePath, jsonFilePath)