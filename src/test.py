import csv

with open('data/wikibooks/wikibooks-en.csv', newline='') as csvfile:
    reader = csv.reader(csvfile)

    for row in reader:
        print(', '.join(row))