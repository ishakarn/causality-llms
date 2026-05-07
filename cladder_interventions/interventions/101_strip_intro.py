# script to remove the Intro from the background field

import json

print("----- Running intervener_strip_background.py -----")
print()

INPUT_FILE = 'data/queries_noncommonsense.json'
OUTPUT_FILE = 'data/101_drop_intro_noncommonsense.json'

with open(INPUT_FILE, 'r') as f:
    data = json.load(f)
print("Number of samples in the dataset: " + str(len(data)))

suffix = 'Output one word to answer the question with just \"Yes\" or \"No\".'
for query in data:
    query['original_text'] = query['text']
    bg = query['background']
    split_index = bg.index(':') + 2
    query['text'] = query['background'][split_index:] + '. ' + query['given_info'] + ' ' + query['question'] + ' ' + suffix

with open(OUTPUT_FILE, "w") as outfile:
    json.dump(data, outfile, indent=4)
    print("Data saved")

print()
print("done")