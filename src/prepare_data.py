"""
Loads the pre-downloaded data ie. gloss and correspond text
into json file

Authors:
    * Megan Dare, 2022
    * Sangeet Sagar, 2022
"""

import json

data = {
    'TRAIN_GLOSS' : 'data/phoenix2014T.train.gloss',
    'TRAIN_DE'    : 'data/phoenix2014T.train.de',
    'DEV_GLOSS'   : 'data/phoenix2014T.dev.gloss',
    'DEV_DE'      : 'data/phoenix2014T.dev.de',
    'TEST_GLOSS'  : 'data/phoenix2014T.test.gloss',
    'TEST_DE'     : 'data/phoenix2014T.test.de',
    'VOCAB_GLOSS' : 'data/phoenix2014T.vocab.gloss',
    'VOCAB_DE'    : 'data/phoenix2014T.vocab.de'
}

def extract_content(path):
    gloss = open(path[0], "r")
    text = open(path[1], "r", encoding='utf-8-sig')
    content_gloss = gloss.read().splitlines()
    content_text = text.read().splitlines()
    content = []
    for item in list(zip(content_gloss, content_text)):
        content.append(
            {
            "gloss": item[0],
            "text": item[1]
        }
        )
    return content
    
def prepare_data(data, save_json_train, save_json_valid, save_json_test):
    train_path = [data['TRAIN_GLOSS'], data['TRAIN_DE']]
    valid_path = [data['DEV_GLOSS'], data['DEV_DE']]
    test_path = [data['TEST_GLOSS'], data['TEST_DE']]
    train = extract_content(train_path)
    valid = extract_content(valid_path)
    test = extract_content(test_path)
    
    with open(save_json_train, 'w') as f:
        json.dump(train, f, indent=4, ensure_ascii=False)
    with open(save_json_valid, 'w') as f:
        json.dump(valid, f, indent=4, ensure_ascii=False)
    with open(save_json_test, 'w') as f:
        json.dump(test, f, indent=4, ensure_ascii=False)
    
    print("****** Stats for nerds ******")
    print("Total train sents: ", len(train))
    print("Total valid sents: ", len(valid))
    print("Total test sents: ", len(test))
    print("\nExample of dataset: ")
    print("Gloss: ", train[0]["gloss"])
    print("Test: ", train[0]["text"])
    

if __name__ == "__main__":
    prepare_data(data, 'train.json', 'valid.json', 'test.json')
    
    