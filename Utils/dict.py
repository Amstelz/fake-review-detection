import re
from nltk.corpus import stopwords

def get_most_frequent(text_string:str, remove_stop_word:bool):        
    frequency = {}

    words = re.findall(r'\b[A-Za-z][a-z]{2,9}\b', text_string)
    stop = stopwords.words('english')
    
    for word in words:
        if remove_stop_word:
            if word not in stop:
                count = frequency.get(word,0)
                frequency[word] = count + 1
        else:
            count = frequency.get(word,0)
            frequency[word] = count + 1

    most_frequent = dict(sorted(frequency.items(), key=lambda elem: elem[1], reverse=True))

    return most_frequent

def print_most_frequent(most_frequent:dict):
    top_count = 0
            
    for idx, (words, frequency) in enumerate(most_frequent.items()):
        if idx == 0:
            top_count = frequency
        print(f'{words}: {frequency}: {round(top_count/frequency, 2)}')

def print_word_and_delta(words:list, deltas:list):
    print('word: delta_KL')
    for w,d in zip(words, deltas):
        print(f'{w}: {d}')

def print_comporison(words:list, common_dict_fake:dict, common_dict_genuine:dict):
    sum_fake = sum(list(common_dict_fake.values()))
    sum_genuine = sum(list(common_dict_genuine.values()))
    for word in words:
        y_prob = common_dict_fake[word]/sum_fake
        n_prob = common_dict_genuine[word]/sum_genuine
        status = 'fake' if y_prob > n_prob else 'genuine'
        print(f'{word}:fake -> {y_prob}, genuine -> {n_prob} : {status}')