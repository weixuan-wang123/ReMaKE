import json
import os
from tqdm import tqdm

from transformers import LlamaTokenizer, AutoTokenizer

tokenizer = LlamaTokenizer.from_pretrained("./model/llama2-7b")

def obtain_f1_and_em(a, b):
    global tokenizer

    a_words = tokenizer.encode(a, add_special_tokens=False)
    b_words = tokenizer.encode(b, add_special_tokens=False)
    if len(a_words) == 0 and len(b_words) == 0:
        return 1.0, 1
    if len(a_words) == 0 or len(b_words) == 0:
        return 0.0, 0

    em = 1 if a == b else 0
    k = len(a_words) * len(b_words)

    intersecting_words = []
    for word in a_words.copy():
        if word in b_words:
            a_words.remove(word)
            b_words.remove(word)
            intersecting_words.append(word)

    f1_score = (len(intersecting_words) * len(intersecting_words)) / float(k)
    return f1_score, em


def my_avg(a):
    return round(sum(a) * 100 / float(len(a)), 2)


def calculate_metrics(file_root):
    with open(file_root, "r", encoding="utf-8") as f:
        data = json.load(f)

    reliablilty_f1_list = []
    reliablilty_em_list = []

    generalization_f1_list = []
    generalization_em_list = []

    locality_f1_list = []
    locality_em_list = []
    specificity_f1_list = []
    specificity_em_list = []

    portablility_f1_list = []
    portablility_em_list = []

    for item in tqdm(data):
        reliablilty_f1, reliablilty_em = obtain_f1_and_em(item["post"]["reliability"]["ans"],
                                                          item["post"]["reliability"]["target"])
        reliablilty_f1_list.append(reliablilty_f1)
        reliablilty_em_list.append(reliablilty_em)

        generalization_f1, generalization_em = obtain_f1_and_em(item["post"]["generalization"]["rephrase_acc"]["ans"],
                                                                item["post"]["generalization"]["rephrase_acc"][
                                                                    "target"])
        generalization_f1_list.append(generalization_f1)
        generalization_em_list.append(generalization_em)

        locality_f1, locality_em = obtain_f1_and_em(item["post"]["specificity"]["neighborhood_acc"]["ans"],
                                                          item["pre"]["specificity"]["neighborhood_acc"]["ans"])
        locality_f1_list.append(locality_f1)
        locality_em_list.append(locality_em)


        portablility_f1, portablility_em = obtain_f1_and_em(item["post"]["portability"]["one_hop_acc"]["ans"],
                                                            item["post"]["portability"]["one_hop_acc"]["target"])
        portablility_f1_list.append(portablility_f1)
        portablility_em_list.append(portablility_em)



    print("=" * 20 + file_root + "=" * 20)
    print("F1 score")
    print("reliablilty_f1: %f" % (my_avg(reliablilty_f1_list)))
    print("generalization_f1: %f" % my_avg(generalization_f1_list))
    print("locality_f1: %f"%my_avg(locality_f1_list))
    print("portablility_f1: %f" % my_avg(portablility_f1_list))

    print("EM score")
    print("reliablilty_em: %f" % (my_avg(reliablilty_em_list)))
    print("generalization_em: %f" % my_avg(generalization_em_list))
    print("locality_em: %f"%my_avg(locality_em_list))
    print("portablility_em: %f" % my_avg(portablility_em_list))

    reli, gene, loca, port = str(my_avg(reliablilty_f1_list)) + '/' + str(my_avg(reliablilty_em_list)),str(my_avg(generalization_f1_list)) + '/' + str(my_avg(generalization_em_list)),str(my_avg(locality_f1_list)) + '/' + str(my_avg(locality_em_list)),str(my_avg(portablility_f1_list)) + '/' + str(my_avg(portablility_em_list))


    return reli, gene, loca, port


if __name__ == "__main__":

    path = "./results/llama2-7b/16shot/"
    out_f = open(
        "./csv-results/llama7-remake-16.csv",
        "w", encoding="utf-8")

    files = os.listdir(path)
    out_f.write(
        'lang' + '\t' + 'reliability' + '\t' + 'generalization' + '\t' + 'locality' + '\t' + 'portability' + '\n')
    for f in files:
        if f.endswith('json'):
            file = path + "/" + f
            reli, gene, loca, port = calculate_metrics(file)
            out_f.write(f + '\t' + reli + '\t' + gene + '\t' + loca + '\t' + port + '\n')
    out_f.close()
