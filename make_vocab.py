f = open("data/next_sentence",encoding="utf-8",mode="r")
lines = f.readlines()
vocab_set = set()

def clean_word(word):
    word = word.lower()
    word = word.replace(".", "")
    word = word.replace("\n","")
    word = word.replace('"', '')
    word = word.replace("#", "")
    word = word.replace("&", "")
    word = word.replace("$", "")
    word = word.replace("(", "")
    word = word.replace(")", "")
    if word != "" and word != " " and word != None:
        return word
    else:
        return "<UNK>"


for line in lines:
    word_list = line.split(" ")
    for word in word_list:
        if "\t" in word:
            ws = word.split("\t")
            for w in ws:
                vocab_set.add(clean_word(w))
        else:
            vocab_set.add(clean_word(word))

f.close()

f = open("data/vocab",encoding="utf-8",mode="w")

f.write("<PAD>")
f.write("\n")
f.write("<SEP>")
f.write("\n")
f.write("</S>")
f.write("\n")
f.write("<S>")
f.write("\n")
f.write("<UNK>")
f.write("\n")
vocab_list = sorted(list(vocab_set))
for word in vocab_list:
    f.write(word)
    f.write("\n")

f.close()