import torch

with open("1342-0.txt", encoding='utf8') as f:
    text = f.read()
print(type(text), len(text))

lines= text.split('\n')
random_pick = 200
line = lines[random_pick]
letter_tensor = torch.zeros(len(line), 128)
print(line,"\n", letter_tensor.shape)

for i, letter in enumerate(line.lower().strip()):
    letter_index = ord(letter) if ord(letter) < 128 else 0
    letter_tensor[i][letter_index] = 1
def str_to_clean_word_list(input_str):
    punctuation = '.,;:"!?_-”“'
    word_list = input_str.lower().replace('\n',' ').split()
    word_list = [word.strip(punctuation) for word in word_list]
    return word_list


words_in_line = str_to_clean_word_list(line)
print(line, "\n", words_in_line)


text_words = str_to_clean_word_list((text))
words_list = sorted(set(text_words))
word2index_dict = {word:i for (i, word) in enumerate(words_list)}
word_tensor = torch.zeros(len(text_words), len(word2index_dict))
for i, word in enumerate(text_words):
    word_index = word2index_dict[word]
    word_tensor[i][word_index]=1
    print('{:2} {:4} {}'.format(i, word_index, word))


print(len(word2index_dict))
print(line)
line_words = str_to_clean_word_list(line)
line_tensor = torch.zeros(len(line_words), len(word2index_dict))
for i, word in enumerate(line_words):
    word_index = word2index_dict[word]
    line_tensor[i][word_index] = 1
    print('{:2} {:4} {}'.format(i, word_index, word))
print(line_tensor.shape)