def tp_learn(vec_in, vec_out, weights, vec_len, learning_rate = .1, forget_rate = .01):
    for v_i in vec_in:
        for v_o,col in enumerate(weights[v_i]):
            if v_o in vec_out:
                weights[v_i][v_o] =  min(col + learning_rate, 1)
            else:
                weights[v_i][v_o] = max(col - forget_rate, 0)
            # display(vec_in, vec_out, weights, vec_len)
    return weights

def tp_predict(vec_in,weights,vec_len,activation = .5):
    vec_out = set()
    for v_i in vec_in:
        for v_o,w in enumerate(weights[v_i]):
            if w > activation:
                vec_out.add(v_o)
    return vec_out

def display(vector_in,vector_out,weights,vector_len):
    print(vector_in,'->',vector_out)
    print('   v_out',[1 if i in vector_out else 0 for i in range(vector_len)])
    print('v_in')
    [print([1],'\t',w) if v in vector_in else print([0],'\t',w) for v,w in zip(range(vector_len),weights)]
    print()

def compare_sets(set_v1, set_v2): #1.0 = 0% identical
    return len(set_v1 ^ set_v2) / (len(set_v1) + len(set_v2) + 1e-8)

def synonyms(vec_w, vector_dictionary, threshold = 1.):    
    costs = {}
    for word,vec_dic in vector_dictionary.items():
        cost = compare_sets(vec_w,vec_dic)
        if cost < threshold:
            if cost in costs:
                costs[cost].append(word)
            else:
                costs[cost] = [word]
    return [words for _,words in sorted(zip(costs,costs.values()))][0]



vocabulary = {
    "a":{0},
    "man":{1},
    ".":{2},
}
training_data = (
    ("a","man"),
    ("man","."),   
)
v_len = 10
ws = [[0 for _ in range(v_len)] for _ in range(v_len)]
for _ in range(10):
    for word1,word2 in training_data:
        vector_in = vocabulary[word1]
        vector_out = vocabulary[word2]
        ws = tp_learn(vector_in, vector_out, ws, v_len)
word_in = "man"
v_in = vocabulary[word_in]
v_out = tp_predict(v_in,ws,v_len)
words_out = synonyms(v_out,vocabulary)
display(v_in,v_out,ws,v_len)
print(word_in,'-->',words_out)
