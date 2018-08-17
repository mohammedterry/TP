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

training_data = (
    ({0},{1}),
    ({1},{2}),   
)
v_len = 10
ws = [[0 for _ in range(v_len)] for _ in range(v_len)]
for _ in range(10):
    for vector_in,vector_out in training_data:
        ws = tp_learn(vector_in, vector_out, ws, v_len)
v_in = {0,1}
v_out = tp_predict(v_in,ws,v_len)
display(v_in,v_out,ws,v_len)