# -*- coding: utf-8 -*-

"""
Reference:https://blog.csdn.net/qianwenhong/article/details/41512671
"""
from asyncio import IncompleteReadError
from statistics import mode
from hmm import Model
import hmm

import tools_ascc


state_list, symbol_list = tools_ascc.get_activity_for_state_list()
# print(state_list)
# print(len(state_list))

sequences = []
for i in range(len(state_list) -15):
    print(state_list[i])
    print("==")
    seq = (state_list[i], symbol_list[i])
    sequences.append(seq)


print('len sequence:', len(sequences))
print(sequences[1])


# testseq = state_list[len(state_list) -10]
# # testseq = ('Master_Bathroom', 'Kitchen_Activity', 'Read', 'Guest_Bathroom', 'Kitchen_Activity', 'Morning_Meds', 'Master_Bedroom_Activity', 'Master_Bathroom', 'Read', 'Kitchen_Activity', 'Guest_Bathroom', 'Master_Bedroom_Activity', 'Read', 'Desk_Activity', 'Master_Bathroom', 'Kitchen_Activity', 'Read', 'Sleep', 'Master_Bathroom', 'Kitchen_Activity', 'Master_Bedroom_Activity', 'Master_Bathroom', 'Kitchen_Activity', 'Leave_Home', 'Desk_Activity', 'Kitchen_Activity', 'Guest_Bathroom', 'Watch_TV', 'Master_Bedroom_Activity', 'Master_Bathroom', 'Master_Bathroom', 'Master_Bathroom', 'Watch_TV', 'Guest_Bathroom', 'Kitchen_Activity', 'Watch_TV', 'Guest_Bathroom', 'Kitchen_Activity', 'Watch_TV', 'Master_Bathroom', 'Guest_Bathroom', 'Kitchen_Activity', 'Master_Bathroom')
# print('symbolist:', testseq)

# exit(0)

model = hmm.train(sequences, delta=0.001, smoothing=0)


# correct = 0
# incorrect = 0
# print('test==========================')
# # todo new testseq
# testseq = symbol_list[len(state_list) -10]
# # testseq = ('Master_Bathroom', 'Kitchen_Activity', 'Read', 'Guest_Bathroom', 'Kitchen_Activity', 'Morning_Meds', 'Master_Bedroom_Activity', 'Master_Bathroom', 'Read', 'Kitchen_Activity', 'Guest_Bathroom', 'Master_Bedroom_Activity', 'Read', 'Desk_Activity', 'Master_Bathroom', 'Kitchen_Activity', 'Read', 'Sleep', 'Master_Bathroom', 'Kitchen_Activity', 'Master_Bedroom_Activity', 'Master_Bathroom', 'Kitchen_Activity', 'Leave_Home', 'Desk_Activity', 'Kitchen_Activity', 'Guest_Bathroom', 'Watch_TV', 'Master_Bedroom_Activity', 'Master_Bathroom', 'Master_Bathroom', 'Master_Bathroom', 'Watch_TV', 'Guest_Bathroom', 'Kitchen_Activity', 'Watch_TV', 'Guest_Bathroom', 'Kitchen_Activity', 'Watch_TV', 'Master_Bathroom', 'Guest_Bathroom', 'Kitchen_Activity', 'Master_Bathroom')
# print('symbolist:', testseq)
# print(model.evaluate(testseq))
# decode_seq = model.decode(testseq)
# print("decode list:", decode_seq)

# target_trans = model._trans_prob
# print('target_trans:', target_trans)


# activity_pro = {}
# for a in model._start_prob.keys:
#     activity_pro = {a: 0}



# # todo check top 1, top2 accuracy of HMM model
# correct = 0
# incorrect = 0
# for i in range(len(testseq)):
#     cur = testseq[i]

#     max = 0
#     max_activity = ''
#     for p in target_trans[cur]:
#         if p[2] > max:
#             max = p[2]
#             max_activity = p[1]

#     if max_activity == testseq[i+1]:
#         correct = correct + 1
#     else:
#         incorrect = incorrect + 1

# print('acc:', correct*1.0/(correct+incorrect))


# exit(0)

# for i in range(len(testseq)):
#     if decode_seq[i] == testseq[i]:
#         correct = correct + 1
#     else:
#         incorrect = incorrect + 1

# print('acc:', correct*1.0/(correct+incorrect))

correct = 0
incorrect = 0

testsequences = []
for i in range(len(symbol_list) -15, len(symbol_list)):
    print("========================================================================")
    print('Symbol_list:')
    print(symbol_list[i])
    print('State list')
    print(state_list[i])

    test_symbol_list = symbol_list[i]
    testseq = state_list[i]

    evalu = model.evaluate(test_symbol_list)
    print('evaluate:', evalu)

    decode_seq = model.decode(test_symbol_list)
    print('decode:')
    print(decode_seq)

    for i in range(len(testseq)):
        if decode_seq[i] == testseq[i]:
            correct = correct + 1
        else:
            incorrect = incorrect + 1
            print('not equal:', testseq[i], ' ', decode_seq[i])

print('acc:', correct*1.0/(correct+incorrect))


# todo: output top1 and top2  decode results

exit(0)

states = ('H', 'C')
symbols = ('S', 'M', 'L')

start_prob = {
    'H' : 0.6,
    'C' : 0.4
}

trans_prob = {
    'H': { 'H' : 0.7, 'C' : 0.3 },
    'C': { 'H' : 0.4, 'C' : 0.6 }
}

emit_prob = {
    'H': { 'S' : 0.1, 'M' : 0.4, 'L' : 0.5 },
    'C': { 'S' : 0.7, 'M' : 0.2, 'L' : 0.1 }
}

sequence = ['S', 'M', 'S', 'L']
model = Model(states, symbols, start_prob, trans_prob, emit_prob)

print(model.evaluate(sequence))
print(model.decode(sequence))

