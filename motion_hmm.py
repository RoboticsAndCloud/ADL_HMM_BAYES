# -*- coding: utf-8 -*-

"""
Reference:https://blog.csdn.net/qianwenhong/article/details/41512671
"""
from asyncio import IncompleteReadError
from operator import ne
from statistics import mode
from hmm import Model
import hmm

import tools_ascc


state_list, symbol_list = tools_ascc.get_activity_for_state_list()
# print(state_list)
# print(len(state_list))
# exit(0)
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


# test_seq = ['Kitchen_Activity_M_0', 'Leave_Home_M_0', 'Guest_Bathroom_M_0', 'Read_M_0', 'Kitchen_Activity_M_0', 'Morning_Meds_M_0', 'Kitchen_Activity_M_0', 'Guest_Bathroom_A_0', 'Master_Bathroom_A_0', 'Read_A_0', 'Kitchen_Activity_A_0', 'Guest_Bathroom_A_0', 'Master_Bathroom_A_0', 'Leave_Home_A_0', 'Read_A_0', 'Kitchen_Activity_A_0', 'Master_Bedroom_Activity_A_0', 'Read_A_0', 'Guest_Bathroom_A_0', 'Kitchen_Activity_A_0', 'Watch_TV_A_0', 'Read_A_0', 'Kitchen_Activity_A_0', 'Guest_Bathroom_A_0', 'Kitchen_Activity_A_0', 'Watch_TV_A_0', 'Kitchen_Activity_A_0', 'Watch_TV_A_0', 'Guest_Bathroom_A_0', 'Kitchen_Activity_A_0', 'Watch_TV_A_0', 'Guest_Bathroom_N_0', 'Kitchen_Activity_N_0', 'Morning_Meds_N_0']
# print(model.evaluate(test_seq))

# exit(0)

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
    # print('Symbol_list:')
    print(symbol_list[i])
    # print('State list')
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

exit(0)

# todo: output top1 and top2  decode results

'''
Symbol_list:
('Kitchen_Activity_M_0', 'Morning_Meds_M_0', 'Master_Bathroom_M_0', 'Guest_Bathroom_M_0', 'Master_Bedroom_Activity_M_0', 'Master_Bathroom_M_0', 'Kitchen_Activity_M_0', 'Guest_Bathroom_M_0', 'Guest_Bathroom_M_0', 'Master_Bathroom_M_0', 'Guest_Bathroom_M_0', 'Kitchen_Activity_M_0', 'Leave_Home_M_0', 'Kitchen_Activity_A_0', 'Master_Bedroom_Activity_A_0', 'Guest_Bathroom_A_0', 'Kitchen_Activity_A_0', 'Dining_Rm_Activity_A_0', 'Master_Bedroom_Activity_A_0', 'Master_Bathroom_A_0', 'Sleep_A_0', 'Watch_TV_A_0', 'Kitchen_Activity_A_0', 'Leave_Home_A_0', 'Kitchen_Activity_A_0', 'Guest_Bathroom_A_0', 'Watch_TV_A_0', 'Master_Bedroom_Activity_A_0', 'Master_Bathroom_A_0', 'Kitchen_Activity_A_0', 'Watch_TV_A_0', 'Master_Bedroom_Activity_A_0', 'Guest_Bathroom_N_0', 'Kitchen_Activity_N_0', 'Master_Bedroom_Activity_N_0', 'Master_Bathroom_N_0', 'Master_Bathroom_N_0', 'Watch_TV_N_0', 'Guest_Bathroom_N_0', 'Kitchen_Activity_N_0', 'Watch_TV_N_0', 'Read_N_0', 'Kitchen_Activity_N_0', 'Kitchen_Activity_N_0', 'Guest_Bathroom_N_0')
State list
('Kitchen_Activity', 'Morning_Meds', 'Master_Bathroom', 'Guest_Bathroom', 'Master_Bedroom_Activity', 'Master_Bathroom', 'Kitchen_Activity', 'Guest_Bathroom', 'Guest_Bathroom', 'Master_Bathroom', 'Guest_Bathroom', 'Kitchen_Activity', 'Leave_Home', 'Kitchen_Activity', 'Master_Bedroom_Activity', 'Guest_Bathroom', 'Kitchen_Activity', 'Dining_Rm_Activity', 'Master_Bedroom_Activity', 'Master_Bathroom', 'Sleep', 'Watch_TV', 'Kitchen_Activity', 'Leave_Home', 'Kitchen_Activity', 'Guest_Bathroom', 'Watch_TV', 'Master_Bedroom_Activity', 'Master_Bathroom', 'Kitchen_Activity', 'Watch_TV', 'Master_Bedroom_Activity', 'Guest_Bathroom', 'Kitchen_Activity', 'Master_Bedroom_Activity', 'Master_Bathroom', 'Master_Bathroom', 'Watch_TV', 'Guest_Bathroom', 'Kitchen_Activity', 'Watch_TV', 'Read', 'Kitchen_Activity', 'Kitchen_Activity', 'Guest_Bathroom')
'''

print('================================================')
pre_act_list = ['Kitchen_Activity_M_0', 'Morning_Meds_M_0', 'Master_Bathroom_M_0', 'Guest_Bathroom_M_0', 'Master_Bedroom_Activity_M_0', 'Master_Bathroom_M_0']
pre_act_list = ['Kitchen_Activity', 'Morning_Meds', 'Master_Bathroom', 'Guest_Bathroom_M', 'Master_Bedroom_Activity', 'Master_Bathroom']


'''
res:

[('Desk_Activity_M_0', 1.8623982760695668e-09), ('Guest_Bathroom_M_0', 2.980555827185469e-10), ('Kitchen_Activity_M_0', 1.0016603910175761e-10), ('Master_Bathroom_M_0', 1.2101368066561746e-11), ('Meditate_M_0', 1.7932082130730003e-13), ('Watch_TV_M_0', 1.1818092915624083e-14), ('Sleep_M_0', 9.791065829199752e-18), ('Read_M_0', 1.5190234024797781e-18), ('Bed_to_Toilet_M_0', 1.0935579770578827e-21), ('Chores_M_0', 1.515004274111183e-23), ('Dining_Rm_Activity_M_0', 1.6451103756130846e-26), ('Eve_Meds_M_0', 2.9856230184917954e-29), ('Leave_Home_M_0', 1.623934968595129e-30), ('Morning_Meds_M_0', 5.284790190764109e-32), ('Master_Bedroom_Activity_M_0', 2.1300100178437638e-33)]

no type info
[('Desk_Activity_M_0', 2.9687738820853098e-21), ('Guest_Bathroom_M_0', 4.746848387476407e-22), ('Kitchen_Activity_M_0', 1.5952430135128177e-22), ('Master_Bathroom_M_0', 1.9272622741847656e-23), ('Meditate_M_0', 2.855861023108703e-25), ('Watch_TV_M_0', 1.8821479111653402e-26), ('Sleep_M_0', 1.5593238460791244e-29), ('Read_M_0', 2.41919465720972e-30), ('Bed_to_Toilet_M_0', 1.7415989846691785e-33), ('Chores_M_0', 2.4127937986975524e-35), ('Dining_Rm_Activity_M_0', 2.620000603484075e-38), ('Eve_Meds_M_0', 4.754899261582551e-41), ('Leave_Home_M_0', 2.58627667833687e-42), ('Morning_Meds_M_0', 8.416549852424719e-44), ('Master_Bedroom_Activity_M_0', 3.3922511309297697e-45)]


'''

res = {}
act_type_list = ['M', 'A', 'N']
act_type_list = ['M']

for index in tools_ascc.ACTIVITY_DICT.keys():
    act = tools_ascc.ACTIVITY_DICT[index]
    for type in act_type_list:
        activity_type = type
        node = tools_ascc.Activity_Node_Observable(act, activity_type, 0)
        
        next_act = node.activity_res_generation()

        test_lis = pre_act_list
        test_lis.append(next_act)
        prob = model.evaluate(test_lis)

        print('test_lis:', test_lis)
        print('prob:', prob)
        res[next_act] = prob

print('=========================================================')

sd = sorted(res.items(), key=tools_ascc.sorter_take_count, reverse=True)
print(sd)


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

