import matplotlib.pyplot as plt
import numpy as np
import constants

# Refer: https://www.runoob.com/matplotlib/matplotlib-hist.html

# for test 
Watch_TV = [0.5, 0.6666666666666666, 2.183333333333333, 3.283333333333333, 4.383333333333334, 4.516666666666667, 5.8, 6.45, 7.483333333333333, 9.366666666666667, 10.616666666666667, 10.866666666666667, 11.033333333333333, 14.25, 14.316666666666666, 15.183333333333334, 15.716666666666667, 15.966666666666667, 16.1, 16.75, 17.016666666666666, 17.766666666666666, 18.35, 18.366666666666667, 18.4, 18.7, 19.166666666666668, 19.266666666666666, 20.016666666666666, 21.85, 22.116666666666667, 22.833333333333332, 23.483333333333334, 23.766666666666666, 23.966666666666665, 24.016666666666666, 24.75, 24.85, 26.5, 27.4, 28.116666666666667, 29.4, 29.683333333333334, 30.083333333333332, 30.1, 31.25, 32.46666666666667, 33.13333333333333, 33.516666666666666, 33.916666666666664, 34.4, 34.8, 37.06666666666667, 38.53333333333333, 39.31666666666667, 41.36666666666667, 42.11666666666667, 43.8, 44.78333333333333, 44.9, 45.18333333333333, 45.5, 46.266666666666666, 46.86666666666667, 47.083333333333336, 51.03333333333333, 51.916666666666664, 52.21666666666667, 54.666666666666664, 56.016666666666666, 56.6, 61.86666666666667, 62.7, 67.33333333333333, 68.4, 69.43333333333334, 71.73333333333333, 73.73333333333333, 74.03333333333333, 76.31666666666666, 78.08333333333333, 82.06666666666666, 87.31666666666666, 96.26666666666667, 99.71666666666667, 100.56666666666666, 100.86666666666666, 101.7, 105.88333333333334, 107.03333333333333, 108.25, 116.98333333333333, 120.11666666666666, 122.98333333333333, 125.2, 125.25, 132.23333333333332, 139.98333333333332, 145.65, 153.91666666666666, 154.3, 177.13333333333333, 178.56666666666666]



def duration_histograms(activity,data_lis):
    data = data_lis
    
    plt.clf()
    # 绘制直方图
    plt.hist(data, bins=30, color='skyblue', alpha=0.8)
    
    # 设置图表属性
    plt.title('Activity Duration Hist')
    plt.xlabel('Duration')
    plt.ylabel('Frequency')
    
    img_file = './plot_figure/'+'gauss_' + activity  +'.png'
    plt.savefig(img_file)
    plt.clf()
    
    # 显示图表
    #plt.show()

#duration_histograms('Watch_TV', Watch_TV)
duration_histograms(constants.ACTIVITY_WATCH_TV, constants.duration_dict[constants.ACTIVITY_WATCH_TV])

for _, v_act in constants.ACTIVITY_DICT.items():
    duration_histograms(v_act, constants.duration_dict[v_act])





import collections
def get_activity_prob_dist(activity, data_list):
    #d_lis = act_duration_cnt_dict[activity]
    d_lis = data_list
    dis_p = collections.defaultdict(int)

    for d in d_lis:
        print('d:{}, intd{}'.format(d, int(d)))
        dis_p[d] = dis_p[int(d)] + 1

    f_data = []
    v_data = []

    for k,v in dis_p.items():
        f_data.append(v)
        v_data.append(k)

    data = f_data
    print('{}, {}', f_data, v_data)
    
    # 绘制直方图
    plt.hist(data, bins=30, color='skyblue', alpha=0.8)
    
    # 设置图表属性
    plt.title('Activity Duration hist')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    
    img_file = 'gauss_' + activity  +'.png'
    plt.savefig(img_file)
    
    # 显示图表
    #plt.show()


#get_activity_prob_dist('Watch_TV', Watch_TV)
