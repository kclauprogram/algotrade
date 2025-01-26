from utility_func import data_preprocess
from utility_func import backtest
from utility_func import review
from utility_func import review_plot
from pair_search import dtw_pair_search
import pandas as pd
import dtw
import matplotlib.pyplot as plt
import random

def main():
    path = 'D:\project\data\sp500\summary.csv'
    raw_data = pd.read_csv(path)
    
    data = data_preprocess(raw_data, 'mean', 1000)
    pair_data = dtw_pair_search(data.train_norm_data, True)
    
    pair_list = pair_data.dtw_list[0:1000]
    pairs_list = [['EXPD', 'ROK']]
    
    bt_results = backtest(data, pair_list, 20, 1.75)
    
    return 0

def main3():
    
    path = 'D:\project\data\sp500\summary.csv'
    raw_data = pd.read_csv(path)
    
    data = data_preprocess(raw_data, 'mean', 1000)
    pair_data = dtw_pair_search(data.train_norm_data, True)
    pair_list = pair_data.dtw_list[0:1000]
    
    tmp = review(False)
    review_list = []
    for i in range(100):
        pair_list = pair_data.dtw_list[0 : i * 10 + 10]
        result = tmp.aggregated_result(pair_list)
        review_list.append(result)
        #print('finished' + str(i*10))
    
    foo = pd.DataFrame(review_list, columns=['return', 'max_down'])
    foo.to_csv('D:/project/backtest_result/aggre/meta_dtw.csv', index=False)
    
    return 0

def main2():
    path = 'D:\project\data\sp500\summary.csv'
    raw_data = pd.read_csv(path)
    
    data = data_preprocess(raw_data, 'mean', 506)
    data2 = data.train_norm_data
    test_pair = ['A', 'TGT']
    
    print(data.average_spread)
    
    stock1 = data2[test_pair[0]]
    stock2 = data2[test_pair[1]]
    
    tmp = dtw.dtw(stock1.values, stock2.values)
    foo, bar = tmp.index1, tmp.index2
    tmp2 = foo - bar # x-y
    print(tmp2)
    
    plt.plot(stock1)
    plt.plot(stock2)
    plt.show()
    
    return 0

def main4():
    
    path = 'D:\project\data\sp500\summary.csv'
    raw_data = pd.read_csv(path)
    
    #data = data_preprocess(raw_data, 'mean', 1000)
    #pair_data = dtw_pair_search(data.train_norm_data, True)
    #pairs_list = pair_data.dtw_list[0:1000]
    
    result = review([''], True)
    plotting = review_plot(result)
    plotting.plot_aggre_result()
    
    return 0

main4()