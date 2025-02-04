from utility_function import data_preprocess
from utility_function import backtest
from utility_function import review
from utility_function import review_plot
from pair_search import pair_search
from pair_search import mr_test
import pandas as pd
import numpy as np
import dtw
import matplotlib.pyplot as plt
import random

#backtest
def main():
    path = 'D:\project\data\sp500_new\summary.csv'
    raw_data = pd.read_csv(path)
    
    data = data_preprocess(raw_data, 'day0', 1000)
    pair_data = pair_search(df = data.train_norm_data, method = 'ssd', reload = False)
    
    bt_result = backtest(data = data, pairs = pair_data , num_pair = 1000, leverage = 20, trade_gap = 1.75, reload = False)
    
    return 0

def main3():
    path = 'D:\project\data\sp500_new\summary.csv'
    raw_data = pd.read_csv(path)
    
    data = data_preprocess(raw_data, 'mean', 1000)
    pair_data = dtw_pair_search(data.train_norm_data, True)
    
    pair_list = pair_data.dtw_list[0:1000]
    #mr_data = mr_test(data.train_norm_data, pair_list, True).mr_list[0:1000]
    
    #mr_df = pd.DataFrame(mr_data, columns=['pair1', 'pair2', 'dist', 'mr'])
    #mr_df = mr_df[~mr_df['mr']]
    
    #mr_list = mr_df.values.tolist()
    tmp = []
    for pair in pair_list:
        name = pair[0] + "&" + pair[1]
        bt_res = pd.read_csv('D:/project/backtest_result/dtw/' + name + '.csv')
        bt_res = bt_res['value'].to_list()[-1]
        tmp.append(bt_res)
    print(tmp)
    
    print(np.mean(tmp))
    backtest(data, pair_list, 20, 1.75)
    
    return 0

#data explore
def main2():
    path = 'D:\project\data\sp500\summary.csv'
    raw_data = pd.read_csv(path)
    
    data = data_preprocess(raw_data, 'mean', 1000)
    pair_data = dtw_pair_search(data.train_norm_data, True)
    pair_list = pair_data.dtw_list[0:1000]
    
    data2 = data.train_norm_data
    test_pair = ['A', 'TGT'] #
    test_pair2 = ['V', 'WMT']
    
    for pair in pair_list:
    
        stock1 = data2[pair[0]]
        stock2 = data2[pair[1]]
    
        tmp = dtw.dtw(stock1.values, stock2.values)
        foo, bar = tmp.index1, tmp.index2
        tmp2 = foo - bar # x-y
        tmp2 = np.array(tmp2, dtype=int)
        print(len(tmp2))
        
        hist, bins = np.histogram(tmp2, bins=max(tmp2)-min(tmp2)+1)
        plt.bar(bins[:-1], hist, width=np.diff(bins), align='edge')
        plt.xlabel('Value')
        plt.ylabel('Frequency')
        plt.title('Histogram of Random Data')
        plt.show()  
        
        #plt.plot(stock1)
        #plt.plot(stock2)
        plt.plot(stock1 - stock2)
        plt.show()
    
    return 0

#result review
def main4():
    
    path = 'D:\project\data\sp500_new\summary.csv'
    raw_data = pd.read_csv(path)
    
    data = data_preprocess(raw_data, 'day0', 1000)
    pair_data = pair_search(data = data, method = 'ssd', reload = True)
    pairs_list = pair_data.trade_pair_list[0:1000]
    
    master_control = False
    bt_result = backtest(data = data, pairs = pair_data ,
                         num_pair = 1000, leverage = 20, trade_gap = 1.75, transaction_cost = 0, reload = master_control)
    result = review(bt_result, reload = master_control)
    plotting = review_plot(result)
    tpm =plotting.plot_aggre_result()
    tpp =plotting.plot_compare_result()
    
    return 0

# mr testing
def main5():
    
    
    path = 'D:\project\data\sp500_new\summary.csv'
    raw_data = pd.read_csv(path)
    
    data = data_preprocess(raw_data, 'mean', 1000)
    pair_data = dtw_pair_search(data.train_norm_data, True)
    
    pair_list = pair_data.dtw_list[0:1000]
    #pair_list = [['GOOG', 'GOOGL']]
    
    result = mr_test(data.train_norm_data, pair_list, False).mr_list
    print(result[3])
        
    return 0

def mpt():
    ssd_data = pd.read_csv('D:/project/backtest_result/aggre/' + 'ssd' + '_aggregated_review.csv')
    dtw_data = pd.read_csv('D:/project/backtest_result/aggre/' + 'dtw' + '_aggregated_review.csv')
        
    x = ssd_data['num']
    y_ssd = ssd_data['total_return']
    y_dtw = dtw_data['total_return']
        
    plt.plot(x, y_ssd, label='ssd', linestyle='-')
    plt.plot(x, y_dtw, label='dtw', linestyle='-')
    plt.xlabel('num')
    plt.ylabel('return')
    plt.grid(True)
    plt.show()
    plt.savefig('D:/project/backtest_result/plot/' + 'compare' + '_aggre_result_return.png')
    plt.close()
        
    y_ssd_2 = ssd_data['max_down']
    y_dtw_2 = dtw_data['max_down']
        
    plt.plot(x, y_ssd_2, label='ssd', linestyle='-')
    plt.plot(x, y_dtw_2, label='dtw', linestyle='-')
    plt.grid(True)    
    plt.savefig('D:/project/backtest_result/plot/' + 'compare' + '_aggre_result_maxdown.png')
    plt.show()
    
    return 0

main4()
#mpt()
