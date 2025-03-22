from utility_function import data_preprocess
from utility_function import backtest
from utility_function import review
from utility_function import review_plot
from pair_search import pair_search
from pair_search import mr_test
import pandas as pd
import numpy as np
import dtw
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import random
import statsmodels.api as sm

def main_train():
    path = 'D:\\project\\data\\sp500_new\\summary.csv'
    raw_data = pd.read_csv(path)
    
    test_data = raw_data.head(20)
    test_data = test_data[['A', 'AAPL', 'ANSS', 'AOS']]
    
    data = data_preprocess(raw_data, 'day0', 1000)
    #pair_data_ssd = pair_search(data = data, method = 'ssd', corr_method = 'reg', reload = True)
    pair_data_dtw = pair_search(data = data, method = 'dtw', corr_method = 'reg', reload = False)
    
    #print(pair_data_ssd.corr_dict)
    bt_result_dtw = backtest(data = data, pairs = pair_data_dtw ,
                         num_pair = 1000, leverage = 30, trade_gap = 2, transaction_cost = 0, reload = False)
    
    result_dtw = review(bt_result_dtw, reload = False)
    plot_dtw = review_plot(result_dtw)
    
    plot_dtw.plot_aggre_result()
    
    
    return 0

def main_plot():
    path = 'D:\\project\\data\\sp500_new\\summary.csv'
    raw_data = pd.read_csv(path)
    
    data = data_preprocess(raw_data, 'day0', 1000)
    pair_data_ssd = pair_search(data = data, method = 'ssd', reload = True)
    pair_data_dtw = pair_search(data = data, method = 'dtw', reload = True)
    
    data2 = data.train_norm_data
    test_pair = ['A', 'TGT'] #
    test_pair2 = ['V', 'WMT']
    
    pair_list = [['KO', 'PEP']]
    pair_list = [pair_data_ssd.trade_pair_list[199]]
    
    for pair in pair_list:
    
        pair_name = pair[0] + "&" + pair[1]
        stock1 = data2[pair[0]]
        stock2 = data2[pair[1]]
        x = range(len(stock1))
        
        if pair_name == 'CL&SO' or pair_name ==  'OMC&RTX':
            spread = stock1 - stock2
            sd = np.std(spread)
            print(f"{sd} {pair_name}")
            
        plt.title(f"Comparison of top 200 SSD pair {pair[0]} & {pair[1]}")
        plt.plot(x, stock1, label=pair[0], linestyle='-')
        plt.plot(x, stock2, label=pair[1], linestyle='-')
        plt.grid(True)
        plt.savefig('D:\\project\\backtest_result\\plot\\day0dtw200.png')
        plt.show()
    
    
    
    
    return 0

#data explore
def main_explore():
    path = 'D:\\project\\data\\sp500_new\\summary.csv'
    raw_data = pd.read_csv(path)
    
    data = data_preprocess(raw_data, 'mean', 1000)
    pair_data_ssd = pair_search(data = data, method = 'ssd', reload = True)
    pair_data_dtw = pair_search(data = data, method = 'dtw', reload = True)
    
    data2 = data.train_norm_data
    test_pair = ['A', 'TGT'] #
    test_pair2 = ['V', 'WMT']
    
    pair_list = ['KO', 'PEP']
    
    for pair in pair_list:
    
        stock1 = data2[pair[0]]
        stock2 = data2[pair[1]]
    
        tmp = dtw.dtw(stock1.values, stock2.values)
        foo, bar = tmp.index1, tmp.index2
        tmp2 = foo - bar # x-y
        tmp2 = np.array(tmp2, dtype=int)
        print(tmp2)
        
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
def main_review():
    
    path = 'D:\\project\\data\\sp500_new\\summary.csv'
    raw_data = pd.read_csv(path)
    
    train_control = True
    
    data = data_preprocess(raw_data, 'day0', 1000)
    pair_data_ssd = pair_search(data = data, method = 'ssd', corr_method = None, reload = train_control)
    pair_data_dtw = pair_search(data = data, method = 'dtw', corr_method = None, reload = train_control)
    #pairs_list = pair_data.trade_pair_list[0:1000]
    
    master_control = False
    
    bt_result_ssd = backtest(data = data, pairs = pair_data_ssd ,
                         num_pair = 1000, leverage = 30, trade_gap = 2.5, transaction_cost = 0, reload = master_control)
    bt_result_dtw = backtest(data = data, pairs = pair_data_dtw ,
                         num_pair = 1000, leverage = 30, trade_gap = 2.5, transaction_cost = 0, reload = master_control)
    #"""
    result_ssd = review(bt_result_ssd, reload = master_control)
    result_dtw = review(bt_result_dtw, reload = master_control)
    
    plot_ssd = review_plot(result_ssd)
    plot_dtw = review_plot(result_dtw)
    
    #plot_ssd.plot_aggre_result()
    #plot_dtw.plot_aggre_result()
    plot_dtw.plot_all_compare_result()
    #"""
    return 0












def mpt():
    path = 'D:\\project\\data\\sp500_new\\summary.csv'
    raw_data = pd.read_csv(path)
    
    data = data_preprocess(raw_data, 'mean', 1000)
    pair_data_ssd = pair_search(data = data, method = 'ssd', reload = True)
    pair_data_dtw = pair_search(data = data, method = 'dtw', reload = True)
   
    pairs_list = pair_data_ssd.trade_pair_list[0:1000]
    data2 = data.train_norm_data
    
    
    pair_list = [['KO', 'PEP']]
    for pair in pair_list:
        stock1 = np.array(data2[pair[0]]).reshape(-1, 1) 
        stock2 = np.array(data2[pair[1]])
        
        # fit a model
        reg = LinearRegression(fit_intercept=False)
        model = reg.fit(stock1, stock2)
        
        
        
        
        y_pred = model.predict( stock1)

        # Print coefficients
        print(f"Intercept: {model.intercept_}")
        print(f"Coefficient: {model.coef_[0]}")

        print(model.score(stock1, stock2))

        # Plot the regression line
        plt.scatter( stock1, stock2, color='blue', label="Data points")
        plt.plot( stock1, y_pred, color='red', label="Regression line")
        plt.legend()
        plt.show()
        
        
        print(model)
   
    return 0

def mpt2():
    path = 'D:\\project\\data\\sp500_new\\summary.csv'
    raw_data = pd.read_csv(path)
    
    data = data_preprocess(raw_data, 'mean', 1000)
    pair_data_ssd = pair_search(data = data, method = 'ssd', corr_method = '', reload = True)
    #pair_data_dtw = pair_search(data = data, method = 'dtw', reload = True)
   
    pairs_list = pair_data_ssd.trade_pair_list[0:1000]
    data2 = data.train_norm_data
    
    #tmp = pair_data_ssd.corr_dict
    #print(tmp)
    
    pair_list = [['KO', 'PEP']]
    for pair in pair_list:
        
        stock1 = data2[pair[0]]
        stock2 = data2[pair[1]]
        
        # fit a model
        reg = sm.OLS(stock1, stock2)
        model = reg.fit()
        beta = model.params
        beta = beta[pair[1]]
        
        print(beta.dtype)
        print(stock2 * beta)
        
        
        #print(model.summary())
        
        
      
    return 0

main_review()
#mpt2()