import numpy as np
import pandas as pd
import dtw
import statsmodels.api as sm
import utility_function as utils
from statsmodels.tsa.stattools import adfuller 

class pair_search:
    def __init__(self, data : object, method : str, corr_method : str,  reload : bool):
        self.data = data.train_norm_data
        self.norm_method = data.norm_method
        self.method = method
        self.corr_method = corr_method
        self.reload = reload
        self.pair_list = self.all_pair_list()
        self.corr_dict = self.corr_para() if corr_method != None else 0
        self.trade_pair_list = self.pair_search()
        
        
    def all_pair_list(self):
        name_list = self.data.columns.to_list()
        return utils.combination(name_list)
        
    def distance(self, pair : list):
        
        distance = 0
        name = pair[0] + '&' + pair[1]
        stock1 = self.data[pair[0]].dropna()
        stock2 = self.data[pair[1]].dropna()
        
        if self.corr_method != None:
            stock2 = self.corr_dict[name] * stock2
        
        if self.method == 'dtw':
            distance = dtw.dtw(stock1.values, stock2.values).distance
                
        if self.method == 'ssd':
            distance = np.sqrt(np.sum((stock1 - stock2) ** 2))
                
        return distance
    
    def pair_search(self):
        if self.reload:
            print("retreive last pair data")
            pair_record = pd.read_csv("D:/project/backtest_result/" + self.method +
                                      "/pair/train_time_" + str(self.data.shape[0]) +
                                      "_" + str(self.norm_method) +
                                      "_" + str(self.corr_method) +
                                      ".csv")
            print(f'retreived {pair_record.shape[0]} {self.method} pair data')
            output = pair_record.values.tolist()
            return output
        
        output = []
        for pair in self.pair_list:
            name = pair[0] + '&' + pair[1]
            pair_dist = self.distance(pair)
            pair_beta = self.corr_dict[name]
            print(f'Calculated {pair[0] + ' & ' + pair[1]} {self.method} distance')
            output.append([pair[0], pair[1], pair_dist, pair_beta])
                        
        pair_record = pd.DataFrame(output, columns=['pair1', 'pair2', 'dist', 'beta'])
        pair_record = pair_record.sort_values('dist', ascending=True)
        pair_record.to_csv("D:/project/backtest_result/" + self.method +
                           "/pair/train_time_" + str(self.data.shape[0]) +
                           "_" + str(self.norm_method) +
                           "_" + str(self.corr_method) + 
                           ".csv", index=False)
        output = pair_record.values.tolist()
        
        return output
    
    def corr_para(self):
        corr_dict = {}
        
        if self.reload:
            pair_record = pd.read_csv("D:/project/backtest_result/" + self.method +
                                      "/pair/train_time_" + str(self.data.shape[0]) +
                                      "_" + str(self.norm_method) +
                                      "_" + str(self.corr_method) +
                                      ".csv")
            corr_dict = pair_record[['pair1', 'pair2', 'beta']]
            pair_record['key'] = pair_record['pair1'] + '&' + pair_record['pair2']
            pair_record = pair_record.set_index('key')
            corr_dict = pair_record['beta'].to_dict()
            
            return corr_dict
        
        if self.corr_method == 'reg':
            for pair in self.pair_list:
                name = pair[0] + '&' + pair[1]
                stock1 = self.data[pair[0]].dropna()
                stock2 = self.data[pair[1]].dropna()
                
                # y, X
                reg = sm.OLS(stock1, stock2)
                model = reg.fit()
                beta = model.params
                
                corr_dict[name] = beta[pair[1]]
                print('regression of ' + name + ' finished')
                
            return corr_dict
                
                
            
        
        return 0
    
    
    
            
class mr_test:
    def __init__(self, data : pd.DataFrame, pair_list : list, reload : bool):
        self.data = data
        self.pair_list = pair_list
        self.reload = reload
        self.mr_list = self.mr_test_result()
        
    def mr_test(self, pair : list):
        
        stock1 = self.data[pair[0]]
        stock2 = self.data[pair[1]]
        spread = stock1 - stock2
            
        # DF-test, null hypo = no stationary
        #p_value = adfuller(spread, maxlag=0, regression='n', autolag=None, store=True, regresults=True)[1]
        p_value = adfuller(spread, maxlag=None, regression='n', autolag='AIC', store=True, regresults=True)[1]
        stationary = False if p_value > 0.01 else True
            
        return stationary
    
    def mr_test_result(self):
        
        if self.reload:
            result = pd.read_csv("D:/project/results/pair/dwt" + str(self.data.shape[0]) + "mr.csv")
            return result.values.tolist()
        
        result = []
        for pair in self.pair_list:
            stationary = self.mr_test(pair)
            pair.append(stationary)
            result.append(pair)
            print('finished mean-reverting test' + str(pair))
            
        mr_test_df = pd.DataFrame(result, columns=['pair1', 'pair2', 'dist', 'mr'])
        mr_test_df.to_csv("D:/project/results/pair/dwt" + str(self.data.shape[0]) + "mr.csv", index=False)
            
        return result
      

