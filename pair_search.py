import numpy as np
import pandas as pd
import dtw
import utility_function as utils
from statsmodels.tsa.stattools import coint
from statsmodels.tsa.stattools import adfuller

class pair_search:
    def __init__(self, data : object, method : str, reload : bool):
        self.data = data.train_norm_data
        self.norm_method = data.norm_method
        self.method = method
        self.reload = reload
        self.pair_list = self.all_pair_list()
        self.trade_pair_list = self.pair_search()
        
    def all_pair_list(self):
        name_list = self.data.columns.to_list()
        return utils.combination(name_list)
        
    def distance(self, pair : list):
        stock1 = self.data[pair[0]].dropna()
        stock2 = self.data[pair[1]].dropna()
        
        if self.method == 'dtw':
            distance = dtw.dtw(stock1.values, stock2.values).distance
                
        if self.method == 'ssd':
            distance = np.sqrt(np.sum((stock1 - stock2) ** 2))
                
        return distance
    
    def pair_search(self):
        if self.reload:
            print("retreive last pair data")
            pair_record = pd.read_csv("D:/project/backtest_result/" + self.method + "/pair/train_time_" + str(self.data.shape[0]) + "_" + str(self.norm_method) + ".csv")
            print(f'retreived {pair_record.shape[0]} {self.method} pair data')
            output = pair_record.values.tolist()
            return output
        
        output = []
        for pair in self.pair_list:
            pair_dist = self.distance(pair)
            print(f'Calculated {pair[0] + ' & ' + pair[1]} {self.method} distance')
            output.append([pair[0], pair[1], pair_dist])
                        
        pair_record = pd.DataFrame(output, columns=['pair1', 'pair2', 'dist'])
        pair_record = pair_record.sort_values('dist', ascending=True)
        pair_record.to_csv("D:/project/backtest_result/" + self.method + "/pair/train_time_" + str(self.data.shape[0]) + "_" + str(self.norm_method) + ".csv", index=False)
        output = pair_record.values.tolist()
        
        return output
      

