import numpy as np
import pandas as pd
import dtw
import utility_func as utils
from statsmodels.tsa.stattools import coint

class dtw_pair_search:
    def __init__(self, df : pd.DataFrame, reload : bool):
        self.data = df
        self.reload = reload
        self.pair_list = self.all_pair_list()
        self.dtw_list = self.dtw_pair_search()
        
    def all_pair_list(self):
        name_list = self.data.columns.to_list()
        return utils.combination(name_list)
        
    def dtw_distance(self, pair : list):
        stock1 = self.data[pair[0]].dropna()
        stock2 = self.data[pair[1]].dropna()
        distance = dtw.dtw(stock1.values, stock2.values).distance
        return distance
    
    def dtw_pair_search(self):
        if self.reload:
            print("retreive last pair data")
            pair_record = pd.read_csv("D:/project/results/pair/dwt" + str(self.data.shape[0]) + ".csv")
            print(f'retreived {pair_record.shape[0]} dtw pair data')
            return pair_record.values.tolist()
        
        dist = []
        for pair in self.pair_list:
            pair_dist = self.dtw_distance(pair)
            print(f'Calculated {pair[0] + ' & ' + pair[1]} dtw distance')
            dist.append([pair[0], pair[1], pair_dist])
            
        pair_record = pd.DataFrame(dist, columns=['pair1', 'pair2', 'dist'])
        pair_record = pair_record.sort_values('dist', ascending=True)
        pair_record.to_csv("D:/project/results/pair/dwt" + str(self.data.shape[0]) + ".csv", index=False)
        return dist
            
class mr_test:
    def __init__(self, data : pd.DataFrame, pair_list : list):
        self.data = data
        self.pair_list = pair_list
        self.mr_list = self.mr_test_result()
        
    def mr_test_result(self):
        for index in range(self.pair_list.size()):
            pair_name_1 = self.pair_list[index][0]
            pair_name_2 = self.pair_list[index][1]
            
            stock1 = self.data[pair_name_1].dropna()
            stock2 = self.data[pair_name_2].dropna()
            
            result = coint(stock1, stock2)

