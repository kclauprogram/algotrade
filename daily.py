from utility_function import data_preprocess
from utility_function import backtest
from utility_function import review
from utility_function import review_plot
from pair_search import pair_search
from pair_search import mr_test
import data_IB_extract as ibex 
import pandas as pd
import numpy as np
import dtw
import matplotlib.pyplot as plt
import random
import pypfopt as ppo
import quantstats as qstat

class data_pipeline:
    def __init__(self, data_source_path : str, data_store_path : str):
        self.date_source_path = data_source_path
        self.date_store_path = data_store_path
        self.name_list = pd.read_csv('D:/project/data/name.csv')['Symbol']
        
    def read_data(self):
        new_data = {}
        for stock in self.name_list:
            data = pd.read_csv(self.date_source_path + stock + '.csv')
            new_data[stock] = data.tolist()[-1]
        
        return new_data
    
    def data_summary(self):
        new_data = []
        for stock in self.name_list:
            data = pd.read_csv(self.date_store_path + stock + '.csv', index_col='date')
            data[stock] = data['close']
            new_data.append(data[stock])
            
            
        output = pd.concat(new_data, axis=1, join='outer')
        
        return output
    
    def data_update(self):
        
        for stock in self.name_list:
            
            data = pd.read_csv(self.date_store_path + stock + '.csv' , index_col='date')
            new_data = pd.read_csv(self.date_source_path + stock + '.csv', index_col='date')
            
            data = pd.concat([data, new_data], axis=0, join='outer')
            data = data.drop_duplicates()
            
            data.to_csv(self.date_store_path + stock + '.csv')
            print('update ' + stock + ' data')
        
        return 0
    
    def generate_report():
        report = 0
        
        
        return report
    
    
def test():
    store_path = 'D:/project/data/sp500/'
    source_path = 'D:/project/data/sp500_tmp/'

    name_list = ['A', 'SW']
    new_data = []
    for stock in name_list:
        
            data = pd.read_csv(store_path + stock + '.csv', index_col='date')
            new_data = pd.read_csv(source_path + stock + '.csv', index_col='date')
        
            print(data.shape)
            print(new_data.shape)
            
            data = pd.concat([data, new_data], axis=0, join='outer')
            data = data.drop_duplicates()
            print(data.shape)
    data.to_csv('D:/project/data/test.csv')
    print(data)
            
    return 0

def main():
    
    store_path = 'D:/project/data/sp500/'
    source_path = 'D:/project/data/sp500_tmp/'

    data_object = data_pipeline(data_source_path=source_path, data_store_path=store_path)

    master_control = False
    
    ### daliy stock price extract
    extract_control = False
    if extract_control or master_control:
        ibex.main(duration = '3 D')
    
    ### update stored data form source 
    update_control = False
    if update_control or master_control:
        df = data_object.data_update()
        print('updated to latest price')
        
    #### update data summary for training
    update_sum_control = True
    if update_sum_control or master_control:
        df = data_object.data_summary()
        df.to_csv('D:/project/data/sp500_work/summary.csv')
        print('price summary ready')
    
    #### generate trade signal report from data source
    report_control = False
    if report_control or master_control:
        df = data_object.data_summary()
        df.to_csv('D:/project/trade_report.csv')
        print('report ready')
    
    return 0

#test()
main()