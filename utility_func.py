import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class data_preprocess:
    def __init__(self, df : pd.DataFrame, norm : str, train_test_split: int):
        self.data = df
        self.norm_method = norm
        self.train_test_split = train_test_split
        self.train_data, self.test_data = self.split_data(self.data)
        self.train_norm_data, self.test_norm_data = self.normalize()
        self.average_spread = self.get_average_norm_spread()
        
    # This function split data into train and test set, train data used in scaler
    def split_data(self, data : pd.DataFrame):
        size = data.shape[0]
        train_data = data[0:self.train_test_split]
        test_data = data[self.train_test_split:size]
        
        # This part drop columns with null values
        columns_with_nulls = train_data.columns[train_data.isnull().any()].tolist()
        self.data = self.data.drop(columns=columns_with_nulls)
        train_data = train_data.drop(columns=columns_with_nulls)
        test_data = test_data.drop(columns=columns_with_nulls)
        
        return train_data, test_data
        
    # This function return a dict of data mean
    def scaler_fit_mean(self):
        col_name = self.train_data.columns
        scaling = {}
        for ticket in col_name:
            scaling[ticket] = np.mean(self.train_data[ticket])
        return scaling
    
    # This function return df with normalized data
    def scaler_transform_mean(self):
        data = self.data.copy()
        for element in data.columns:
            data[element] = data[element]/self.scaler[element]
        return data.reset_index(drop=True)
    
    # control the method of normalization
    def normalize(self):
        if self.norm_method == 'mean':
            self.scaler = self.scaler_fit_mean()
            self.norm_data = self.scaler_transform_mean()
            return self.split_data(self.norm_data)
        
    # Return a dict for the average spread of the pairs
    def get_average_norm_spread(self):
        all_pairs = combination(self.train_data.columns)
        spread_list = {}
        for pair in all_pairs:
            stock1 = self.train_norm_data[pair[0]]
            stock2 = self.train_norm_data[pair[1]]
            spread = np.mean(np.abs(stock1 - stock2))
            name = pair[0] + '&' + pair[1]
            spread_list[name] = spread
        return spread_list
            
            
        
            
class backtest:
    def __init__(self, data : object, pairs_list : list , leverage : int, trade_gap : float):
        self.price_data = data.test_data
        self.price_norm_data = data.test_norm_data
        self.pairs_list = pairs_list
        self.average_spread = data.average_spread
        self.leverage = leverage
        self.trade_gap = trade_gap
        self.backtest()

    def trade_execution(self, pair : list):

        trade_record_list = []
        name = pair[0] + '&' + pair[1]
        
        stock1 = self.price_norm_data[pair[0]].values.tolist()
        stock2 = self.price_norm_data[pair[1]].values.tolist()
        stock1_price = self.price_data[pair[0]].values.tolist()
        stock2_price = self.price_data[pair[1]].values.tolist()
        
        trade_period = min(len(stock1), len(stock2))
        spread = self.average_spread[name]
        in_position = False
        initial_value = 10000
        
        first_record = [pair[0], 0, stock1_price[0], pair[1], 0, stock2_price[0], 0, in_position, initial_value]
        trade_record_list.append(first_record)
        for i in range(1, trade_period):
            
            direction_before = bool(stock1[i-1] >= stock2[i-1])
            direction = bool(stock1[i] >= stock2[i])
            price_cross = (direction_before != direction)
            record_before = trade_record_list[i-1]
            invest = initial_value * self.leverage
            
            # update position status when no event
            if True:
                record = record_before
                record[2] = stock1_price[i] if not np.isnan(stock1_price[i]) else record_before[2]
                record[5] = stock2_price[i] if not np.isnan(stock2_price[i]) else record_before[5]
                record[6] = i
                record[8] = initial_value + record[1]*record[2] + record[4]*record[5]
                
            # not in position, meet enter criteria
            if (not in_position) & (abs(stock1[i] - stock2[i]) > spread * self.trade_gap):
                in_position = True
                record[1] = -invest/stock1_price[i] if direction else invest/stock1_price[i]
                record[4] = invest/stock2_price[i] if direction else -invest/stock2_price[i]
                record[7] = in_position
                
            # in position, price cross
            if in_position & price_cross:
                in_position = False
                initial_value = record[8]
                record[1] = 0
                record[4] = 0
                record[7] = in_position
                
            # in postion, forcibly liquidate when time end
            if i == trade_period - 1:
                in_position = False
                record[1] = 0
                record[4] = 0
                record[7] = in_position
                break
            
            trade_record_list.append(record.copy())
            
        return trade_record_list

    def backtest(self):
        
        for pair in self.pairs_list:
            
            trade_record = self.trade_execution(pair)
            name = pair[0] + '&' + pair[1]
            
            trade_record_df = pd.DataFrame(trade_record, columns=['stock1', 'unit1', 'price1','stock2', 'unit2', 'price2', 'time', 'in_position', 'value'])     
            trade_record_df.to_csv('D:/project/backtest_result/dtw/' + name + '.csv', index=False)
            print(f'Pair: {name} trade execution finished')                
            
        return 0
    
    
    
    
class review:
    def __init__(self, pair_list : list, reload : bool):
        self.reload = reload
        self.pair_list = pair_list
        self.aggre_result = self.aggregated_result_output()
        
    def aggregated_result_output(self):
        
        output_path = 'D:/project/backtest_result/aggre/' + 'meta_dtw' + '.csv'
        
        if self.reload:
            aggregated_result_df = pd.read_csv(output_path)
            return aggregated_result_df
        
        for pair in self.pair_list:
            aggregated_result_df = self.aggregated_result(pair)
            aggregated_result_df.to_csv(output_path, index=False)
            
        return aggregated_result_df
            
    
    def aggregated_result(self, pair_list : list):
        
        data = pd.read_csv('D:/project/backtest_result/dtw/' + 'GOOG&GOOGL' + '.csv')
        
        value = np.zeros(data.shape[0])
        aggre_result_list = []
        for pair in pair_list:
            name = pair[0] + '&' + pair[1]
            path = 'D:/project/backtest_result/dtw/' + name + '.csv'
            trade_record = pd.read_csv(path)
            value += trade_record['value']
            
        percentage_change = (value[len(value)-1] - value[0]) / value[0]
        max_down = min(value)/value[0]
        aggre_result_list = [percentage_change, max_down]
            
            
        print(f'Aggregated {len(pair_list)} review finished')
            
        #aggregated_result_df = pd.DataFrame(aggre_result_list, columns=['return', 'max_down'])
        
        
        return aggre_result_list
            
class review_plot:
    def __init__(self, review : object):
        self.aggre_result = review.aggre_result
        
    def plot_aggre_result(self):
        x = range(10, 1000, 10)
        y = self.aggre_result['value']
        
        plt.plot(x, y)
        plt.show()
        
        return 0                    
    
        
            
########################################################################
def combination(list):
    output = []
    for i in range(len(list)-1):
        for j in range(len(list)-i-1):
            output.append([list[i], list[i+j+1]])
    return output