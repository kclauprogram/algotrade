import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm


class data_preprocess:
    def __init__(self, df : pd.DataFrame, norm : str, train_test_split: int):
        self.data = df
        self.norm_method = norm
        self.train_test_split = train_test_split
        self.train_data, self.test_data = self.split_data(self.data)
        self.train_norm_data, self.test_norm_data = self.normalize()
        
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
        
    # This function return a dict of data scalar used
    def fit(self):
        col_name = self.train_data.columns
        scaling = {}
        
        if self.norm_method == 'mean':
            for ticket in col_name:
                scaling[ticket] = np.mean(self.train_data[ticket])
                
        if self.norm_method == 'day0':
            for ticket in col_name:
                scaling[ticket] = self.train_data[ticket].tolist()[0]    
                                
        return scaling
    
    # This function return df with normalized data
    def transform(self):
        data = self.data.copy()
        for element in data.columns:
            data[element] = data[element]/self.scaler[element]
        return data.reset_index(drop=True)
    
    # control the method of normalization
    def normalize(self):
        self.scaler = self.fit()
        self.norm_data = self.transform()
        return self.split_data(self.norm_data)
                
            
        
            
class backtest:
    def __init__(self, data : object, pairs : object , num_pair : int, leverage : int, trade_gap : float, transaction_cost : float, reload : bool):
        
        # from data
        self.price_data = data.test_data
        self.price_norm_data = data.test_norm_data
        #self.average_spread = data.average_spread
        
        # from pairs
        self.pairs_list = pairs.trade_pair_list[0:num_pair]
        self.method = pairs.method
        self.corr_method = pairs.corr_method
        self.beta_dict = pairs.corr_dict
        
        # trade criteria
        self.propotion = pairs.corr_dict
        self.average_spread = trade_rule(data = data, pairs = pairs, num_pairs=num_pair).get_average_norm_spread()
        
        # from self
        self.leverage = leverage
        self.trade_gap = trade_gap
        self.transaction_cost = transaction_cost
        self.reload = reload
        
        # output
        self.fail_pair_list = []
        
        ## output of trade results
        self.backtest()
        ## output of failed pairs
        df_fail_pair = pd.DataFrame(self.fail_pair_list, columns=['name'])
        df_fail_pair.to_csv('D:/project/backtest_result/' + self.method + '/pair/failed_pair.csv', index=False)

    def trade_execution(self, pair : list):
        
        name = pair[0] + '&' + pair[1]
        beta = self.beta_dict[name] if self.corr_method != None else 1
        
        
        stock1 = self.price_norm_data[pair[0]].values.tolist()
        stock2 = self.price_norm_data[pair[1]] if self.corr_method != None else self.price_norm_data[pair[1]] * beta
        stock2 = stock2.values.tolist()
        stock1_price = self.price_data[pair[0]].values.tolist()
        stock2_price = self.price_data[pair[1]].values.tolist()
        
        
        trade_period = max(len(stock1), len(stock2))
        spread = self.average_spread[name]
        in_position = False
        hold_time = 0
        failed_pair = False
        initial_value = 10000
        
        # [name of stock1, unit of stock1, price of stock1, name of stock2, unit of stock2, price of stock2, time, in_position, profolio value]
        trade_record_list = [[pair[0], 0, stock1_price[0], pair[1], 0, stock2_price[0], 0, in_position, initial_value]]
        for i in range(1, trade_period):
            
            
            direction_before = bool(stock1[i-1] >= stock2[i-1])
            direction = bool(stock1[i] >= stock2[i])
            price_cross = (direction_before != direction)
            record_before = trade_record_list[i-1]
            invest = min(initial_value * self.leverage, 10000 * self.leverage)
            #invest = initial_value * self.leverage
            
            # if stop trade criteria is true, stop the profolio update
            price_missing = np.isnan(stock1_price[i]) or np.isnan(stock2_price[i]) 
                       
            if price_missing:
                record = record_before
                record[2] = record_before[2] if np.isnan(stock1_price[i]) else stock1_price[i]
                record[5] = record_before[5] if np.isnan(stock2_price[i]) else stock2_price[i]
                record[6] = i
                in_position = False
                record[1] = 0
                record[4] = 0
                record[7] = in_position
                trade_record_list.append(record.copy())
                continue
                
                
            # update position status when no event
            if True:
                hold_time = hold_time + 1 if in_position else 0    
                record = record_before
                
                # stop trading the failed pairs
                if failed_pair == True:
                    in_position = False
                    record[1] = 0
                    record[4] = 0
                    record[7] = in_position
                    trade_record_list.append(record.copy())
                    continue
                
                record[2] = stock1_price[i]
                record[5] = stock2_price[i] 
                record[6] = i
                record[8] = initial_value + record[1]*record[2] + record[4]*record[5]
                
            # not in position, meet enter criteria
            if (not in_position) & (abs(stock1[i] - stock2[i]) > spread * self.trade_gap):
                in_position = True
                record[1] = -invest/stock1_price[i] if direction else invest/stock1_price[i]
                record[4] = invest/stock2_price[i]/beta if direction else -invest/stock2_price[i]/beta
                record[7] = in_position
                
            # in position, price cross
            if in_position & price_cross:
                in_position = False
                initial_value = record[8] * (1 - self.transaction_cost)
                record[1] = 0
                record[4] = 0
                record[7] = in_position
                record[8] = record[8] * (1 - self.transaction_cost)
                
            # in postion, forcibly liquidate when time end
            if i == trade_period - 1:
                in_position = False
                initial_value = record[8] * (1 - self.transaction_cost)
                record[1] = 0
                record[4] = 0
                record[7] = in_position
                record[8] = record[8] * (1 - self.transaction_cost)
                
                
            ##### For swing trade rule, liquidate every 40 trade days
            if hold_time >= 40 and in_position:
                in_position = False
                initial_value = record[8] * (1 - self.transaction_cost)
                record[1] = 0
                record[4] = 0
                record[7] = in_position
                record[8] = record[8] * (1 - self.transaction_cost)
                if record[8] < 0:
                    failed_pair = True
                    self.fail_pair_list.append(name)
                
            
            trade_record_list.append(record.copy())

        return trade_record_list[:-1]

    def backtest(self):
        
        if self.reload:
            return 0
        
        for pair in self.pairs_list:
            
            trade_record = self.trade_execution(pair)
            name = pair[0] + '&' + pair[1]
            
            trade_record_df = pd.DataFrame(trade_record, columns=['stock1', 'unit1', 'price1','stock2', 'unit2', 'price2', 'time', 'in_position', 'value'])     
            trade_record_df.to_csv('D:/project/backtest_result/' + self.method + '/' + name + '.csv', index=False)
            print(f'Pair: {name} trade execution finished')                
            
        return 0
    
    
class trade_rule:
    def __init__(self, data : object, pairs : object, num_pairs : int):
        
        self.pair_list = pairs.trade_pair_list[0:num_pairs]
        self.corr_method = pairs.corr_method
        self.corr_dict = pairs.corr_dict
        
        self.data = data.train_norm_data
        
    def get_average_norm_spread(self):
        spread_dict = {}
        
        for pair in self.pair_list:
            name = pair[0] + '&' + pair[1]
            stock1 = self.data[pair[0]]
            stock2 = self.data[pair[1]] if self.corr_method is None else self.data[pair[1]] * self.corr_dict[name]
            spread = np.mean(np.abs(stock1 - stock2))
            
            spread_dict[name] = spread
            
        return spread_dict
        
    def reg_para(self):
        reg_para_dict = {}
        
        for pair in self.pair_list:
            stock1 = self.data[pair[0]]
            stock2 = self.data[pair[1]]
            
            model = sm.OLS()
            model.OLS(stock1, stock2)
            name = pair[0] + '&' + pair[1]
            reg_para_dict[name] = model.params
            
        return reg_para_dict
        
    
        
        
class review:
    def __init__(self, result : object, reload : bool):
        self.reload = reload
        self.pair_list = result.pairs_list
        self.method = result.method
        self.datasize = result.price_data.shape[0]
        self.tmp_result = np.zeros(len(self.pair_list)).tolist()
        self.aggre_result = self.aggregated_result_output()
        
    def aggregated_result_output(self):
        
        output_path = 'D:/project/backtest_result/aggre/' + self.method + '_aggregated_review.csv'
        
        if self.reload:
            aggregated_result_df = pd.read_csv(output_path)
            return aggregated_result_df
        
        review_list = []
        for i in range(len(self.pair_list)):
            result = self.aggregated_result(i)
            review_list.append(result)
        
        aggregated_result_df = pd.DataFrame(review_list, columns=['num', 'total_return', 'max_down', 'sharpe_ratio' , 'return', 'is_profit', 'ftmo'])
        aggregated_result_df.to_csv('D:/project/backtest_result/aggre/' + self.method + '_aggregated_review.csv', index=False)
            
        return aggregated_result_df
    
    def individual_result(self, pair : list):
        
        name = pair[0] + '&' + pair[1]
        data = pd.read_csv('D:/project/backtest_result/' + self.method + '/' + name + '.csv')
        
        value = data['value'].tolist()
            
        percentage_change = (value[len(value)-1] - value[0]) / value[0]
        is_profit = percentage_change > 0
        
        print(f'Individual review of {name} finished')
            
        return [percentage_change, is_profit]       
    
    def aggregated_result(self, index : int):

        name = self.pair_list[index][0] + '&' + self.pair_list[index][1]
        path = 'D:/project/backtest_result/' + self.method + '/' + name + '.csv'
        trade_record = pd.read_csv(path)
            
        self.tmp_result[index] = trade_record['value'] if index == 0 else trade_record['value'] + self.tmp_result[index-1]
            
        value = self.tmp_result[index]
        indi_result_list = self.individual_result(self.pair_list[index])            
            
        total_percentage_change = (value[len(value)-1] - value[0]) / value[0]
        max_down = min(value)/value[0]
        norm_value = value / value[0]
        sharpe_ratio = (total_percentage_change - 0.055)/np.std(norm_value)
        
        first_40 = (value[40] - value[0]) / value[0]

        aggre_result_list = [index+1, total_percentage_change, max_down, sharpe_ratio, indi_result_list[0], indi_result_list[1], first_40]
    
        print(f'Aggregated {index+1} review and {name} review finished')
            
        return aggre_result_list
            
class review_plot:
    def __init__(self, review : object):
        self.aggre_result = review.aggre_result
        self.method = review.method
        
        self.ssd_data = pd.read_csv('D:/project/backtest_result/aggre/' + 'ssd' + '_aggregated_review.csv')
        self.dtw_data = pd.read_csv('D:/project/backtest_result/aggre/' + 'dtw' + '_aggregated_review.csv')
        
        self.features_list = ['total_return', 'max_down', 'sharpe_ratio', 'ftmo']
        
    def plot_aggre_result(self):
        x = self.aggre_result['num']
        y = self.aggre_result['total_return']
        y_2 = self.aggre_result['max_down']
        
        plt.plot(x, y)
        plt.savefig('D:/project/backtest_result/plot/' + self.method + '_aggre_result_return.png')
        plt.show()
        
        plt.plot(x, y_2)
        plt.savefig('D:/project/backtest_result/plot/' + self.method + '_aggre_result_maxdown.png')
        plt.show()
        
        return 0
    
    def plot_compare_result(self, feature : str):
        
        x = self.ssd_data['num']
        y_ssd = self.ssd_data[feature]
        y_dtw = self.dtw_data[feature]
        
        plt.plot(x, y_ssd, label='ssd', linestyle='-')
        plt.plot(x, y_dtw, label='dtw', linestyle='-')
        plt.grid(True)
        plt.xlabel('Number of pairs')
        plt.ylabel(feature)
        plt.title('Comparison of SSD and DTW ' + feature + ' over number of pairs')
        plt.legend()
        plt.savefig('D:/project/backtest_result/plot/' + 'compare' + '_aggre_result_' + feature + '.png')
        plt.show()
        
        return 0
    
    def plot_all_compare_result(self):
        for feature in self.features_list:
            self.plot_compare_result(feature)
            
        return 0
        
        
    
        
            
########################################################################
def combination(list):
    output = []
    for i in range(len(list)-1):
        for j in range(len(list)-i-1):
            output.append([list[i], list[i+j+1]])
    return output