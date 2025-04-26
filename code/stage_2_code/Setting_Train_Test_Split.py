'''
Concrete SettingModule class for a specific experimental SettingModule
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from code.base_class.setting import setting
#from sklearn.model_selection import train_test_split
import numpy as np

class Setting_Train_Test_Split(setting):
    #fold = 3
    
    def load_run_save_evaluate(self):
        # load train dataset
        #train_data_obj = self.dataset.load()
        #train_data_obj.dataset_source_file_name = 'train.csv'
        self.dataset.dataset_source_file_name = 'train.csv'
        loaded_train_data = self.dataset.load()

        # Load testing dataset
        #test_data_obj = self.dataset.load()
        #test_data_obj.dataset_source_file_name = 'test.csv'
        self.dataset.dataset_source_file_name = 'test.csv'
        loaded_test_data = self.dataset.load()

        # run MethodModule
        self.method.data = {'train': {'X': loaded_train_data['X'], 'y': loaded_train_data['y']}, 'test': {'X': loaded_test_data['X'], 'y': loaded_test_data['y']}}
        learned_result = self.method.run()

        # save raw ResultModule
        self.result.data = learned_result
        self.result.save()

        self.evaluate.data = learned_result

        return self.evaluate.evaluate(), None
'''
        # load dataset
        loaded_data = self.dataset.load()

        X_train, X_test, y_train, y_test = train_test_split(loaded_data['X'], loaded_data['y'], test_size = 0.33)

        # run MethodModule
        self.method.data = {'train': {'X': X_train, 'y': y_train}, 'test': {'X': X_test, 'y': y_test}}
        learned_result = self.method.run()
            
        # save raw ResultModule
        self.result.data = learned_result
        self.result.save()
            
        self.evaluate.data = learned_result
        
        return self.evaluate.evaluate(), None
'''