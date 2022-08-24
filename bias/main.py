try:
    from utils import *
    from algorithms import *
except ImportError:
    from .utils import *
    from .algorithms import *
import pathlib
import pandas as pd
import numpy as np


def main():
    # Load data and preprocessing
    file_path = pathlib.Path.cwd().parent/'HCUP'/'NIS_2012_Core.csv'
    HCUP_data_2012 = load_datafile(pathlib.Path(file_path), n=1000000, seed=42)
    file_path = pathlib.Path.cwd().parent/'HCUP'/'NIS_2014_Core.csv'
    HCUP_data_2014 = load_datafile(pathlib.Path(file_path), n=1000000, seed=42)
    processed_data = peprocessing(pd.concat([HCUP_data_2012, HCUP_data_2014]),
                                  statistics=True)
    train_data = processed_data[processed_data['YEAR'] == 2012]
    test_data = processed_data[processed_data['YEAR'] == 2014]
    train_data = processed_data[processed_data['YEAR'] == 2012]
    test_data = processed_data[processed_data['YEAR'] == 2014]

    for protected in ['race', 'income']:
        # Set the objective
        objective = Objective(train_data,
                              test_data=test_data,
                              protected=protected)

        # Frequency
        print(train_data[objective.protected].value_counts(dropna=False))
        print(test_data[objective.protected].value_counts(dropna=False))

        # Initialisation metrics
        metric_AUPRC = dict()
        metric_DI = dict()
        DI_table = dict()
        for i in range(3):
            metric_AUPRC[objective.percentiles_str_list[i]] = list()
            metric_DI[objective.percentiles_str_list[i]] = list()
            DI_table[objective.percentiles_str_list[i]+'_list'] = list()
        
        
        # Train all models and save the metrics
        model = Linear(objective)
        for i in range(3):
            metric_AUPRC[objective.percentiles_str_list[i]].\
                append(model.AUPRC[objective.percentiles_str_list[i]])
            metric_DI[objective.percentiles_str_list[i]].\
                append(model.DI[objective.percentiles_str_list[i]])
            DI_table[objective.percentiles_str_list[i]+'_list'].\
                append(model.DI[objective.percentiles_str_list[i]+'_list'])
        print(model.time)

        model = Logistic(objective)
        for i in range(3):
            metric_AUPRC[objective.percentiles_str_list[i]].\
                append(model.AUPRC[objective.percentiles_str_list[i]])
            metric_DI[objective.percentiles_str_list[i]].\
                append(model.DI[objective.percentiles_str_list[i]])
            DI_table[objective.percentiles_str_list[i]+'_list'].\
                append(model.DI[objective.percentiles_str_list[i]+'_list'])
        print(model.time)

        model = Lasso(objective, seed=42)
        for i in range(3):
            metric_AUPRC[objective.percentiles_str_list[i]].\
                append(model.AUPRC[objective.percentiles_str_list[i]])
            metric_DI[objective.percentiles_str_list[i]].\
                append(model.DI[objective.percentiles_str_list[i]])
            DI_table[objective.percentiles_str_list[i]+'_list'].\
                append(model.DI[objective.percentiles_str_list[i]+'_list'])
        print(model.time)

        model = ElasticNet(objective, seed=42)
        for i in range(3):
            metric_AUPRC[objective.percentiles_str_list[i]].\
                append(model.AUPRC[objective.percentiles_str_list[i]])
            metric_DI[objective.percentiles_str_list[i]].\
                append(model.DI[objective.percentiles_str_list[i]])
            DI_table[objective.percentiles_str_list[i]+'_list'].\
                append(model.DI[objective.percentiles_str_list[i]+'_list'])

        model = SVM(objective, seed=42)
        for i in range(3):
            metric_AUPRC[objective.percentiles_str_list[i]].\
                append(model.AUPRC[objective.percentiles_str_list[i]])
            metric_DI[objective.percentiles_str_list[i]].\
                append(model.DI[objective.percentiles_str_list[i]])
            DI_table[objective.percentiles_str_list[i]+'_list'].\
                append(model.DI[objective.percentiles_str_list[i]+'_list'])
        print(model.time)

        model = Xgboost(objective, seed=42)
        for i in range(3):
            metric_AUPRC[objective.percentiles_str_list[i]].\
                append(model.AUPRC[objective.percentiles_str_list[i]])
            metric_DI[objective.percentiles_str_list[i]].\
                append(model.DI[objective.percentiles_str_list[i]])
            DI_table[objective.percentiles_str_list[i]+'_list'].\
                append(model.DI[objective.percentiles_str_list[i]+'_list'])
        print(model.time)

        model = NN(objective, seed=42)
        for i in range(3):
            metric_AUPRC[objective.percentiles_str_list[i]].\
                append(model.AUPRC[objective.percentiles_str_list[i]])
            metric_DI[objective.percentiles_str_list[i]].\
                append(model.DI[objective.percentiles_str_list[i]])
            DI_table[objective.percentiles_str_list[i]+'_list'].\
                append(model.DI[objective.percentiles_str_list[i]+'_list'])
        print(model.time)

        #
        print(metric_AUPRC)
        print(metric_DI)
        print(DI_table)

        # Table output 
        label_list = ['OLS', 'RL', 'Lasso', 'Elastic', 'SVM', 'XGB', 'NN']
        print(pd.DataFrame(metric_AUPRC, index=label_list).T)
        print(pd.DataFrame(metric_DI, index=label_list).T)
        for index in DI_table:
            print(pd.DataFrame(DI_table[index], index=label_list,
                               columns=range(1, 
                               len(DI_table[list(DI_table.keys())[0]][0])+1)))

        # Plot output
        for i in range(3):
            metric_plot(metric_AUPRC[objective.percentiles_str_list[i]],
                        metric_DI[objective.percentiles_str_list[i]],
                        label_list=label_list)
    

if __name__ == '__main__':
    main()
