import pandas as pd
import pathlib


usecols_list = [
    'AGE',
    'AMONTH',
    'DIED',
    'DISPUNIFORM',
    'DRG24',
    'FEMALE',
    'HCUP_ED',
    'LOS',
    'MDC24',
    'NCHRONIC',
    'NDX',
    'NECODE',
    'NEOMAT',
    'NPR',
    'ORPROC',
    'RACE',
    'TOTCHG',
    'TRAN_IN',
    'TRAN_OUT',
    'YEAR',
    'ZIPINC_QRTL',
    ]
    
dtype_dict = {
    'AGE': 'Int16',
    'AMONTH': 'Int16',
    'DIED': 'Int16',
    'DISPUNIFORM': 'Int16',
    'DRG24': 'Int16',
    'FEMALE': 'Int16',
    'HCUP_ED': 'Int16',
    'LOS': 'Int16',
    'MDC24': 'Int16',
    'NCHRONIC': 'Int16',
    'NDX': 'Int16',
    'NECODE': 'Int16',
    'NEOMAT': 'Int16',
    'NPR': 'Int16',
    'ORPROC': 'Int16',
    'RACE': 'Int16',
    'TOTCHG': 'Int32',
    'TRAN_IN': 'Int16',
    'TRAN_OUT': 'Int16',
    'YEAR': 'Int16',
    'ZIPINC_QRTL': 'Int16',
    }


def get_not_train_vars():
    """
    Get variables not involved in training
    
    Returns
    -------
    list
       List of svariables not involved in training 
    """

    return ['RACE', 'ZIPINC_QRTL', 'FEMALE', 'YEAR', 'TOTCHG']


def load_datafile(file_path, n=None, frac=None, seed=None):
    """
    Load data from a given Path (object) and return DataFrame object
    
    Parameters
    ----------
    file_path: PurePath
        Path of the comma-separated values (csv) file
    n: int
        Number of items from axis to return
    frac: folat
        Fraction of axis items to return
    
    Returns
    -------
    DataFrame
       DataFram from the comma-separated values (csv) file
    """

    if not isinstance(file_path, pathlib.PurePath):
        try:
            pathlib.Path(file_path)
        except:
            raise TypeError('Input must be a pathlib Path object, found {}'
                            .format(type(file_path)))
    if not file_path.exists():
        raise FileExistsError('Path does not exist')

    data_concat = pd.DataFrame()
    data_chunk = pd.read_csv(
        file_path,
        usecols=usecols_list,
        dtype=dtype_dict,
        na_values=['A', 'B', 'C'],
        iterator=True,
        chunksize=100000,
        )
    for chunk in data_chunk:
        data_concat = pd.concat([data_concat, chunk])
    if not ((n is None) & (frac is None)):
        data_concat.sample(n=n, frac=frac, random_state=seed)

    return data_concat


def peprocessing(raw_data, statistics=False):
    """
    Peprocessing the data and return a DataFrame
    
    Parameters
    ----------
    raw_data: DataFrame
        DataFrame that need to be preprocessed
    
    Returns
    -------
    DataFrame
       preprocessed DataFrame 
    """

    if not isinstance(raw_data, pd.DataFrame):
        raise TypeError('Input must be a DataFrame, found {}'
                        .format(type(raw_data)))
    
    preprocessed_data = raw_data.dropna()

    if statistics:
        with pd.option_context('display.max_rows', None):
            print (preprocessed_data.describe(percentiles=[]).T)
    onehot_list = ['DIED', 'DISPUNIFORM', 'DRG24', 'HCUP_ED', 'MDC24',
                   'NEOMAT', 'ORPROC', 'TRAN_IN', 'TRAN_OUT']

    preprocessed_data = pd.get_dummies(preprocessed_data,
                                   columns=onehot_list).astype('int')

    return preprocessed_data


def trans_percentile(percentile):
    """
    Convert percentage numbers to indexes in statistics
    
    Parameters
    ----------
    percentile: float or list
        A value or a list of values as thresholds
    
    Returns
    -------
    list
       List of threshold indices
    """

    if not isinstance(percentile, float):
        try:
            percentile = float(percentile)
        except:
            raise TypeError('Input must be a float, found {}'
                            .format(type(percentile)))
    percentile = str(round(percentile * 100, 1)).rstrip('0')
    if percentile[-1] == '.':
        percentile = percentile.rstrip('.')
    return percentile + '%'


class Objective:

    def __init__(self, train_data, test_data=None,
                 train_size=None, seed=None, protected='race',
                 percentiles_list=[.6, .75, .9]):
        """
        Build what objective is expected to be done
    
        Parameters
        ----------
        train_data: DataFrame
            Train data
        test_data: DataFrame
            Test data
        train_size: float
            The proportion of the dataset to include in the test split
        seed: int
            A number used to initialise a pseudorandom number generator
        protected: 'race' or 'income'
            Protected attribute
        percentiles_list: float or list
            A value or a list of values as thresholds
        """

        if not isinstance(train_data, pd.DataFrame):
            raise TypeError('Input must be a DataFrame, found {}'
                            .format(type(train_data)))
        if test_data is None:
            if train_size is None:
                train_size = 0.25
            else:
                if not isinstance(train_size, float):
                    try:
                        train_size = float(train_size)
                    except:
                        raise TypeError('Input must be a float, found {}'
                                        .format(type(train_size)))
                if not (train_size > 0) & (train_size < 1):
                    raise ValueError(
                        'Train size should be a float between 0 and 1')
            from sklearn.model_selection import train_test_split
            (self.train_data, self.test_data) = \
                train_test_split(train_data, test_size=train_size,
                                 random_state=seed)
        else:
            if not isinstance(test_data, pd.DataFrame):
                raise TypeError('Input must be a DataFrame, found {}'
                                .format(type(test_data)))
            if train_data.shape[1] != test_data.shape[1]:
                raise ValueError('Train data and Validation data' +
                                 ' should be same number of features')
            self.train_data = train_data
            self.test_data = test_data
        if protected not in ['race', 'income']:
            raise ValueError("Protected should be 'race' or 'income', found {}"
                             .format(protected))
        if not isinstance(percentiles_list, list):
            try:
                percentiles_list = list([percentiles_list])
            except:
                raise TypeError('Input must be a list, found {}'
                                .format(type(percentiles_list)))
        if not all((percentile > 0) & (percentile < 1)
                   for percentile in percentiles_list):
            raise ValueError(
                'All of the percentiles should be between 0 and 1')

        if protected == 'race':
            self.protected = 'RACE'
        if protected == 'income':
            self.protected = 'ZIPINC_QRTL'
        self.percentiles_list = percentiles_list
        self.percentiles_str_list = \
            [trans_percentile(_) for _ in percentiles_list]

        self.threshold_train = list()
        statistics_train = train_data['TOTCHG']\
            .describe(percentiles=percentiles_list)
        for _ in self.percentiles_str_list:
            self.threshold_train.append(statistics_train[_])

        self.threshold_test = list()
        statistics_test = train_data['TOTCHG']\
            .describe(percentiles=percentiles_list)
        for _ in self.percentiles_str_list:
            self.threshold_test.append(statistics_test[_])


def metric_plot(AUPRC, DI, label_list):
    """
    Returns visualizations based on metrics
    
    Parameters
    ----------
    AUPRC: dict
        Dictionary of AUPRC
    DI: dict
        Dictionary of disparate impact
    label_list: list
        List of label
    """

    import matplotlib.pyplot as plt
    plt.figure(figsize=(8, 6), dpi=100)
    for i in range(len(AUPRC)):
        plt.scatter(AUPRC[i], DI[i], label=label_list[i])

    plt.axhline(y=1, color='black', linestyle='--')
    plt.axvline(x=1, color='black', linestyle='--')
    plt.xlim(0.3, 1.01)
    plt.ylim(0.8, 1.007)

    plt.legend(loc='upper right')
    plt.yscale('log')
    plt.show()
