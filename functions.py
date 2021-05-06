
"""
# -- --------------------------------------------------------------------------------------------------- -- #
# -- project: A SHORT DESCRIPTION OF THE PROJECT                                                         -- #
# -- script: functions.py : python script with general functions                                         -- #
# -- author: YOUR GITHUB USER NAME                                                                       -- #
# -- license: GPL-3.0 License                                                                            -- #
# -- repository: YOUR REPOSITORY URL                                                                     -- #
# -- --------------------------------------------------------------------------------------------------- -- #
"""

# basic
import pandas as pd
import numpy as np
import datetime
from datetime import timedelta

# data amd processing
# import MetaTrader5 as mt5
from sklearn.preprocessing import StandardScaler, RobustScaler, MaxAbsScaler
from gplearn.genetic import SymbolicTransformer
from sklearn.model_selection import train_test_split

# -------------------------------------------------------------------------- MT5: INITIALIZATION / LOGIN -- #
# ------------------------------------------------------------------------------------------------------ -- #

def f_init_login(param_acc, param_pass, param_exe):
    """
    Initialize conexion and Login into a Meta Trader 5 account in the local computer where this code is executed, using the MetaTrader5 python package.

    Parameters
    ----------

    param_acc: int
        accout number used to login into MetaTrader5 Web/Desktop App (normally is a 8-9 digit integer number)
        param_acc = 41668916
    param_pass: str
        accout trader's password (or just password) to login into MetaTrader5 Web/Desktop App 
        (normally alphanumeric include uppercase and sometimes symbols). If the investor's password 
        is provided, the some actions do not work like open trades.
        param_pass = "n2eunlnt"
    
    param_direxe: str
        Route in disk where is the executable file of the MetaTrader5 desktop app which will be used 
        param_direxe = 'C:\\Program Files\\MetaTrader 5\\terminal64.exe'
    
    Return
    ------

        if connection is succesful then returns connected client object and prints message,
        if connection is not succesful then returns error message and attempts a shutdown of connection.
    
    References
    ----------
    https://www.mql5.com/en/docs/integration/python_metatrader5/mt5login_py
    
    
    """
    
    # server name (as it is specified in the terminal)
    mt5_ser = "MetaQuotes-Demo"
    # timeout (in miliseconds)
    mt5_tmo = 10000

    # Perform initialization handshake
    ini_message = mt5.initialize(param_exe, login=param_acc, password=param_pass, server=mt5_ser,
                                 timeout=mt5_tmo, portable=False)

    # resulting message
    if not ini_message:
        print(" **** init_login failed, error code =", mt5.last_error())
        mt5.shutdown()
    else: 
        print(" ++++ init_login succeded, message = ", ini_message)
    
    # returns an instance of a connection object (or client)
    return mt5


# ------------------------------------------------------------------------------------ MT5: ACCOUNT INFO -- #
# ------------------------------------------------------------------------------------------------------ -- #

def f_acc_info(param_ct):
    """
    Get the info of the account associated with the initialized client param_ct

    Params
    ------

    param_ct: MetaTrader5 initialized client object
        this is an already succesfully initialized conexion object to MetaTrader5 Desktop App

    Returns
    -------

    df_acc_info: pd.DataFrame 
        Pandas DataFrame with the account info         

    References
    ----------
    https://www.mql5.com/en/docs/integration/python_metatrader5/mt5login_py

    """

    # get the account info and transform it into a dataframe format
    acc_info = param_ct.account_info()._asdict()
    
    # select especific info to display
    df_acc_info = pd.DataFrame(list(acc_info.items()), columns=['property','value'])

    # return dataframe with the account info
    return df_acc_info


# ------------------------------------------------------------------------------- MT5: HISTORICAL ORDERS -- #
# ------------------------------------------------------------------------------------------------------ -- #

def f_hist_trades(param_ct, param_ini, param_end):
    """
    Get the historical executed trades in the account associated with the initialized MetaTrader5 client

    Params
    ------

    param_ct: MetaTrader5 initialized client object
        This is an already succesfully initialized conexion object to MetaTrader5 Desktop App
    
    param_ini: datetime
        Initial date to draw the historical trades
        
        param_ini = datetime(2021, 2, 1)
    param_end: datetime
        Final date to draw the historical trades
        
        param_end = datetime(2021, 3, 1)

    Returns
    -------
        df_hist_trades: pd.DataFrame

    References
    ----------
    https://www.mql5.com/en/docs/integration/python_metatrader5/mt5historydealsget_py
    https://www.mql5.com/en/docs/constants/tradingconstants/dealproperties  
    
    https://www.mql5.com/en/docs/integration/python_metatrader5/mt5historyordersget_py
    https://www.mql5.com/en/docs/constants/tradingconstants/orderproperties#enum_order_property_integer

    """

    # get historical info of deals in the account
    history_deals = param_ct.history_deals_get(param_ini, param_end)

    # get historical info of orders in the account
    history_orders = param_ct.history_orders_get(param_ini, param_end)

    # check for returned results
    if (len(history_orders) > 0) & (len(history_deals) > 0):
        print(" ++++ Historical orders retrive: OK")
        print(" ++++ Historical deals retrive: OK")
    else:
        print("No orders and/or deals returned")

    # historical deals of the account
    df_deals = pd.DataFrame(list(history_deals), columns=history_deals[0]._asdict().keys())
    
    # historical orders of the account
    df_orders = pd.DataFrame(list(history_orders), columns=history_orders[0]._asdict().keys())
   
    # useful columns from orders
    df_hist_trades = df_orders[['time_setup', 'symbol', 'position_id', 'type', 'volume_current',
                           'price_open', 'sl', 'tp']]

    # rename columns 
    df_hist_trades.columns = ['OpenTime', 'Symbol', 'Ticket', 'Type', 'Volume', 'OpenPrice', 'S/L', 'T/P']

    # choose only buy or sell transactions (ignore all the rest, like balance ...)
    df_hist_trades = df_hist_trades[(df_hist_trades['Type'] == 0) | (df_hist_trades['Type'] == 1)]
    df_hist_trades['OpenTime'] = pd.to_datetime(df_hist_trades['OpenTime'], unit='s')
    
    # unique values for position_id
    uni_id = df_hist_trades['Ticket'].unique()
    
    # first and last index for every unique value of position_id
    ind_opens = [df_hist_trades.index[df_hist_trades['Ticket'] == i][0] for i in uni_id]
    ind_closes = [df_hist_trades.index[df_hist_trades['Ticket'] == i][-1] for i in uni_id]
    
    # generate lists with values to add
    cts = df_hist_trades['OpenTime'].loc[ind_closes]
    cps = df_hist_trades['OpenPrice'].loc[ind_closes]

    # resize dataframe to have only the first value of every unique position_id
    df_hist_trades = df_hist_trades.loc[ind_opens]

    # add close time and close price as a column to dataframe
    df_hist_trades['CloseTime'] = cts.to_list()
    df_hist_trades['ClosePrice'] = cps.to_list()
    df_hist_trades['Profit'] = df_deals['profit'].loc[df_deals['position_id'].isin(uni_id) & 
                                                      df_deals['entry'] == 1].to_list()
  
    return df_hist_trades

# ------------------------------------------------------------------------------- MT5: HISTORICAL PRICES -- #
# ------------------------------------------------------------------------------------------------------ -- #

def f_hist_prices(param_ct, param_sym, param_tf, param_ini, param_end):
    """
    Historical prices retrival from MetaTrader 5 Desktop App.

    Parameters
    ----------

    param_ct: MetaTrader5 initialized client object
        This is an already succesfully initialized conexion object to MetaTrader5 Desktop App
      
    param_sym: str
        The symbol of which the historical prices will be retrieved
        
        param_sym = 'EURUSD'
    
    param_tf: str
        The price granularity for the historical prices. Check available timeframes and nomenclatures from 
        the references. The substring 'TIMEFRAME_' is automatically added.
        
        param_tf = 'M1'
    param_ini: datetime
        Initial date to draw the historical trades
        
        param_ini = datetime(2021, 2, 1)
    param_end: datetime
        Final date to draw the historical trades
        
        param_end = datetime(2021, 3, 1)
    
    **** WARNINGS ****
    
    1.- Available History
    
        MetaTrader 5 terminal provides bars only within a history available to a user on charts. The number of # bars available to users is set in the "Max.bars in chart" parameter. So this must be done
        manually within the desktop app to which the connection is made.
    
    2.- TimeZone
        When creating the 'datetime' object, Python uses the local time zone, 
        MetaTrader 5 stores tick and bar open time in UTC time zone (without the shift).
        Data received from the MetaTrader 5 terminal has UTC time.
        Perform a validation whether if its necessary to shift time to local timezone.

    **** ******** ****

    References
    ----------
    https://www.mql5.com/en/docs/integration/python_metatrader5/mt5copyratesfrom_py#timeframe

    """


    # get hour info in UTC timezone (also GMT+0)
    hour_utc = datetime.datetime.now().utcnow().hour
    # get hour info in local timezone (your computer)
    hour_here = datetime.datetime.now().hour
    # difference (in hours) from UTC timezone
    diff_here_utc = hour_utc - hour_here
    # store the difference in hours
    tdelta = datetime.timedelta(hours=diff_here_utc)
    # granularity 
    param_tf = getattr(param_ct, 'TIMEFRAME_' + param_tf)
    # dictionary for more than 1 symbol to retrieve prices
    d_prices = {}

    # retrieve prices for every symbol in the list
    for symbol in param_sym:
        # prices retrival from MetaTrader 5 Desktop App
        prices = pd.DataFrame(param_ct.copy_rates_range(symbol, param_tf,
                                                         param_ini - tdelta,
                                                         param_end - tdelta))
        # convert to datetime 
        prices['time'] = [datetime.datetime.fromtimestamp(times) for times in prices['time']]

        # store in dict with symbol as a key
        d_prices[symbol] = prices
    
    # return historical prices
    return d_prices


def autoregressive_features(p_data, p_memory):
    """
    Creacion de variables de naturaleza autoregresiva (resagos, promedios, diferencias)
    Parameters
    ----------
    p_data: pd.DataFrame
        with OHLCV columns: Open, High, Low, Close, Volume
    p_memory: int
        A value that represents the implicit assumption of a "memory" effect in the prices
    Returns
    -------
    r_features: pd.DataFrame

    """

    # work with a separate copy of original data
    data = p_data.copy()

    # nth-period final price "movement"
    data['co'] = (data['close'] - data['open'])
    # nth-period uptrend movement
    data['ho'] = (data['high'] - data['open'])
    # nth-period downtrend movement
    data['ol'] = (data['open'] - data['low'])
    # nth-period volatility measure
    data['hl'] = (data['high'] - data['low'])

    # N features with window-based calculations
    for n in range(0, p_memory):
        data['ma_ol'] = data['ol'].rolling(n + 2).mean()
        data['ma_ho'] = data['ho'].rolling(n + 2).mean()
        data['ma_hl'] = data['hl'].rolling(n + 2).mean()

        data['lag_ol_' + str(n + 1)] = data['ol'].shift(n + 1)
        data['lag_ho_' + str(n + 1)] = data['ho'].shift(n + 1)
        data['lag_hl_' + str(n + 1)] = data['hl'].shift(n + 1)

        data['sd_ol_' + str(n + 1)] = data['ol'].rolling(n + 1).std()
        data['sd_ho_' + str(n + 1)] = data['ho'].rolling(n + 1).std()
        data['sd_hl_' + str(n + 1)] = data['hl'].rolling(n + 1).std()

        data['lag_vol_' + str(n + 1)] = data['volume'].shift(n + 1)
        data['sum_vol_' + str(n + 1)] = data['volume'].rolling(n + 1).sum()
        data['mean_vol_' + str(n + 1)] = data['volume'].rolling(n + 1).mean()

    # timestamp as index
    data.index = pd.to_datetime(data.index)
    # select columns, drop for NAs, change column types, reset index
    r_features = data.drop(['open', 'high', 'low', 'close', 'hl', 'ol', 'ho', 'volume'], axis=1)
    r_features = r_features.dropna(axis='columns', how='all')
    r_features = r_features.dropna(axis='rows')
    r_features.iloc[:, 1:] = r_features.iloc[:, 1:].astype(float)
    r_features.reset_index(inplace=True, drop=True)

    return r_features


# ---------------------------------------------------------- FUNCTION: Autoregressive Feature Engieering -- #
# ---------------------------------------------------------- ---------------------------------------------- #

def linear_features(p_data, p_memory, p_target):
    """
    autoregressive process for feature engineering
    Parameters
    ----------
    p_data: pd.DataFrame
        con datos completos para ajustar modelos
        p_data = m_folds['periodo_1']
    p_memory: int
        valor de memoria maxima para hacer calculo de variables autoregresivas
        p_memory = 7
    Returns
    -------
    model_data: dict
        {'train_x': pd.DataFrame, 'train_y': pd.DataFrame, 'val_x': pd.DataFrame, 'val_y': pd.DataFrame}
    References
    ----------
    """

    # hardcopy of data
    data = p_data.copy()

    # funcion para generar variables autoregresivas
    data_ar = autoregressive_features(p_data=data, p_memory=p_memory)

    # y_t = y_t+1 in order to prevent filtration, that is, at time t, the target variable y_t
    # with the label {co_d}_t will be representing the direction of the price movement (0: down, 1: high)
    # that was observed at time t+1, and so on applies to t [0, n-1]. the last value is droped
    data_ar[p_target] = data_ar[p_target].shift(-1, fill_value=999)
    data_ar = data_ar.drop(data_ar[p_target].index[[-1]])

    # separacion de variable dependiente
    data_y = data_ar[p_target].copy()

    # separacion de variables independientes
    data_arf = data_ar.drop(['timestamp', p_target], axis=1, inplace=False)

    # datos para utilizar en la siguiente etapa
    next_data = pd.concat([data_y.copy(), data_arf.copy()], axis=1)

    # keep the timestamp as index
    next_data.index = data_ar['timestamp'].copy()

    return next_data


def data_scaler(p_data, p_trans):
    """
    Estandarizar (a cada dato se le resta la media y se divide entre la desviacion estandar) se aplica a
    todas excepto la primera columna del dataframe que se use a la entrada
    Parameters
    ----------
    p_trans: str
        Standard: Para estandarizacion (restar media y dividir entre desviacion estandar)
        Robust: Para estandarizacion robusta (restar mediana y dividir entre rango intercuartilico)
    p_datos: pd.DataFrame
        Con datos numericos de entrada
    Returns
    -------
    p_datos: pd.DataFrame
        Con los datos originales estandarizados
    """

    # hardcopy of the data
    data = p_data.copy()
    # list with columns to transform
    lista = data[list(data.columns)]
    # choose to scale from 1 in case timestamp is present
    scale_ind = 1 if 'timestamp' in list(data.columns) else 0

    if p_trans == 'standard':

        # removes the mean and scales the data to unit variance
        data[list(data.columns[scale_ind:])] = StandardScaler().fit_transform(lista.iloc[:, scale_ind:])
        return data

    elif p_trans == 'robust':

        # removes the meadian and scales the data to inter-quantile range
        data[list(data.columns[scale_ind:])] = RobustScaler().fit_transform(lista.iloc[:, scale_ind:])
        return data

    elif p_trans == 'scale':

        # scales to max value
        data[list(data.columns[scale_ind:])] = MaxAbsScaler().fit_transform(lista.iloc[:, scale_ind:])
        return data

    else:
        print('Error in data_scaler, p_trans value is not valid')


def symbolic_features(p_x, p_y, p_params):
    """
    Feature engineering process with symbolic variables by using genetic programming.
    Parameters
    ----------
    p_x: pd.DataFrame / np.array / list
        with regressors or predictor variables
        p_x = data_features.iloc[:, 1:]
    p_y: pd.DataFrame / np.array / list
        with variable to predict
        p_y = data_features.iloc[:, 0]
    p_params: dict
        with parameters for the genetic programming function
        p_params = {'functions': ["sub", "add", 'inv', 'mul', 'div', 'abs', 'log'],
        'population': 5000, 'tournament':20, 'hof': 20, 'generations': 5, 'n_features':20,
        'init_depth': (4,8), 'init_method': 'half and half', 'parsimony': 0.1, 'constants': None,
        'metric': 'pearson', 'metric_goal': 0.65,
        'prob_cross': 0.4, 'prob_mutation_subtree': 0.3,
        'prob_mutation_hoist': 0.1. 'prob_mutation_point': 0.2,
        'verbose': True, 'random_cv': None, 'parallelization': True, 'warm_start': True }
    Returns
    -------
    results: dict
        With response information
        {'fit': model fitted, 'params': model parameters, 'model': model,
         'data': generated data with variables, 'best_programs': models best programs}
    References
    ----------
    https://gplearn.readthedocs.io/en/stable/reference.html#gplearn.genetic.SymbolicTransformer


    **** NOTE ****
    simplified internal calculation for correlation (asuming w=1)

    y_pred_demean = y_pred - np.average(y_pred)
    y_demean = y - np.average(y)
                              np.sum(y_pred_demean * y_demean)
    pearson =  ---------------------------------------------------------------
                np.sqrt((np.sum(y_pred_demean ** 2) * np.sum(y_demean ** 2)))
    """

    # Function to produce Symbolic Features
    model = SymbolicTransformer(function_set=p_params['functions'], population_size=p_params['population'],
                                tournament_size=p_params['tournament'], hall_of_fame=p_params['hof'],
                                generations=p_params['generations'], n_components=p_params['n_features'],

                                init_depth=p_params['init_depth'], init_method=p_params['init_method'],
                                parsimony_coefficient=p_params['parsimony'],
                                const_range=p_params['constants'],

                                metric=p_params['metric'], stopping_criteria=p_params['metric_goal'],

                                p_crossover=p_params['prob_cross'],
                                p_subtree_mutation=p_params['prob_mutation_subtree'],
                                p_hoist_mutation=p_params['prob_mutation_hoist'],
                                p_point_mutation=p_params['prob_mutation_point'],
                                max_samples=p_params['max_samples'],

                                verbose=p_params['verbose'], warm_start=p_params['warm_start'],
                                random_state=123, n_jobs=-1 if p_params['parallelization'] else 1,
                                feature_names=p_x.columns)

    # SymbolicTransformer fit
    model_fit = model.fit_transform(p_x, p_y)

    # output data of the model
    data = pd.DataFrame(model_fit)

    # parameters of the model
    model_params = model.get_params()

    # best programs dataframe
    best_programs = {}
    for p in model._best_programs:
        factor_name = 'sym' + str(model._best_programs.index(p))
        best_programs[factor_name] = {'raw_fitness': p.raw_fitness_, 'reg_fitness': p.fitness_,
                                      'expression': str(p), 'depth': p.depth_, 'length': p.length_}

    # format and sorting
    best_programs = pd.DataFrame(best_programs).T
    best_programs = best_programs.sort_values(by='raw_fitness', ascending=False)

    # results
    results = {'fit': model_fit, 'params': model_params, 'model': model, 'data': data,
               'best_programs': best_programs, 'details': model.run_details_}

    return results


def genetic_programed_features(p_data, p_target, p_params):
    """
    El uso de programacion genetica para generar variables independientes simbolicas
    Parameters
    ----------
    p_data: pd.DataFrame
        con datos completos para ajustar modelos

        p_data = m_folds['periodo_1']
    p_split: int
        split in val
        p_split = '0'
    p_params:
        parameters for symbolic_features process
    Returns
    -------
    model_data: dict
        {'train_x': pd.DataFrame, 'train_y': pd.DataFrame, 'val_x': pd.DataFrame, 'val_y': pd.DataFrame}
    References
    ----------
    https://stackoverflow.com/questions/3819977/
    what-are-the-differences-between-genetic-algorithms-and-genetic-programming
    """

    # separacion de variable dependiente
    datos_y = p_data[p_target].copy().astype(int)

    # separacion de variables independientes
    datos_had = p_data.copy().drop([p_target], axis=1, inplace=False)

    # Lista de operaciones simbolicas
    sym_data = symbolic_features(p_x=datos_had, p_y=datos_y, p_params=p_params)

    # Symbolic variables output
    datos_sym = sym_data['data'].copy()
    datos_sym.columns = ['sym_' + str(i) for i in range(0, len(sym_data['data'].iloc[0, :]))]
    datos_sym.index = datos_y.index

    return {'sym_data': sym_data, 'sym_features': datos_sym}


def data_split(p_data, p_target, p_split):
    # separacion de variable dependiente
    datos_y = p_data[p_target].copy().astype(float)

    # if size != 0 then an inner fold division is performed with size*100 % as val and the rest for train
    size = float(p_split) / 100

    # automatic data sub-sets division according to inner-split
    xtrain, xval, ytrain, yval = train_test_split(p_data, datos_y, test_size=size, shuffle=False)

    return {'train_x': xtrain, 'train_y': ytrain, 'val_x': xval, 'val_y': yval}


def backtest(df: pd.DataFrame, x_train: pd.DataFrame, y_hat, trading_volume=1, capital=100000):
    """
    df: pd.DataFrame with historic OHLCV used to train the model and generate signals
    x_train: pd.DataFrame with data used to train the model. The index must be set to timestamp, which will be used to
        match prices and operations
    y_hat: List or array with the ML model's prediction. Should be a regression prediction, internally will generate
        buy/sellsignals. If next movement is predicted to be positive, signal is buy, otherwise is sell
    trading_volume: int or float, size of contract or position that will be used to trade. Constant.
        Example: trade 1 contract of EURUSD, then trading_volume is 100000 since 1 contract has a 100000 EUR size.
        BTCUSD contract size is 1.
    capital: account funds. Used to initialize the account trading history.
    """
    signal = [1 if i > 0 else 0 for i in y_hat]
    operations = pd.DataFrame(columns=['date', 'signal', 'open', 'close'])
    operations['date'] = x_train.index
    operations['signal'] = signal
    df_operations = df.copy()
    df_operations = df_operations.set_index('timestamp')
    df_operations = df_operations.shift(-1)
    operations['open'] = df_operations['open'][x_train.index].values
    operations['close'] = df_operations['close'][x_train.index].values
    profit = []
    for i in range(len(operations)):
        if operations['signal'][i] == 1:
            profit.append((operations['close'][i] - operations['open'][i]) * trading_volume)
        else:
            profit.append((operations['open'][i] - operations['close'][i]) * trading_volume)
    operations['profit'] = profit
    operations = operations.set_index('date')
    operations['cum_profit'] = operations['profit'].cumsum() + capital
    return operations


def f_estadisticas_mad(df_evolucion_capital: pd.DataFrame):
    # calcular rendimiento del capital
    rp = np.log(df_evolucion_capital['cum_profit'] / \
                df_evolucion_capital['cum_profit'].shift(1))
    # Sharpe Ratio original
    sharpe_original = (rp.mean() - 0.05 / 360 / 24) / rp.std()
    # Calculo de drawdown del capital
    drawdown = []
    drawdown_fechas = []
    for i in range(len(df_evolucion_capital)):
        drawdown.append(df_evolucion_capital['cum_profit'][i] -\
                        df_evolucion_capital['cum_profit'][i:].min())
        drawdown_fechas.append((df_evolucion_capital.index[i],
                                df_evolucion_capital['cum_profit'][i:].idxmin()))
    drawdown_max = np.argmax(drawdown)
    drawdown_capital = {'fecha_inicial': drawdown_fechas[drawdown_max][0],
                        'fecha_final': drawdown_fechas[drawdown_max][1],
                        'Drawdown_capital': drawdown[drawdown_max]}
    # Calculo de drawup del capital
    drawup = []
    drawup_fechas = []
    for i in range(1,len(df_evolucion_capital)):
        drawup.append(df_evolucion_capital['cum_profit'][i] -\
                      df_evolucion_capital['cum_profit'][:i].min())
        drawup_fechas.append((df_evolucion_capital.index[i],
                              df_evolucion_capital['cum_profit'][:i].idxmin()))
    drawup_max = np.argmax(drawup)
    drawup_capital = {'fecha_inicial': drawup_fechas[drawup_max][1],
                      'fecha_final': drawup_fechas[drawup_max][0],
                      'Drawup_capital': drawup[drawup_max]}
    # Creacion de DataFrame con metricas de atribucion al desempeÃ±o
    df_estadisticas_mad = pd.DataFrame()
    df_estadisticas_mad['metrica'] = ['sharpe_original',
                                      'drawdown_capi',
                                      'drawdown_capi',
                                      'drawdown_capi',
                                      'drawup_capi',
                                      'drawup_capi',
                                      'drawup_capi']
    df_estadisticas_mad['tipo'] = ['Cantidad',
                                   'Fecha Inicial',
                                   'Fecha Final',
                                   'DrawDown $ (capital)',
                                   'Fecha Inicial',
                                   'Fecha Final',
                                   'DrawUP $ (capital)']
    df_estadisticas_mad['valor'] = [sharpe_original,
                                    drawdown_capital['fecha_inicial'],
                                    drawdown_capital['fecha_final'],
                                    drawdown_capital['Drawdown_capital'],
                                    drawup_capital['fecha_inicial'],
                                    drawup_capital['fecha_final'],
                                    drawup_capital['Drawup_capital']]
    df_estadisticas_mad['descripcion'] = ['Sharpe Ratio FÃ³rmula Original',
                                          'Fecha inicial del DrawDown de Capital',
                                          'Fecha final del DrawDown de Capital',
                                          'MÃ¡xima pÃ©rdida flotante registrada',
                                          'Fecha inicial del DrawUp de Capital',
                                          'Fecha final del DrawUp de Capital',
                                          'MÃ¡xima ganancia flotante registrada']
    return df_estadisticas_mad


def f_estadisticas_ba(param_data):
    Profit = param_data["profit"]
    largo = len(param_data)
    datos = param_data
    tabla = pd.DataFrame(columns=["medida", "valor", "descripcion"])
    tabla.loc[0] = "Ops totales", largo, "Operaciones totales"
    tabla.loc[1] = "Ganadoras", sum([1 for i in range(largo) if Profit[i] > 0]), "Operaciones ganadoras"
    tabla.loc[2] = "Ganadoras_c", sum(
        [1 for i in range(largo) if Profit[i] > 0 and datos['signal'][i] == 1]), "Operaciones ganadoras de compra"
    tabla.loc[3] = "Ganadoras_v", sum(
        [1 for i in range(largo) if Profit[i] > 0 and datos['signal'][i] == 0]), "Operaciones ganadoras de venta"
    tabla.loc[4] = "Perdedoras", sum([1 for i in range(largo) if Profit[i] < 0]), "Operaciones perdedoras"
    tabla.loc[5] = "Perdedoras_c", sum(
        [1 for i in range(largo) if Profit[i] < 0 and datos['signal'][i] == 1]), "Operaciones perdedoras de compra"
    tabla.loc[6] = "Perdedoras_v", sum(
        [1 for i in range(largo) if Profit[i] < 0 and datos['signal'][i] == 0]), "Operaciones perdedoras de venta"
    tabla.loc[7] = "Mediana (Profit)", Profit.median(), "Mediana de profit de operaciones"
    tabla.loc[9] = "r_efectividad", tabla.iloc[1, 1] / tabla.iloc[0, 1], "Ganadoras Totales/Operaciones Totales"
    tabla.loc[10] = "r_proporcion", tabla.iloc[1, 1] / tabla.iloc[4, 1], "Ganadoras Totales/Perdedoras Totales"
    tabla.loc[11] = "r_efectividad_c", tabla.iloc[2, 1] / tabla.iloc[0, 1], "Ganadoras Compras/Operaciones Totales"
    tabla.loc[12] = "r_efectividad_v", tabla.iloc[3, 1] / tabla.iloc[0, 1], "Ganadoras Ventas/Operaciones Totales"
    return tabla
