from typing import TypedDict

class Data_cfg(TypedDict):
    dataset : str
    usage_perc : float
    nof_days_to_pred : int
    split_train_val : list[float]
    loader : object
    label_column : str

class Training_cfg(TypedDict):
    nof_ens : int
    model_type : str
    dataset : str
    retrain_period : int

class Lstm_cfg(TypedDict):
    units : int
    n_layers : int
    quantiles : list[float]
    l2_lambda : float
    learning_rate : float
    batch_size : int           
    time_steps_in : int
    time_steps_out : int         
    epochs : int
    patience : int
    start_from_epoch : int
    alpha : float
    loss_func : str

class LstmConv_cfg(TypedDict):
    units : int
    n_filters : int
    kernel_size : int
    n_layers : int
    quantiles : list[float]
    l2_lambda : float
    learning_rate : float
    batch_size : int           
    time_steps_in : int
    time_steps_out : int         
    epochs : int
    patience : int
    start_from_epoch : int
    alpha : float
    loss_func : str

class Tcn_cfg(TypedDict):
    dilations : list[int]
    n_filters : int
    kernel_size : int
    quantiles : list[float]
    l2_lambda : float
    learning_rate : float
    batch_size : int           
    time_steps_in : int
    time_steps_out : int     
    epochs : int
    patience : int
    start_from_epoch : int
    alpha : float
    loss_func : str

class Mlp_cfg(TypedDict):
    units : int
    n_layers : int
    quantiles : list[float]
    l2_lambda : float
    learning_rate : float
    batch_size : int           
    time_steps_in : int
    time_steps_out : int      
    epochs : int
    patience : int
    start_from_epoch : int
    alpha : float
    loss_func : str

class Plot_cfg(TypedDict):
    plot_training : bool

class Cfg(TypedDict):
    data_cfg : Data_cfg
    training_cfg : Training_cfg
    lstm_cfg : Lstm_cfg
    tcn_cfg : Tcn_cfg
    mlp_cfg : Mlp_cfg
    plot_cfg : Plot_cfg