from typedefs_config import Data_cfg, Training_cfg, Lstm_cfg, Mlp_cfg, Tcn_cfg, Plot_cfg, LstmConv_cfg
import utils.data_loaders
import copy

# choose network settings here

training_cfg : Training_cfg = dict(
    retrain_period = 0, # take these amount of new (now available) data into account
    dataset = None,     # forward declaration
    verbose = 0,        # verbose level of training
)

ttf_before_shock_cfg : Data_cfg = dict(
    dataset = "new",
    usage_perc = 0.873,                 # use whole dataset
    nof_days_to_pred = 60,              # nof last datapoints of used dataset to predict
    split_train_val = [0.90,0.1],       # split ratio of remaining points
    n_vars = 1,
    loader = utils.data_loaders.get_gas_data,
    label_column = ["Price"]
)

ttf_after_shock_cfg : Data_cfg = dict(
    dataset = "new",                    # new or old TTF_FM set
    usage_perc = 1.0,                   # use whole dataset
    nof_days_to_pred = 60,              # nof last datapoints of used dataset to predict
    split_train_val = [0.90,0.1],      # split ratio of remaining points
    n_vars = 1,
    loader = utils.data_loaders.get_gas_data,
    label_column = ["Price"]
)
ttf_during_shock_cfg : Data_cfg = dict(
    dataset = "old",                    # new or old TTF_FM set
    usage_perc = 1.0,                   # use whole dataset
    nof_days_to_pred = 165,             # nof last datapoints of used dataset to predict
    split_train_val = [0.90,0.10],      # split ratio of remaining points
    n_vars = 1,
    loader = utils.data_loaders.get_gas_data,
    label_column = ["Price"]
)

ttf_during_shock_phelix_cfg : Data_cfg = dict(
    dataset = "old",                    # new or old TTF_FM set
    usage_perc = 1.0,                   # use whole dataset
    nof_days_to_pred = 165,             # nof last datapoints of used dataset to predict
    split_train_val = [0.90,0.10],      # split ratio of remaining points
    n_vars = 1,
    loader = utils.data_loaders.get_gas_elec_data,
    label_column =["Price"]
)

ttf_synth_cfg : Data_cfg = dict(
    dataset = "synth",                  # new or old TTF_FM set
    usage_perc = 1.0,                   # use whole dataset
    nof_days_to_pred = 165,             # nof last datapoints of used dataset to predict
    split_train_val = [0.90,0.10],      # split ratio of remaining points
    n_vars = 1,
    loader = utils.data_loaders.get_test_data,
    label_column = ["Price"]
)

airline_cfg : Data_cfg = dict(
    dataset = "airline",                # new or old TTF_FM set
    usage_perc = 1.0,                   # use whole dataset
    nof_days_to_pred = 20,              # nof last datapoints of used dataset to predict
    split_train_val = [0.90,0.10],      # split ratio of remaining points
    n_vars = 1,
    loader = utils.data_loaders.get_airline_data,
    label_column = ["Passengers"]
)

alpha = 0.1  # confidence lvl

lstm_cfg : Lstm_cfg = dict(
    units = 5,                  # nof LSTM units
    n_layers = 1,               # nof LSTM layers
    alpha = alpha,
    quantiles = [
        alpha/8,                # quantiles to predict - pb only
        1-(alpha/16)
    ],
    l2_lambda = 0.01,           # weight of l2 regularization
    learning_rate = 0.002,      # learning rate 
    batch_size = 100,           # size of batches using to train
    time_steps_in = 12,         # how many previous values are taken into account
    time_steps_out = 1,         # predict one day
    epochs = 1000,              # max training iterations
    patience = 50,              # iterations to wait without improving
    start_from_epoch = 100,     # epoch to start with patience
    loss_func = "pb",           # pb - pinball or qd - quality driven
    eta=150,                    # softening factor  - qd only
    lamb=0.01                   # lagrange multiplier - qd only
)

tcn_cfg : Tcn_cfg = dict(
    dilations = [1],            # dilation rate of the Conv1D layers
    n_filters = 16,             # filters in each Conv1D layer 
    kernel_size = 3,            # kernel size in each ConvID layer
    alpha = alpha,
    quantiles = [
        alpha/2,                # quantiles to predict
        # 0.5,
        1-(alpha/2)
    ],
    l2_lambda = 1e-5,           # weight of l2 regularization
    learning_rate = 0.005,      # learning rate
    batch_size = 20,            # size of batches using to train
    time_steps_in = 3,          # how many previous values are taken into account
    time_steps_out = 1,         # predict one day
    epochs = 1000,              # max training iterations
    patience = 30,              # iterations to wait without improving
    start_from_epoch = 0,       # epoch to start with patience
    loss_func = "pb",           # pb - pinball or qd - quality driven
    eta=150,                    # softening factor  - qd only
    lamb=0.01                   # lagrange multiplier - qd only
)

mlp_cfg : Mlp_cfg = dict(
    units = 32,                 # nof MLP units
    n_layers = 3,               # nof hidden layers
    alpha = alpha,
    quantiles = [
        alpha/2,                # quantiles to predict - pb only
        1-(alpha/2)
    ],
    l2_lambda = 0.0001,         # weight of l2 regularization
    learning_rate = 0.001,
    batch_size = 32,            # size of batches using to train
    time_steps_in = 3,          # how many previous values are taken into account
    time_steps_out = 1,         # predict one day
    epochs = 1000,              # max training iterations
    patience = 50,              # iterations to wait without improving
    start_from_epoch = 50,      # epoch to start with patience
    loss_func = "qd",           # pb - pinball or qd - quality driven
    eta=100,                    # softening factor  - qd only
    lamb=0.01                   # lagrange multiplier - qd only
)

plot_cfg : Plot_cfg = dict(
    plot_training = False       # plot training hist
)

# define main config class which is composed of the other configs

class Config:

    def __init__(self, dataset, model_type):

        self.training_cfg : Training_cfg = training_cfg
        self.plot_cfg : Plot_cfg = plot_cfg
    
        if dataset == "before_shock":
            self.data_cfg = copy.deepcopy(ttf_before_shock_cfg)
        elif dataset == "during_shock":
            self.data_cfg = copy.deepcopy(ttf_during_shock_cfg)
        elif dataset == "after_shock":
            self.data_cfg = copy.deepcopy(ttf_after_shock_cfg)
        elif dataset == "synth":
            self.data_cfg = copy.deepcopy(ttf_synth_cfg)
        elif dataset == "airline":
            self.data_cfg = copy.deepcopy(airline_cfg)
        elif dataset == "shock_with_phelix":
            self.data_cfg = copy.deepcopy(ttf_during_shock_phelix_cfg)
        else:
            raise ValueError("dataset must be 'after_shock', 'during_shock', 'before_shock', 'shock_with_phelix', 'synth', or 'airline'")
        
        self.training_cfg["dataset"] = dataset
        
        if model_type == "mlp":
            self.model_cfg =  mlp_cfg
            self.training_cfg["model_type"] = 'mlp'
        elif model_type == "lstm":
            self.model_cfg = lstm_cfg
            self.training_cfg["model_type"] = 'lstm'
        elif model_type == "tcn":
            self.model_cfg = tcn_cfg
            self.training_cfg["model_type"] = 'tcn'
        else:
            raise ValueError("model_type must be 'mlp', 'lstm', 'tcn'")
        
        self.data_cfg["nof_days_to_pred"] += self.model_cfg["time_steps_in"]

    def get_datasets(self, days_to_predict):
        data_loader = self.data_cfg["loader"]
        return data_loader(self.data_cfg, days_to_predict)

