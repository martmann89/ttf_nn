from models.models import regression_model
import utils.utils as utils
from config import Config
from keras import backend as K
import tensorflow as tf
import gc

def predict(train_data, val_x, val_y, test_x, test_y, cfg : Config, weights = None):
    # for multiprocessing purpose
    K.clear_session()
    tf.compat.v1.reset_default_graph() # TF graph isn't same as Keras graph
    NN_cfg = cfg.model_cfg
    model = regression_model(cfg)
    hist = model.fit(train_data[0][0], train_data[0][1], val_x, val_y, epochs=NN_cfg['epochs'], patience=NN_cfg['patience'], start_from_epoch=NN_cfg['start_from_epoch'], learning_rate=NN_cfg['learning_rate'],weights=weights, verbose=cfg.training_cfg["verbose"])
    if cfg.plot_cfg['plot_training']:
        utils.plot_history(hist, NN_cfg['alpha'])

    PI = model.transform(test_x)
    # for multiprocessing purpose
    del model
    gc.collect()
    return PI, hist
                                                   
                                                  