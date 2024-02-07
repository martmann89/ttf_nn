import numpy as np
import pandas
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
import tensorflow as tf

class WindowGenerator():
  
    def __init__(self, input_width, label_width, shift, df, label_columns=None):
    
        # Get the label column indices.
        self.label_columns = label_columns
        if label_columns is not None:
          self.label_columns_indices = {name: i for i, name in
                                        enumerate(label_columns)}
        self.column_indices = {name: i for i, name in enumerate(df.columns)}
    
        # Get the window parameters.
        self.input_width = input_width
        self.label_width = label_width
        self.shift = shift
    
        self.total_window_size = input_width + shift
    
        self.input_slice = slice(0, input_width)
        self.input_indices = np.arange(self.total_window_size)[self.input_slice]
    
        self.label_start = self.total_window_size - self.label_width
        self.labels_slice = slice(self.label_start, None)
        self.label_indices = np.arange(self.total_window_size)[self.labels_slice]
    
    def split_window(self, features):
        inputs = features[:, self.input_slice, :]
        labels = features[:, self.labels_slice, :]
        if self.label_columns is not None:
            labels = np.stack(
                [labels[:, :, self.column_indices[name]] for name in self.label_columns],
                axis=-1)
            
        return inputs, labels
    
    def __repr__(self):
        return '\n'.join([
            f'Total window size: {self.total_window_size}',
            f'Input indices: {self.input_indices}',
            f'Label indices: {self.label_indices}',
            f'Label column name(s): {self.label_columns}'])

    

def data_windowing(df, time_steps_in, time_steps_out, shift=None, label_columns=None, train_len=.8, val_len=.1, val_data=None, test_data=None, lstm=True):
    train_date = df.pop("Date")
    val_date = val_data.pop("Date")
    test_date = test_data.pop("Date")
    test_date = pandas.DataFrame([test_date,test_date]).transpose()
    if shift is None:
        shift=time_steps_out
    
    win = WindowGenerator(
        input_width=time_steps_in, 
        label_width=time_steps_out, 
        shift=shift,
        df=df,
        label_columns=label_columns)
    
    # Split data, if the val and test set are not given
    if val_data is None and test_data is None:
        assert(train_len+val_len < 1)
        N = df.shape[0]
        train_data = np.array(df[:int(N*train_len)].values).astype(np.float32)
        val_data = np.array(df[int(N*train_len):int(N*(train_len+val_len))].values).astype(np.float32)
        test_data = np.array(df[int(N*(train_len+val_len)):].values).astype(np.float32)
    else:
        train_data = np.array(df.values).astype(np.float32)
        val_data = np.array(val_data.values).astype(np.float32)
        test_data = np.array(test_data.values).astype(np.float32)
        test_date = np.array(test_date.values).astype(np.object_)
        
    # Initialize scaler
    scaler = xy_scaler()
    y_index = [df.columns.get_loc(col) for col in label_columns]
    scaler.fit(train_data, y_index)
    # Make windows
    train_window = np.stack([ train_data[i:i+win.total_window_size] for i in range(0, train_data.shape[0] - win.total_window_size, time_steps_out)])
    train_x, train_y = win.split_window(train_window)
    train_y = train_y[:,:,0]
    
    val_window = np.stack([ val_data[i:i+win.total_window_size] for i in range(0, val_data.shape[0] - win.total_window_size, time_steps_out)])
    val_x, val_y = win.split_window(val_window)
    val_y = val_y[:,:,0]
    
    test_window = np.stack([ test_data[i:i+win.total_window_size] for i in range(0, test_data.shape[0] - win.total_window_size+1, time_steps_out)])
    test_date_window = np.stack([ test_date[i:i+win.total_window_size] for i in range(0, test_data.shape[0] - win.total_window_size+1, time_steps_out)])
    test_x, test_y = win.split_window(test_window)
    test_x2, test_date_y = win.split_window(test_date_window)
    test_date = test_date_y[:,:,0]
    test_y = test_y[:,:,0]


    
    
    # Rescale data
    train_x = scaler.transform_x(train_x)
    train_y = scaler.transform_y(train_y)
    val_x = scaler.transform_x(val_x)
    val_y = scaler.transform_y(val_y)
    test_x = scaler.transform_x(test_x)
    test_y = scaler.transform_y(test_y)

    # Create additional features to improve LSTM performance (differences of first 4 Tschebicheff plolynomials)
    # if lstm == True:
    #     train_x = add_differences(train_x)
    #     val_x = add_differences(val_x)
    #     test_x = add_differences(test_x)


    # for i in range(train_x.shape[2]):
    #     temp_scaler = xy_scaler()
        
    #     temp_scaler.fit(train_x[:,:,i].reshape((train_x.shape[0]*train_x.shape[1],-1)), y_index)
    #     train_x[:,:,i] = temp_scaler.transform_x(train_x[:,:,i]).reshape((train_x.shape[0],train_x.shape[1]))
    #     print(train_x[:,:,i])
    #     val_x[:,:,i] = temp_scaler.transform_x(val_x[:,:,i]).reshape((val_x.shape[0],val_x.shape[1]))
    #     test_x[:,:,i] = temp_scaler.transform_x(test_x[:,:,i]).reshape((test_x.shape[0],test_x.shape[1]))
    # #     # train_x[:,:,i] = temp_scaler.transform_x(train_x[:,:,i]).reshape((train_x.shape[0],train_x.shape[1]))
    # #     # val_x[:,:,i] = temp_scaler.transform_x(val_x[:,:,i]).reshape((val_x.shape[0],val_x.shape[1]))
    # #     # test_x[:,:,i] = temp_scaler.transform_x(test_x[:,:,i]).reshape((test_x.shape[0],test_x.shape[1]))
    
    # train_y = scaler.transform_y(train_y)
    # val_y = scaler.transform_y(val_y)
    # test_y = scaler.transform_y(test_y)
    
   
    # Make training batches
    batch_len = int(np.floor(train_x.shape[0]))
    to_del = time_steps_in//time_steps_out
    train_data = []
    b=0
    train_data.append([train_x[b*batch_len:(b+1)*batch_len-to_del], train_y[b*batch_len:(b+1)*batch_len-to_del]])
        
    return train_data, val_x, val_y, test_x, test_y, scaler, train_date, val_date, test_date

class xy_scaler:
    """
    Transform X and Y data
        
    """
    
    def __init__(self, Scaler=MinMaxScaler):
        self.x_scaler = Scaler()
        self.y_scaler = Scaler()
        
    def fit(self, data, y_index):
        """
        Parameters
        ----------
        data : must have shape [time_steps, variables]
        y_index : list specifying the target variables

        """
        assert(len(data.shape) == 2)
        assert(isinstance(y_index, list))
        self.x_scaler.fit(data)
        data_y = data[:, y_index]
        self.y_scaler.fit(data_y)
                
    def transform_x(self, data):
        if len(data.shape) == 2:
            data = data[..., None]
        data_r = data.reshape(data.shape[0]*data.shape[1], -1)
        data_r = self.x_scaler.transform(data_r)
        data = data_r.reshape(data.shape[0], data.shape[1], -1)
        return data
    
    def transform_y(self, data):
        data_r = data.reshape(data.shape[0]*data.shape[1], 1)
        data_r = self.y_scaler.transform(data_r)
        data = data_r.reshape(data.shape[0], data.shape[1])
        return data
        
    def inverse_transform_x(self, data):
        if len(data.shape) == 2:
            data = data[..., None]
        data_r = data.reshape(data.shape[0]*data.shape[1], -1)
        data_r = self.x_scaler.inverse_transform(data_r)
        data = data_r.reshape(data.shape[0], data.shape[1], -1)
        return data
    
    def inverse_transform_y(self, data):
        data_r = data.reshape(data.shape[0]*data.shape[1], 1)
        data_r = self.y_scaler.inverse_transform(data_r)
        data = data_r.reshape(data.shape[0], data.shape[1])
        return data
    
    def inverse_transform_OneDim(self,data):
        return self.y_scaler.inverse_transform(data)
    

def add_differences(x):
    new = np.zeros((x.shape[0],x.shape[1],5))
    mean = tf.reduce_mean(x,axis=1)
    new[:,:,0] = mean
    diff = tf.subtract(x[:,:,0], mean)
    new[:,:,1] = diff
    new[:,:,2] = 2*tf.square(diff) - 1
    new[:,:,3] = 4*tf.multiply(diff,tf.square(diff)) - 3*diff
    new[:,:,4] = 8*tf.square(tf.square(diff)) - 8*tf.square(diff) + 1
    return new
