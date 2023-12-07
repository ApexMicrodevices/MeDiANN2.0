
import numpy as np
from util import * 
# util -different utility functions


class DataImporter():

  def __init__(self,root_path=None, data_path = None, experiment_name='test',X_path_name=X_PATH_NAME ,R_path_name=R_PATH_NAME ,T_path_name = T_PATH_NAME):
    self.name = experiment_name
    if data_path == None:
      data_path = root_path + '/data'
      check_if_data_folder_exists(data_path)
    try:
      self.path = data_path
      self.data_type = X_path_name.split('.')[-1]
      self.X_path = data_path + '/' + X_path_name
      self.R_path = data_path + '/' + R_path_name
      self.T_path = data_path + '/' + T_path_name
    except:
      raise ValueError('Please indicate the path of your data!')
    
  def data_importer(self):
    """
    Given the path of the data and / or the name of the parameter, reflection and transmission
    file, you can run this code to import the dataset

    -----------------------------------------------------------------------------------------
    Input:

    root_path : The path where this folder is in 
    X_path_name: The name of the parameter file
    R_path: The name of the reflection file
    T_path: The name of the transmission file
    
    -----------------------------------------------------------------------------------------

    Output:

    X : preprocessed and filtered parameters
    R : preprocessed and filtered reflection
    T : preprocessed and filtered transmission
    """


    print('Welcome to MeDiANN! Your data is being imported.\n')
    print('The type of your data is selected as '+str(self.data_type))
    if self.data_type == 'csv':
      X,R,T = import_data_csv(self.X_path, self.R_path,self.T_path)
    if self.data_type == 'npy':
      X,R,T = import_data_npy(self.X_path,self.R_path,self.T_path)
    if self.data_type == 'dat':
      X,R,T = import_data_mat(self.X_path,self.R_path,self.T_path)
    return X,R,T

  def import_data(self):
    """
    Given the path of the data and / or the name of the parameter, reflection and transmission
    file, you can run this code to import the dataset

    -----------------------------------------------------------------------------------------
    Input:

    root_path : The path where this folder is in 
    X_path_name: The name of the parameter file
    R_path: The name of the reflection file
    T_path: The name of the transmission file
    
    -----------------------------------------------------------------------------------------

    Output:

    data: the dataset ready to be preprocessed by the input, with 
    parameters, reflection, transmission, training/val/test set division and scalers
    """
    X,R,T = self.data_importer()
    X,R,T = np.array(X),np.array(R),np.array(T)
    data = build_dataset(X,R,T,self.name)
    return data


  



