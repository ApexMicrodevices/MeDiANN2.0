#This library is a work-in-progress and it's meant to be the library that
#we are going to use to analyse our models and plot some statistics out of it 
import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd 
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf
from tensorflow import keras 
from sklearn.metrics import mean_absolute_error as mae 
from constants import * 
from util import * 
from MeDiANN import * 
from scipy.interpolate import interp2d


class ResultAnalyzer():

  def __init__(self,data,model,root_path=None,result_path=None,model_name='test',
               target='T',model_prediction=None, boost_model = None,
               boost_model_prediction=None, heatmap_path = None,experiment_name='test'):
    self.root_path = root_path
    self.data = data
    self.model = model 
    self.target = data[target]
    self.scalers = data['scalers']
    if result_path == None:
      result_path = root_path + '/result'
      check_if_data_folder_exists(result_path)
    self.result_path = result_path
    self.model_name = model_name
    if model_prediction==None:
      self.prediction = model.predict(data['X'])
    self.test_list = data['test_set']
    self.boost_model = boost_model
    if boost_model!=None:
      if boost_model_prediction!=None:
        self.boost_prediction = self.prediction + self.boost_model_prediction
      else:
        boost_model_prediction = boost_model.predict(data['X'])
        self.boost_model_prediction = self.prediction + boost_model_prediction
    self.heatmap_path = heatmap_path
    self.experiment_name = experiment_name

  def validate_method(self,method='baseline'): 

    """
    
    """



    if method=='baseline':
      test_set_pred = self.prediction[self.test_list]
      test_set_real = self.target[self.test_list]
      mae_list = np.array([mae(test_set_pred[i],test_set_real[i]) for i in range(len(self.test_list))])
    else:
      test_set_pred = self.boost_model_prediction[self.test_list]
      test_set_real = self.target[self.test_list]
      mae_list = np.array([mae(test_set_pred[i],test_set_real[i]) for i in range(len(self.test_list))])
    return mae_list

  def compare_methods(self,baseline_error, boost_error=None):
    baseline_error_column = self.experiment_name+', baseline error'
    boosted_model_error_column = self.experiment_name+', boosted model error'
    df = pd.DataFrame(np.array([baseline_error,boost_error]).T,columns = [baseline_error_column,boosted_model_error_column])
    df.describe().to_csv(self.result_path+'/error_comparison_'+self.experiment_name+'.csv')
    return df.describe()
    
  def infere_new_points(self,new_input,trained_model):
    scaled_inputs = []
    for point in new_input:
      scaled_input = np.array([self.scalers[i].transform(np.array(point[i]).reshape(-1,1)) for i in range(len(self.scalers))]).ravel()
      scaled_input = scaled_input.astype(np.float32)
      scaled_inputs.append(scaled_input)
    scaled_inputs = np.array(scaled_inputs)
    model = trained_model#['model']
    pred = model.predict(scaled_inputs) 
    return pred    

  def heatmap_data_importer(self,trained_model=None):
    if self.heatmap_path == None:
      raise ValueError('There is no valid path for the heatmap data. Please add path')
    else:
      heatmap_dict = build_heatmap_data(self.heatmap_path)
      input_data = heatmap_dict['heatmap_input']
      #input_data = filter_X(input_data)
      if trained_model !=None:
        R_pred = self.infere_new_points(input_data,trained_model)
      else:
        R_pred = None
    return heatmap_dict,R_pred


  def heatmap_plotter(self,method='target'):
    trained_model = None
    name_target_heatmap = self.result_path+'/'+self.experiment_name+'_target.png'
    name_heatmap = self.result_path+'/'+self.experiment_name+'_'+method+'_heatmap'
    heatmap_data,R = self.heatmap_data_importer(trained_model)
    if method =='baseline':
      print('The baseline method heatmap is being used to run the heatmap\n')
      trained_model = self.model
      heatmap_data,R = self.heatmap_data_importer(trained_model)
      R_size = R.shape[1]
      R_pred = R[:,R_size//2]
    if method =='boost':
      print('The boost method heatmap is being used to run the heatmap\n')
      trained_model = self.model
      heatmap_data,R = self.heatmap_data_importer(trained_model)
      R_size = R.shape[1]
      R_pred = R[:,R_size//2]
      boost_model = self.boost_model
      heatmap_data,R = self.heatmap_data_importer(boost_model)
      R_pred = R_pred + R[:,R_size//2]
    heatmap_data_params = heatmap_data['raw_heatmap_input']
    Xd = heatmap_data_params['Xd'].astype(float)
    Pitch = heatmap_data_params['Pitch'].astype(float)
    R_target = heatmap_data['target'].astype(float)
    if method=='target':
      R_pred = heatmap_data['target'].astype(float)
      plot_heatmap(Xd,Pitch,R_pred)
    title = 'Method = ' + method 
    plot_heatmap(Xd,Pitch,R_pred,name_heatmap+'.png',title)
    diff = np.abs(R_pred-R_target)
    diff[diff>0.1] = 0.1
    title = 'Method = ' + method + ' difference with target'
    plot_heatmap(Xd,Pitch,diff,name_heatmap+'_diff.png',title)
    title = 'Target heatmap'
    plot_heatmap(Xd,Pitch,R_target,name_target_heatmap,title)
    return R,R_pred
  

  def sample_space(self):
    #Given the input data we build a grid of geometrical parameters that we have
    data = self.data
    return build_grid(self.data,NUM_SAMPLE)
  
  def sample_space_and_infere(self,wavelength):
    data_on_grid = self.sample_space() #Sample the space
    pred = infere_on_grid(data_on_grid,model_baseline=self.model,model_boost=  self.boost_model) #Infere your prediction on the space
    num_wavelength = int(pred.shape[1]/2) #Number os possible wavelengts
    wavelength_list = np.linspace(MIN_WAVELENGTH,MAX_WAVELENGTH,num_wavelength) #Minimum,maximum, number of wavelengths
    picked_wavelength_index = np.argmin(np.abs(wavelength_list-wavelength)) #From the wavelengthyou give me I select the right one 
    return pred[:,picked_wavelength_index],pred[:,picked_wavelength_index+int(pred.shape[1]/2)]
  
  def infere_from_grid(self,data_on_grid,wavelength):
    pred = infere_on_grid(data_on_grid,model_baseline=self.model,model_boost=  self.boost_model)
    num_wavelength = int(pred.shape[1]/2)
    wavelength_list = np.linspace(MIN_WAVELENGTH,MAX_WAVELENGTH,num_wavelength)
    picked_wavelength_index = np.argmin(np.abs(wavelength_list-wavelength))
    return pred[:,picked_wavelength_index],pred[:,picked_wavelength_index+int(pred.shape[1]/2)]