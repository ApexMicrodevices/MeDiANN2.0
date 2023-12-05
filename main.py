import numpy as np 
import matplotlib.pyplot as plt 
from constants import * 
from data_importer import DataImporter
from MeDiANN import MeDiANN_Magic
from util import * 
from result_analyzer import ResultAnalyzer
import warnings

if __name__=='__main__':

    warnings.filterwarnings("ignore")

    #The experiment name is the first thing that is required. The default value is "test"
    experiment_name = 'test_for_Debanik'
    #The first thing that is required in this script is to import the dataset from a path:
    root_path= os.getcwd()
    parameter_file_name = 'X_angle_45_polarization_s.npy' #Name of the parameter name 
    reflection_file_name = 'reflection_angle_45_polarization_s.npy' # Name of the reflection file name  
    transmission_file_name = 'transmission_angle_45_polarization_s.npy' #Name of the transmission file name
    data_path = root_path+'/data_angle_45_polarization_s/' #Name of the path where you see the data
    pre_trained_model_path = root_path+'/model/' #Insert model path 
    output_path = root_path +'/output/' #Insert output 
    result_path = root_path + '/result/' #Result path 
    #output_file_type = 'csv'

    #Let's import the data using the DataImporter Module:

    data_importer = DataImporter(root_path=root_path,data_path=data_path,experiment_name=experiment_name,
                                 X_path_name=parameter_file_name,T_path_name=transmission_file_name,
                                 R_path_name=reflection_file_name)

    data = data_importer.import_data()
    #As the data are imported, we can select if we want to use a pretrained model or train one from scratch
    #################### Pretrained model ##############################################
    
    #activation_function = 'relu' #If you want, you can change the activation function
    mediann = MeDiANN_Magic(data,pre_trained_model_path=None,experiment_name=experiment_name,
                            output_path=None,activation_function = ACTIVATION_FUNCTION,
                            output_type = OUTPUT_FILE_TYPE,root_path = root_path,epochs=2)
    #pre_trained_model, pred = mediann.predict_using_pretrained_model()

    ################## Train baseline and boosted model ###############################

    baseline_model = mediann.train_baseline_model()
    boosted_model = mediann.train_boost_model(baseline_model)
    baseline_pred = mediann.predict(model=baseline_model)
    boosted_pred = mediann.predict_boost(model_pred=baseline_pred, model_boost=boosted_model)

    ################# Validate model ##################################################

    selected_model, selected_model_name = baseline_model['model'], 'baseline model'
    boosted_model = boosted_model['model']
    target = 'R'
    heatmap_path = '/home/apexmds/Desktop/pcm_phc_tr/MeDiANN_AI2.0/data/heatmap_0.5_numbasis51.csv'
    result_analyzer = ResultAnalyzer(data=data,model=selected_model,model_name=selected_model_name,
                                     target=target,boost_model=boosted_model,result_path=result_path, root_path=root_path,
                                     heatmap_path=heatmap_path,experiment_name=experiment_name)
    
    baseline_error = result_analyzer.validate_method(method='baseline')
    boost_error = result_analyzer.validate_method(method='boost')
    comparison_result = result_analyzer.compare_methods(baseline_error,boost_error)

    ####################### Heatmap plotter #########################################

    result_analyzer.heatmap_plotter(method='boost')
    result_analyzer.heatmap_plotter(method='baseline')