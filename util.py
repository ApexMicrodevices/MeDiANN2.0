import numpy as np 
from sklearn.preprocessing import LabelEncoder,MinMaxScaler
from constants import * 
from scipy.io import loadmat
import pandas as pd 
import matplotlib.pyplot as plt
import os
from scipy.interpolate import griddata



def import_data_csv(X_path,R_path,T_path):
    try:
        X = np.array(pd.read_csv(X_path).drop('Unnamed: 0',axis=1))
    except:
        raise ValueError('The parameters are not in a .csv format.')
    try:
        R = np.array(pd.read_csv(R_path).drop('Unnamed: 0',axis=1))
    except:
        raise ValueError('The reflection values are not in a .csv format.')
    try:
        T = np.array(pd.read_csv(T_path).drop('Unnamed: 0',axis=1))
    except:
        raise ValueError('The transmission values are not in a .npy format.')
    return X,R,T


def import_data_npy(X_path,R_path,T_path):
    X = np.load(X_path,allow_pickle=True)
    try:
        X = np.load(X_path,allow_pickle=True)
    except:
        raise ValueError('The parameters are not in a .npy format.')
    try:
        R = np.load(R_path,allow_pickle=True)
    except:
        raise ValueError('The reflection values are not in a .npy format.')
    try:
        T = np.load(T_path,allow_pickle=True)
    except:
        raise ValueError('The transmission values are not in a .npy format.')
    return X,R,T


def import_data_mat(X_path,R_path,T_path):
    try:
        X = loadmat(X_path)
    except:
        raise ValueError('The parameters are not in a .dat format.')
    try:
        R = loadmat(R_path)
    except:
        raise ValueError('The reflection values are not in a .dat format.')
    try:
        T = loadmat(T_path)
    except:
        raise ValueError('The transmission values are not in a .dat format.')
    return X,R,T


def X_converter(X,name):
    """ The parameter space is a mixture of text and values, we 
    are converting them into a list of float using this code
    """
    scalers = []
    for i in range(X.shape[1]):
        #Scale floating features
        try:
            x = X[:,i].astype(float)
            X[:,i] = x
            scaler = MinMaxScaler()
            scaler.fit(X[:,i].reshape(-1,1))
           # scaler_filename = SCALER_PATH+"/scaler_feature_"+str(i)+'_experiment_'+name+'.save'
           # joblib.dump(scaler, scaler_filename) 
            X[:,i] = scaler.transform(X[:,i].reshape(-1,1)).reshape(len(X[:,i]))
            scalers.append(scaler)
        #Converting string into labels (integer)
        except:
            le = LabelEncoder()
            le.fit(X[:,i])
           # le_encoder_filename = SCALER_PATH+"/encoder_feature_"+str(i)+'_experiment_'+name+'.save'
           # joblib.dump(le, le_encoder_filename) 
            X[:,i] = le.transform(X[:,i]).astype(float)
            scalers.append(le)
    print('Transformation of X is successfull! \n')
    return X.astype(float),scalers


def check_irregularity(X,R,T):
    """We shouldn't have any R>1 or T>1, but if it happens this function filters them out
          """
    X =np.array([X[i] for i in range(len(R)) if max(R[i])<=1 and max(T[i])<=1]) #Filtering X
    new_R = np.array([R[i] for i in range(len(R)) if max(R[i])<=1 and max(T[i])<=1]) #Filtering R 
    new_T =np.array([T[i] for i in range(len(R)) if max(R[i])<=1 and max(T[i])<=1]) #Filtering T 
    R = new_R #Replacing R 
    T = new_T #Replacing T 
    print('Filtering is successfull! \n')
    print('The dimension of the parameter array is ',X.shape)
    print('The dimension of the reflection array is ', R.shape)
    print('The dimension of the transmission array is ', T .shape)
    return X,R,T


def split_data(X_shape,split_train,split_val):
    """Building the training set, validation set and test set for the dataset"""
    index_list = np.arange(0,X_shape) #All the index
    np.random.shuffle(index_list) #randomizing them
    #train_list = index_list[0:int(X_shape*split_train)] #building training set 
    train_list = index_list[0:int(X_shape*split_val)]
   #val_list = index_list[int(X_shape*split_train):int(X_shape*split_val)] #building validation set 
    val_list = index_list[int(X_shape*split_val):] 
    test_list = index_list[int(X_shape*split_val):] #building test set
    print('Training, validation, test split is successfull! \n')
    return {'train_list':train_list,'val_list':val_list,'test_list':test_list}


def preprocessing_block(X,R,T,name):
    X,R,T = check_irregularity(X, R, T)
    X = filter_X(X)
    X,scalers = X_converter(X,name)
    R = np.hstack((R,T))
    return X,R,T,scalers


def build_dataset(X,R,T,name):
    """We want to have the same class of parameters for R and T when 'a' and 'c' change.
    This function ensures the same train, validation and test set
    """
    unprocessed_input = X.copy()
    X,R,T,scalers = preprocessing_block(X,R,T,name)
    index_dataset = split_data(len(X),0.98,0.99)
    train_set, test_set, val_set = index_dataset['train_list'], index_dataset['test_list'], index_dataset['val_list']
    return {'X':X,'R':R,'T':T,'train_set':train_set,'test_set':test_set,'val_set':val_set,'scalers':scalers,'unprocessed_input':unprocessed_input}

def extract_root_path():
    path_name = os.path.abspath(os.getcwd())
    return path_name


def check_if_data_folder_exists(data_path):    
    # Check whether the specified
    # path exists or not
    isExist = os.path.exists(data_path)
    if isExist == False:
        os.makedirs(data_path)


def import_heatmap_data(heatmap_path):
    df = pd.read_csv(heatmap_path)
    df_columns = df.columns.tolist()
    Xd = np.array(df[df_columns[0]])
    Pitch = np.array(df[df_columns[1]])
    R = np.array(df[df_columns[-1]])
    return {'Xd':Xd,'Pitch':Pitch,'Value':R}


def build_heatmap_data(heatmap_path,state='a_gst'):
    heatmap_data = import_heatmap_data(heatmap_path)
    Xd, Pitch,R = heatmap_data['Xd'],heatmap_data['Pitch'], heatmap_data['Value']
    heatmap_inputs = []
    for i in range(len(Xd)):
        heatmap_inputs.append([Pitch[i],H_FIGURE_3,Xd[i],state])
        #heatmap_inputs.append([Pitch[i],H_FIGURE_3,Xd[i]])
    return {'raw_heatmap_input':heatmap_data,'heatmap_input':np.array(heatmap_inputs),'target':R}


def plot_heatmap(Xd,Pitch,Value,name_file,title,color_bar_range=[0,1]):
    X = np.array(Xd)
    Y = np.array(Pitch)
    Z = np.array(Value)
    # Create a heatplot
    # Define a grid on which to interpolate
    xi = np.linspace(min(X), max(X), 200)
    yi = np.linspace(min(Y), max(Y), 200)
    xi, yi = np.meshgrid(xi, yi)
    # Interpolate the Z values onto the grid using scipy's griddata
    zi = griddata((X, Y), Z, (xi, yi), method='cubic')
    plt.title(title)
    plt.imshow(zi, extent=(min(X), max(X), min(Y), max(Y)), origin='lower', cmap='viridis', aspect=0.5,vmin=color_bar_range[0], vmax=color_bar_range[1])
    colorbar = plt.colorbar()
    plt.xlabel(r'$X (\mu m)$')
    plt.xlabel(r'$\Lambda (\mu m)$')
    plt.savefig(name_file)



def filter_X(X):
    X_sets = [list(set(X[:,i])) for i in range(len(X.T))]
    keep_i = []
    for i in range(len(X_sets)):
        if len(X_sets[i])>1:
            keep_i.append(i)
    return X[:,keep_i]


# def build_grid(data,N=NUM_SAMPLE):
#     x,h,l = data['X'][:,0],data['X'][:,1],data['X'][:,2]
#     x_min,x_max = x.min(),x.max()
#     h_min,h_max = h.min(),h.max()
#     l_min,l_max = l.min(),l.max()
#     x_space = np.linspace(x_min,x_max,N)
#     l_space = np.linspace(l_min,l_max,N)
#     h_space = np.linspace(h_min,h_max,N)
#     data_space = []
#     for x_i in x_space:
#         for l_i in l_space:
#             for h_i in h_space:
#                 data_space.append([x_i,l_i,h_i])
#     data_space = np.array(data_space)
#     return data_space

def infere_on_grid(data_on_grid,model_baseline,model_boost=None):
    model_pred = model_baseline.predict(data_on_grid)
    if model_boost != None:
        model_pred = model_pred + model_boost.predict(data_on_grid)
    return model_pred


def scale_data(scalers,data):
    """
    Scaling the grid data using the scalers

    -------------------------------------------------------------------------------------
    Input:

    scalers: Scaler list
    -------------------------------------------------------------------------------------
    Output:

    preprocessed_data: The preprocessed data

    """
    data = np.array(data)
    scaled_data = [np.array(scalers[i].transform(data[:,i].reshape(-1,1))) for i in range(len(scalers))]
    preprocessed_data = np.array([scaled_data[i][:,0] for i in range(len(scaled_data)-1)])
    preprocessed_data = np.vstack((preprocessed_data,scaled_data[-1])).T
    return preprocessed_data