# from sklearn import cross_validation
from sklearn.model_selection import train_test_split
from scipy.sparse import coo_matrix
from sklearn.utils import resample



data_metrics_1_7_features_test = np.load("E:\\Internships_19\\Internship(Summer_19)\\Imbalanced_class_classification\\Class_Imabalanced_Learning_Code\\CIL Code\\RESULTS\\New_synthetic_data\\data_metrics_1_7_features_test.npy" , allow_pickle = True)
data_metrics_1_7_features_train = np.load("E:\\Internships_19\\Internship(Summer_19)\\Imbalanced_class_classification\\Class_Imabalanced_Learning_Code\\CIL Code\\RESULTS\\New_synthetic_data\\data_metrics_1_7_features_train.npy" , allow_pickle = True)
data_metrics_1_7_features = data_metrics_1_7_features_test
data_metrics_1_7_features = np.concatenate((data_metrics_1_7_features,data_metrics_1_7_features_train),axis = 0)
X = data_metrics_1_7_features

datasets_nn_1_recall_no_spaces_test = np.load("E:\\Internships_19\\Internship(Summer_19)\\Imbalanced_class_classification\\Class_Imabalanced_Learning_Code\\CIL Code\\RESULTS\\New_synthetic_data\\datasets_nn_1_recall_no_spaces_test.npy", allow_pickle = True)
datasets_nn_1_recall_no_spaces_train = np.load("E:\\Internships_19\\Internship(Summer_19)\\Imbalanced_class_classification\\Class_Imabalanced_Learning_Code\\CIL Code\\RESULTS\\New_synthetic_data\\datasets_nn_1_recall_no_spaces_train.npy", allow_pickle = True)
datasets_nn_1_recall_no_spaces = datasets_nn_1_recall_no_spaces_test
datasets_nn_1_recall_no_spaces = np.concatenate((datasets_nn_1_recall_no_spaces, datasets_nn_1_recall_no_spaces_train), axis = 0)
yrecall = datasets_nn_1_recall_no_spaces

datasets_nn_1_precision_no_spaces_test = np.load("E:\\Internships_19\\Internship(Summer_19)\\Imbalanced_class_classification\\Class_Imabalanced_Learning_Code\\CIL Code\\RESULTS\\New_synthetic_data\\datasets_nn_1_precision_no_spaces_test.npy", allow_pickle = True)
datasets_nn_1_precision_no_spaces_train = np.load("E:\\Internships_19\\Internship(Summer_19)\\Imbalanced_class_classification\\Class_Imabalanced_Learning_Code\\CIL Code\\RESULTS\\New_synthetic_data\\datasets_nn_1_precision_no_spaces_train.npy", allow_pickle = True)
datasets_nn_1_precision_no_spaces = datasets_nn_1_precision_no_spaces_test
datasets_nn_1_precision_no_spaces = np.concatenate((datasets_nn_1_precision_no_spaces, datasets_nn_1_precision_no_spaces_train), axis = 0)
yprecision = datasets_nn_1_precision_no_spaces


X_train_datasets_1_bootstrap = []
X_test_datasets_1_bootstrap = []

y_train_precision_datasets_1_bootstrap = []
y_test_precison_datasets_1_bootstrap = []

y_train_recall_datasets_1_bootstrap = []
y_test_recall_datasets_1_bootstrap = []

l = len(X)
X_sparse = coo_matrix(X)


for i in range(50):
    X, X_sparse, yprecision = resample(X, X_sparse , yprecision, random_state=i)
    Xtrain = X[: int(0.9 * l - 1)]
    Xtest = X[int(0.9 * l - 1):]
    
    ytrain = yprecision[: int(0.9 * l - 1)]
    ytest = yprecision[int(0.9 * l - 1):]
    
    X_train_datasets_1_bootstrap.append(Xtrain)
    X_test_datasets_1_bootstrap.append(Xtest)
    
    y_train_precision_datasets_1_bootstrap.append(ytrain)  
    y_test_precision_datasets_1_bootstrap.append(ytest)
    
    
for i in range(50):
    X, X_sparse, yrecall = resample(X, X_sparse , yrecall, random_state=i)
    Xtrain = X[: int(0.9 * l - 1)]
    ytrain = yrecall[: int(0.9 * l - 1)]
    Xtest = X[int(0.9 * l - 1):]
    ytest = yrecall[int(0.9 * l - 1):]
    
    
    
    y_train_recall_datasets_1_bootstrap.append(ytrain)
    y_test_recall_datasets_1_bootstrap.append(ytest)
    
    
    
np.save("X_train_datasets_1_bootstrap.npy",X_train_datasets_1_bootstrap)
np.save("X_test_datasets_1_bootstrap.npy",X_test_datasets_1_bootstrap)

np.save("y_train_recall_datasets_1_bootstrap.npy",y_train_datasets_1_bootstrap)
np.save("y_test_recall_datasets_1_bootstrap.npy",y_test_datasets_1_bootstrap)

np.save("y_train_precision_datasets_1_bootstrap.npy",y_train_datasets_1_bootstrap)
np.save("y_test_precision_datasets_1_bootstrap.npy",y_test_datasets_1_bootstrap)
   
