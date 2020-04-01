


##################### SYNTHETIC DATASETS ###############################################
##################### SYNTHETIC DATASETS ###############################################
##################### SYNTHETIC DATASETS ###############################################


########################### BOOTSTRAPPING ##################################################

# from sklearn import cross_validation
from sklearn.model_selection import train_test_split
from scipy.sparse import coo_matrix
from sklearn.utils import resample



data_metrics_1_7_features_test = np.load("data_metrics_1_7_features_test.npy" , allow_pickle = True)
data_metrics_1_7_features_train = np.load("data_metrics_1_7_features_train.npy" , allow_pickle = True)
data_metrics_1_7_features = data_metrics_1_7_features_test
data_metrics_1_7_features = np.concatenate((data_metrics_1_7_features,data_metrics_1_7_features_train),axis = 0)
X = data_metrics_1_7_features

datasets_nn_1_recall_no_spaces_test = np.load("datasets_nn_1_recall_no_spaces_test.npy", allow_pickle = True )
datasets_nn_1_recall_no_spaces_train = np.load("datasets_nn_1_recall_no_spaces_train.npy", allow_pickle = True)
datasets_nn_1_recall_no_spaces = datasets_nn_1_recall_no_spaces_test
datasets_nn_1_recall_no_spaces = np.concatenate((datasets_nn_1_recall_no_spaces, datasets_nn_1_recall_no_spaces_train), axis = 0)
y = datasets_nn_1_recall_no_spaces


datasets_nn_1_precision_no_spaces_test = np.load("datasets_nn_1_precision_no_spaces_test.npy", allow_pickle = True )
datasets_nn_1_precision_no_spaces_train = np.load("datasets_nn_1_precision_no_spaces_train.npy", allow_pickle = True)
datasets_nn_1_precision_no_spaces = datasets_nn_1_precision_no_spaces_test
datasets_nn_1_precision_no_spaces = np.concatenate((datasets_nn_1_precision_no_spaces, datasets_nn_1_precision_no_spaces_train), axis = 0)
yprecision = datasets_nn_1_precision_no_spaces






X_train_datasets_1_bootstrap = []
y_train_datasets_1_bootstrap = []
X_test_datasets_1_bootstrap = []
y_test_datasets_1_bootstrap = []


l = len(X)

trainx = X[: int(0.9 * l - 1)]
testx = X[int(0.9 * l - 1):]
trainy = y[: int(0.9 * l - 1)]
testy = y[int(0.9 * l - 1):]
trainyp = yprecision[: int(0.9 * l - 1)]
testyp = yprecision[int(0.9 * l - 1):]



X_sparse_train = coo_matrix(trainx)
X_sparse_test = coo_matrix(testy)


############### RECALL ################################

for i in range(50):
    trainx_resampled, X_sparse_train_resampled, trainy_resampled = resample(trainx, X_sparse_train , trainy, random_state=i)
    Xtrain = trainx_resampled
    ytrain = trainy_resampled

    X_train_datasets_1_bootstrap.append(Xtrain)
    y_train_datasets_1_bootstrap.append(ytrain)



for i in range(50):
    testx_resampled, X_sparse_test_resampled, testy_resampled = resample(testx, X_sparse_test , testy, random_state=i)
    Xtest = testx_resampled
    ytest = testy_resampled


    X_test_datasets_1_bootstrap.append(Xtest)
    y_test_datasets_1_bootstrap.append(ytest)


np.save("X_train_datasets_1_bootstrap.npy",X_train_datasets_1_bootstrap)
np.save("y_train_recall_datasets_1_bootstrap.npy",y_train_datasets_1_bootstrap)
np.save("X_test_datasets_1_bootstrap.npy",X_test_datasets_1_bootstrap)
np.save("y_test_recall_datasets_1_bootstrap.npy",y_test_datasets_1_bootstrap)


######### PRECISION #############################
    
    
y_train_datasets_1_bootstrap = []
y_test_datasets_1_bootstrap = []
    
    
    
    
for i in range(50):
    trainx_resampled, X_sparse_train_resampled, trainyp_resampled = resample(trainx, X_sparse_train , trainyp, random_state=i)
    ytrain = trainyp_resampled

    y_train_datasets_1_bootstrap.append(ytrain)



for i in range(50):
    testx_resampled, X_sparse_test_resampled, testyp_resampled = resample(testx, X_sparse_test , testyp, random_state=i)
    ytest = testy_resampled

    y_test_datasets_1_bootstrap.append(ytest)


    

np.save("y_train_precision_datasets_1_bootstrap.npy",y_train_datasets_1_bootstrap)
np.save("y_test_precision_datasets_1_bootstrap.npy",y_test_datasets_1_bootstrap)


