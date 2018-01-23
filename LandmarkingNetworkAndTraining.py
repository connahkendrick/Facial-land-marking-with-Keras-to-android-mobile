# In[ ]: 
    import cv2
    import os
    import numpy 
    import pandas 
    from sklearn.utils import shuffle  
    from keras.wrappers.scikit_learn import KerasRegressor  
    from sklearn.model_selection import cross_val_score 
    from sklearn.model_selection import KFold
    from sklearn.preprocessing import StandardScaler 
    import matplotlib.pyplot as plt 
    import time 
    from sklearn.pipeline import Pipeline 
    
    from keras.models import Sequential  
    from keras.layers import Dense, Activation 
    from keras.layers import Convolution2D, MaxPooling2D, Flatten
    from keras.optimizers import SGD 
    from keras.models import model_from_json   
    numpy.random.seed(seed=7)
    
# In[ ]:
    # Read full data set we can not provide the dataset 
    # X should be your image data in the format (number of images, width, height, channels) 
    # y should be the landmarks in the format (number of images, x point, y point ....) 
    
    # we store all our training in a single csv file for convience
    
    FTrain = r'' 
    def load(test=False, cols=None):
        fname = FTest if test else FTrain 
        df = pandas.read_csv(os.path.expanduser(fname)) 
        
        df['Image'] = df['Image'].apply(lambda im: numpy.fromstring(im, sep=' ')) 
    
        if cols:
            df = df[list(cols)+['Image']]
            
        print df.count() 
        df = df.dropna() 
        
        X = numpy.vstack(df['Image'].values)/255 
        X = X.astype(numpy.float32) 
        
        if not test: 
            y = df[df.columns[:-1]].values 
            y = y.astype(numpy.float32)
        else: 
            y = None 
            
        return X, y 
    
    def LoadImages(path): 
        df = pandas.read_csv(os.path.expanduser(path)) 
        df['Image'] = df['Image'].apply(lambda im: numpy.fromstring(im, sep=' ')) 
    
        if cols:
            df = df[list(cols)+['Image']]
            
        print df.count() 
        df = df.dropna() 
        
        X = numpy.vstack(df['Image'].values)/255 
        X = X.astype(numpy.float32) 
        
        return X
    
    def GetLineCount(): 
        df = pandas.read_csv(os.path.expanduser(FTrain))
        return df.shape[0]

    def loadParticalData(NumOn): 
        df = pandas.read_csv(FTrain) 
        df['Image'] = df['Image'].apply(lambda im: numpy.fromstring(im, sep=' ')) 
            
        print df.count() 
        df = df.dropna() 
        
        X = numpy.vstack(df['Image'].values)/255 
        X = X.astype(numpy.float32) 
        
        if not test: 
            y = df[df.columns[:-1]].values 
            y = y.astype(numpy.float32)
        else: 
            y = None 
            return X, y

    def load2d(test=False, cols=None): 
        X, y = load(test, cols) 
        X = X.reshape(-1,1,96,96)
        
        return X, y   
    
    def load2dPartial(numOn) :
        X,y = loadParticalData(numOn) 
        X = X.reshape(-1,1,96,96)
        
    
 
# In[ ]:  
print("Loading Data")
X, y = load2d(test=False) 
print("Data Loaded")


#neural net 
model = Sequential() 
model.add(Convolution2D(32, 3, 3, input_shape=(1, 96, 96))) 
model.add(Activation('relu')) 
model.add(MaxPooling2D(pool_size=(2,2))) 

model.add(Convolution2D(64, 2, 2)) 
model.add(Activation('relu')) 
model.add(MaxPooling2D(pool_size=(2,2))) 

model.add(Convolution2D(128, 2, 2)) 
model.add(Activation('relu')) 
model.add(MaxPooling2D(pool_size=(2,2))) 

model.add(Flatten()) 
model.add(Dense(1000)) 
model.add(Activation('relu')) 
model.add(Dense(500)) 
model.add(Activation('relu')) 
model.add(Dense(136)) 


print("Compling Model")
model.compile(loss='mean_squared_error',optimizer='adam',metrics=['accuracy','mae','fmeasure','precision','recall'])   
print("Model Compiled")   

print("Fitting Model")
hist1 = model.fit(X, y, nb_epoch=100, batch_size=60, validation_split=0.1, verbose = 1) 
json_string = model.to_json() 
open('Base100Epoch.json','w').write(json_string) 
model.save_weights('Base100Epoch.h5')   