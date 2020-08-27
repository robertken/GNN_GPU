﻿#option('outputLimit',2000);
#option('hthorMemoryLimit',10000);
IMPORT Python3 AS Python;
IMPORT $.GNN_ML_454_stock AS GNN;
IMPORT GNN.Tensor;
IMPORT GNN.Internal.Types AS iTypes;
IMPORT GNN.Types;
IMPORT GNN.GNNI;
IMPORT GNN.Utils;
IMPORT GNN.Internal AS Int;
IMPORT Std.System.Thorlib;

IMPORT mnist_as_real as mnist;

//t_Tensor := Tensor.R4.t_Tensor;
//TensData := Tensor.R4.TensData;

//RAND_MAX := POWER(2,32) -1;


#WORKUNIT('name','CNN STOCK GNN 454'); 

// Test parameters
trainCount := 60000;
testCount := 8196;
numOfGPUs := 1; //16 for total system GPU count
numOfPhysicalMachines := 2;

aggregateInterval := 1; //how many times do you want the weights to be aggregated per epoch
batch := 128;
numEpochs := 15;
featureCount := 784;
weightAggregateInterval := trainCount / (numOfGPUs * aggregateInterval); //old "batchSize"
numOfGPUperMachine := numOfGPUs / numOfPhysicalMachines;


// END Test parameters

/*
mnist_rec_type := RECORD
			UNSIGNED8 id;
      UNSIGNED1 label;
      SET OF UNSIGNED1 pixel;
END;
*/
//OUTPUT(mnist.train);
//OUTPUT(mnist.test);

trainData := CHOOSEN(mnist.train,trainCount);
testData := CHOOSEN(mnist.test,testCount);

myXTensData := NORMALIZE(trainData, featureCount,
                     TRANSFORM(Tensor.R4.TensData,
                       SELF.indexes := [LEFT.id, (COUNTER-1) DIV 28 + 1, (COUNTER-1) % 28 +1, 1],
                       SELF.value := LEFT.pixel[COUNTER]));

myXTensDataTest := NORMALIZE(testData, featureCount,
                     TRANSFORM(Tensor.R4.TensData,
                       SELF.indexes := [LEFT.id, (COUNTER-1) DIV 28 + 1, (COUNTER-1) % 28 +1, 1],
                       SELF.value := LEFT.pixel[COUNTER]));											 
//OUTPUT(COUNT(myXTensData));											 
											 
// Each source record becomes 10 Y (in this case "label" tensor cells (one per class value)
// using One-Hot encoding.
// But only the record associated with the class (i.e. value of Y)
// will be 1.  The others will be zero.  Since the TensData format
// is sparse, we just skip the zero cells.
myYTensData := NORMALIZE(trainData, 10,
                     TRANSFORM(Tensor.R4.TensData,
                       SELF.indexes := [LEFT.id, COUNTER],
                       SELF.value := IF(LEFT.label != COUNTER - 1, SKIP, 1)));
											 
myYTensDataTest := NORMALIZE(testData, 10,
                     TRANSFORM(Tensor.R4.TensData,
                       SELF.indexes := [LEFT.id, COUNTER],
                       SELF.value := IF(LEFT.label != COUNTER - 1, SKIP, 1)));											 
											 
// Now we convert the Tensor Data to a Tensor dataset by calling MakeTensor()
myXTensor := Tensor.R4.MakeTensor([0,28,28,1], myXTensData);
myYTensor := Tensor.R4.MakeTensor([0,10], myYTensData);

myXTensorTest := Tensor.R4.MakeTensor([0,28,28,1], myXTensDataTest);
myYTensorTest := Tensor.R4.MakeTensor([0,10], myYTensDataTest);


instancesPerSlice := COUNT(myXTensor[1].densedata)/featureCount;
//OUTPUT(myXTensor, NAMED('SeeingTheSlices')); 
//OUTPUT(instancesPerSlice, NAMED('InstancesPerSlice'));

//OUTPUT(myXTensor, NAMED('x1'));
//OUTPUT(myYTensor, NAMED('y1'));

ldef := [	'''layers.Conv2D(32, kernel_size=(3, 3), activation="relu", input_shape=(28,28,1))''',
					'''layers.MaxPooling2D(pool_size=(2, 2))''',
					'''layers.Conv2D(64, kernel_size=(3, 3), activation="relu")''',
					'''layers.MaxPooling2D(pool_size=(2, 2))''',
					'''layers.Flatten()''',
					'''layers.Dropout(0.5)''',
					'''layers.Dense(10, activation="softmax")'''];		
					
					
large_ldef := [
        '''layers.Conv2D(32, kernel_size=(3, 3), activation="relu", input_shape=(28,28,1))''',
        '''layers.MaxPooling2D(pool_size=(2, 2))''',
        '''layers.Conv2D(128, kernel_size=(3, 3), activation="relu")''',
        '''layers.MaxPooling2D(pool_size=(2, 2))''',
        '''layers.Conv2D(2560, kernel_size=(3, 3), activation="relu")''',
        '''layers.MaxPooling2D(pool_size=(2, 2))''',
        '''layers.Dropout(0.25)''',
        '''layers.Flatten()''',
        '''layers.Dense(10240, activation="relu")''',
        '''layers.Dense(3850, activation="relu")''',
        '''layers.Dropout(0.5)''',
        '''layers.Dense(10, activation="softmax")'''];
					
compileDef := '''compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])''';

// GetSession must be called before any other functions
s := GNNI.GetSession();
// DefineModel is dependent on the Session
//   ldef contains the Python definition for each Keras layer
//   compileDef contains the Keras compile statement.

//mod := GNNI.DefineModel(s, large_ldef, compileDef);
mod := GNNI.DefineModel(s, ldef, compileDef);

// GetWeights returns the initialized weights that have been synchronized across all nodes.
wts := GNNI.GetWeights(mod);
OUTPUT(LIMIT(wts,100), NAMED('InitWeights'));

//mod2 := GNNI.NCCLFit(mod, myXTensor, myYTensor, batchSize := weightAggregateInterval, miniBatch := miniBatch, numEpochs := numEpochs);
mod2 := GNNI.Fit(mod, myXTensor, myYTensor, batchSize := batch , numEpochs := numEpochs);


//OUTPUT(mod2, NAMED('mod2_FIT'));

// GetLoss returns the average loss for the final training epoch
losses := GNNI.GetLoss(mod2);
OUTPUT(losses, NAMED('Losses'));

// EvaluateMod computes the loss, as well as any other metrics that were defined in the Keras
// compile line.
metrics := GNNI.EvaluateMod(mod2, myXTensorTest, myYTensorTest);

OUTPUT(metrics, NAMED('metrics'));











