﻿#option('outputLimit',2000);
#option('hthorMemoryLimit',10000);
IMPORT Python3 AS Python;

IMPORT $.GNN_ML_454 AS GNN;
//IMPORT $.GNN_ML_454_stock AS GNN;
//IMPORT $.GNN_ML_454_stock_wGPU AS GNN; 

IMPORT GNN.Tensor;
IMPORT GNN.Internal.Types AS iTypes;
IMPORT GNN.Types;
IMPORT GNN.GNNI;
IMPORT GNN.Utils;
IMPORT GNN.Internal AS Int;
IMPORT Std.System.Thorlib;
IMPORT mnist_as_real as mnist;
IMPORT STD;


//----------------------------------- Change Me -----------------------------------------------
dataSetNameString := 'MNIST60k';
modelType := 'CNN';
trainCount := 60000;
testCount := 8196;

numOfGPUs := 16; //16 for total system GPU count, just used for workunit naming
numOfPhysicalMachines := 2; //-1 makes TF to use CPU across the board, set the number of physical computer, assumes number of GPUs are equal accross machines and assumes number of Thor nodes is equal to total number of GPUs in system

aggregateInterval := 1; //how many times do you want the weights to be aggregated per epoch

mydropzoneIP := '10.0.0.17';
#WORKUNIT('name', dataSetNameString+' param ' + modelType +' '+numOfGPUs+' Thor/GPU Regular GNN looping v high interval modifed loop - ' + aggregateInterval + ' aggs per epoch'); 
//----------------------------------- Change Me -----------------------------------------------

WsTiming := RECORD
  UNSIGNED4 count;
  UNSIGNED4 duration;
  UNSIGNED4 max;
  STRING name{MAXLENGTH(64)};
END;


resultsRec := RECORD
//STRING experimentResult;
DATASET(Types.metrics) performanceMetrics;
//STRING ExperimentTime;
//STRING numPartitions;
INTEGER experimentNum;
END;


//numberOfExperiments := 5;
//numberOfEpochsPerExperiment := 1;

//aggregateInterval := 10; //how many times do you want the weights to be aggregated per epoch

// Test parameters



miniBatch := 128; //32 is Stock-GNN
oldBatchSize := 128;
numEpochs := 15;
featureCount := 784;
weightAggregateInterval := trainCount / (numOfGPUs * aggregateInterval); //old "batchSize"
numOfGPUperMachine := numOfGPUs / numOfPhysicalMachines;

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

ldef := [	'''layers.Conv2D(32, kernel_size=(3, 3), activation="relu", input_shape=(28,28,1))''',
					'''layers.MaxPooling2D(pool_size=(2, 2))''',
					'''layers.Conv2D(64, kernel_size=(3, 3), activation="relu")''',
					'''layers.MaxPooling2D(pool_size=(2, 2))''',
					'''layers.Flatten()''',
					'''layers.Dropout(0.5)''',
					'''layers.Dense(10, activation="softmax")''']; // 34,826 trainable parameters

//compileDef := '''compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])''';
compileDef := '''compile(loss="categorical_crossentropy", optimizer=tf.keras.optimizers.SGD(learning_rate=0.0005, momentum=0.1), metrics=["accuracy"])''';



//DATASET(Types.metrics) startTraining(INTEGER experimentNum) := FUNCTION
DATASET(resultsRec) startTraining(INTEGER experimentNum) := FUNCTION
s := GNNI.GetSession(numOfPhysicalMachines, experimentNum);
mod := GNNI.DefineModel(s, ldef, compileDef);
wts := GNNI.GetWeights(mod);
//OUTPUT(wts, NAMED('InitWeights'));

mod2 := GNNI.Fit(mod, myXTensor, myYTensor, batchSize := weightAggregateInterval, miniBatch := miniBatch, numEpochs := numEpochs);

losses := GNNI.GetLoss(mod2);
//OUTPUT(losses, NAMED('Losses'));

// EvaluateMod computes the loss, as well as any other metrics that were defined in the Keras
// compile line.
metrics := GNNI.EvaluateMod(mod2, myXTensorTest, myYTensorTest);


RETURN DATASET([{metrics, experimentNum}], resultsRec);
END;



//SEQUENTIAL(
//OUTPUT(startTraining(1),,'' + dataSetNameString + '_' + numOfGPUs + 'Node_1t_' + (string)trainCount + '::trial1',OVERWRITE), 
//OUTPUT(startTraining(2),,'' + dataSetNameString + '_' + numOfGPUs + 'Node_1t_' + (string)trainCount + '::trial2',OVERWRITE)
//);

SEQUENTIAL(
OUTPUT(startTraining(1),,'~thor::outdata1.csv',CSV(HEADING('',''), SEPARATOR(','), TERMINATOR('\n'))),
OUTPUT(startTraining(2),,'~thor::outdata2.csv',CSV(HEADING('',''), SEPARATOR(','), TERMINATOR('\n')))
);



//STD.File.DeSpray('' + dataSetNameString + '_' + numOfGPUs + 'Node_1t_' + (string)trainCount + '::trial1', mydropzoneIP, '/var/lib/HPCCSystems/mydropzone/' + dataSetNameString + '_' + numOfGPUs + 'Node_1t_' + (string)trainCount + '_'+aggregateInterval+'_aggs_per_epoch/trial1.csv',,,,TRUE);
//STD.File.DeSpray('' + dataSetNameString + '_' + numOfGPUs + 'Node_1t_' + (string)trainCount + '::trial2', mydropzoneIP, '/var/lib/HPCCSystems/mydropzone/' + dataSetNameString + '_' + numOfGPUs + 'Node_1t_' + (string)trainCount + '_'+aggregateInterval+'_aggs_per_epoch/trial1.csv',,,,TRUE);
