IMPORT Python3 AS Python;
IMPORT STD, $;
//IMPORT $.data_types as recType;
//IMPORT $.MNIST_train as worker;
#option('outputLimit',2000);

mnist_data_type := RECORD
 INTEGER1 label;
 DATA784 image;
END;

//First distribute the training data. Testing data is used for model evaluation and is completed
//on a single node, i.e. no need to distribute the testing data.
trainingData := DISTRIBUTE(CHOOSEN(DATASET('~mnist::train', mnist_data_type, THOR), 60000));
testingData := CHOOSEN(DATASET('~mnist::test', mnist_data_type, THOR), 10000);

OUTPUT(count(trainingData));