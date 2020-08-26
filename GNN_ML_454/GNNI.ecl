﻿/*##############################################################################
## HPCC SYSTEMS software Copyright (C) 2019 HPCC Systems.  All rights reserved.
############################################################################## */
IMPORT PYTHON3 AS PYTHON;
IMPORT $ AS GNN;
IMPORT GNN.Internal as int;
IMPORT GNN.Types;
IMPORT GNN.Internal.Types AS iTypes;
IMPORT GNN.Internal.Keras;
IMPORT GNN.Tensor;
IMPORT Std.System.Thorlib;
IMPORT Std.System.Log AS Syslog;
IMPORT ML_Core.Types AS mlTypes;
NumericField := mlTypes.NumericField;
kString := iTypes.kString;
kStrType := iTypes.kStrType;
initParms := iTypes.initParms;
t_Tensor := Tensor.R4.t_Tensor;
TensData := Tensor.R4.TensData;
nNodes := Thorlib.nodes();
nodeId := Thorlib.node();
FuncLayerDef := GNN.Types.FuncLayerDef;
/**
  * Generalized Neural Network Interface
  *
  * <p>Provides a generalized ECL interface to Keras over Tensorflow.  It supports the Keras Functional
  * API as well as the Sequential API.
  * <p>The Functional style allows models to be defined as a Directed Acyclic Graph (DAG), allowing branching
  * and merging among the layers. It is required for models that have multiple inputs or outputs.
  * For an annotated example using the Functional model with multiple inputs and outputs, see
  * Test/FuncModelTest.ecl
  * <p>The Sequential stle is a bit simpler, but requires a strict sequence of layers, each one feeding
  * into the next.  For an annotated example using the Sequential model, see Test/ClassicTest.ecl
  * <p>THEORY OF OPERATION
  * <p>A Keras / TF model is built on each HPCC node and training data is distributed among the nodes.
  * Distributed Synchronous Batch Gradient Descent is performed across nodes, synchronizing weights
  * periodically based on the 'batchSize' parameter.  Each function performs its work in a
  * distributed manner, using the built-in parallelization of HPCC.
  * <p>PROGRAM FLOW
  * <p>The flow of a program using this interface is as follows:<ul>
  * <li>GetSession() -- Initialized Keras / TF and returns a session token. This must be called
  * before any other operations.</li>
  * <li>DefineModel(...) -- Construct a Keras Sequential model by providing a list of Python statements, one
  * to construct each layer of the neural network as well as an optional compile definition statement.</li>
  * <li>DefineFuncModel(...) -- Construct a Keras Functional Model.  This allows construction of
  * complex models that cannot be expressed as a sequential set of layers. These include models with multiple
  * inputs or outputs, or models that use divergence or convergence among different layers.</li>
  * <li>CompileMod(...) -- (Optional) Pass the Keras model compilation statement and perform it on
  * the model.  This is only required if the compile definition was not provided in DefineModel (above).</li>
  * <li>Fit(...) -- Trains the model across the nodes of the cluster, based on provided training data.</li>
  * <li>EvaluateMod(...) -- (Optional) Evaluate the model against a set of data (typically your validation or test
  * data) and return the loss and any other metrics that were defined by your compileDef.</li>
  * <li>Predict(...) -- (Optional) Use the model to predict the output based on a provided set of input (X) data.</li>
  * <li>GetWeights(...) -- (Optional) Return the trained weights of all layers of the model.</li>
  * </ul>
  * <p>USE OF TENSORS
  * <p>GNNI uses Tensors (effectively N-dimensional array representations) to provide data and weights in
  * and out of Keras / TF.  See the included Tensor module for details.  These Tensor datasets provide an
  * efficient way to store, distribute, and process N-Dimensional data.  The data is packed into 'slices',
  * which can be either sparse or dense, for efficiency and scalability purposes.
  * <p>Tensors can be used to convey record-oriented information such as training data as well as block
  * oriented data like weights.  Both can be N-dimensional.  For record-oriented data, the first shape
  * component is 0 (unspecified) indicating that it can hold an arbitrary set of records.
  * <p>For models with multiple inputs or outputs, Tensor Lists are used (see Tensor.ecl for details),
  * with one Tensor per input or output.
  * <p>USE OF NumericField
  * <p>GNNI also provides a set of interfaces which take in and emit data as 2-dimensional NumericField
  * datasets (see ML_Core.Types.NumericField).  This is purely for convenience for applications that
  * don't require the N-Dimensional capabilities of the Tensor format.  Internally, these functions
  * translate the NumericField format into Tensors, and convert the output from Tensors to NumericField.
  * These functions have the same names as the tensor functions, but with NF appended to the name
  * (e.g. FitNF(...), PredictNF(...)).  Weights are always returned as Tensors, so there is no NF
  * version of GetWeights(...).
  * <p>SEQUENCING OF OPERATIONS
  * The Keras / Tensorflow operations take place under the hood from an ECL perspective.  Therefore
  * normal ECL data dependencies are not sufficient to ensure proper sequencing.  For this reason,
  * GNNI uses a  series of tokens passed from one call to the next to ensure the correct order of
  * command execution.  For example:<ul>
  * <li>GetSession() returns a session-token which must be passed to DefineModel()</li>
  * <li>Subsequent calls return a model-token which must be passed to the following call.  Each
  * call creates a new model token which becomes the input to the next call in sequence.</li>
  * <li>It is critical that this token passing is chained, or calls may occur out of order.
  * For example, Fit() could be called before DefineModel(), which would not produce good results.</li>
  * </ul>
  * <p>MULTIPLE MODEL SUPPORT
  * <p>GNNI supports multiple Keras models within the same work unit.  Multiple models are created
  * by using multiple calls to DefineModel() using the same sessionId.  The returned modelIds are
  * used in subsequent calls to discriminate between the models.  See Test/MultiModel.ecl for an
  * example.
  */
EXPORT GNNI := MODULE
  /**
    * Generate a sequential token.  By making this a python function,
    * we prevent the compiler from pre-determining the result, potentially
    * breaking the dependency chain.
    */
  SHARED UNSIGNED4 getToken(UNSIGNED4 lastToken) := EMBED(Python)
    return lastToken + 1
  ENDEMBED;
  /**
    * Obscure a value (from the ECL compiler) by returning it through a
    * python function so that the compiler can't determine that it is
    * a constant.
    */
  SHARED UNSIGNED4 obscure(UNSIGNED4 value) := EMBED(C++:action)
    return value;
  ENDEMBED;
  /**
    * Each node returns status as a kString.   Returns an error message
    * if there was at least 1 error, or if a reply was not received from
    * every node.  Otherwise returns blank string.
    */
  SHARED STRING reduceResults(DATASET(kString) results) := FUNCTION
    rr0 :=  results(LENGTH(text) > 0);
    rr1 := rr0[1].text;
    rr := IF(COUNT(results) != nNodes,
            '''Didn't recieve reply from all nodes: ''' + COUNT(results), rr1);
    return rr;
  END;

  // Number by which to multiply the Keras model id in order to generate
  // GNNI model ids.
  SHARED kerasIdFactor := 100000;
	
  /**
    * Initialize Keras on all nodes and return a "session" token to be used on the
    * next call to GNNI.
    * <p>This function must be called before any other use of GNNI.
    *
    * @return A session token (UNSIGNED4) to identify this session.
    */
  EXPORT UNSIGNED4 GetSession(INTEGER numPhysicalComps = -1, INTEGER experimentNum = -1) := FUNCTION
    initDat := DATASET(1, TRANSFORM(initParms,
                                      SELF.nodeId := nodeId,
                                      SELF.nNodes := nNodes,
                                      SELF.maxSliceSize := Tensor.MAX_SLICE), LOCAL);
    kstatus := ASSERT(Keras.Init(initDat, numPhysicalComps), LENGTH(text) = 0, 'GetSession Exception: ' + text, FAIL);
    status := reduceResults(kstatus);
    model := IF(LENGTH(status) = 0, getToken(0), 0);
    RETURN model;
 END;
  /**
    * Define a Keras / Tensorflow model using Keras sytax.  Optionally
    * also provide a "compile" line with the compilation parameters for the
    * model.
    * <p>If no compile line is provided (cdef), then the compile specification
    * can be provided in a subsequent call to CompileMod (below).
    * <p>The symbols "tf" (for tensorflow) and "layers" (for tf.keras.layers)
    * are available for use within the definition strings.
    * See GNN/Test/ClassicTest.ecl for an annotated example.
    * @param sess The session token from a previous call to GetSesion().
    * @param ldef A set of python strings as would be passed to Keras
    *         model.add().  Each string defines one layer of the model.
    * @param cdef (optional) A python string as would be passed to Keras model.compile(...).
    *         This line should begin with "compile".  Model is implicit here.
    * @return A model token to be used in subsequent GNNI calls.
    */
  EXPORT UNSIGNED4 DefineModel(UNSIGNED4 sess, SET OF STRING ldef, STRING cdef = '') := FUNCTION
    mdef1 := DATASET(COUNT(ldef), TRANSFORM(kString, SELF.typ := kStrType.layer,
                                            SELF.id  := COUNTER,
                                            SELF.text := ldef[COUNTER]));
    mdef2 := DATASET([{0, COUNT(ldef)+1, kStrType.compile, cdef}], kString);
    mdef := IF(LENGTH(cdef) > 0, mdef1 + mdef2, mdef1);
    mdefRepl0 := SORT(DISTRIBUTE(mdef, ALL), id, LOCAL);
		//mdefRepl0 := SORT(DISTRIBUTE(mdef, 1), id, LOCAL);//RK
    mdefRepl := PROJECT(NOCOMBINE(mdefRepl0), TRANSFORM(RECORDOF(LEFT), SELF.nodeId := nodeId, SELF := LEFT), LOCAL);
    kstatus := ASSERT(Keras.DefineModel(mdefRepl, sess), LENGTH(text) = 0, 'DefineModel Exception: ' + text);
    status := reduceResults(kstatus);
    // Extract the Keras modelId from the id field of the returned status.  Each node should have the
    // same model id since they are kept in sync.  So we just use the one from our own node.
    modelId := kstatus(nodeId = nodeId)[1].id;
    // We need to generate an GNNI modelId that encompasses both the sequence, and encodes
    // the Keras model id.
    // We multiply the modelId by kerasIdFactor and use that as the basis for our returned modelId token.
    // This allows up to kerasIdFactor operations on each model, which should be enough.
    modelBase := modelId * kerasIdFactor;
    model := IF(LENGTH(status) = 0, getToken(sess + modelBase), 0);
    RETURN model;
  END;
  /**
    * DefineFuncModel(...) -- Construct a Keras Functional Model.  This allows construction of
    * complex models that cannot be expressed as a sequential set of layers. These include models with multiple
    * inputs or outputs, or models that use divergence or convergence among different layers.
    * <p>Layers are connected together using the layerName and predecessor fields of the FuncLayerDef.
    * The inputs of a layer are connected to the predecessor layers in the order specified by the
    * set of names in the predecessor field.
    * <p>The inputs and outputs parameter specifies the names of the layers that form the input and
    * output of the model.
    * <p>This is similar to the Keras Functional API, except that the entire model is defined in one
    * call rather than assembled piecemeal as in the Functional API.  The same rules apply here as
    * for the Keras Functional API, and this should be a simple translation of any program using the
    * Functional API.
    * <p>For models with multiple inputs, input is specified as a list of tensors (see Tensor.ecl).
    * <p>For models with multiple outputs, output will be a list of tensors.
    * @param sess The session token from a previous call to GetSesion().
    * @param lDefs A series of layer definitions using the Types.FuncLayerDef format.
    * @param inputs A list of the names of the layers that represent the inputs to the model.
    * @param outputs A list of the names of the layers that represent the outputs of the model.
    * @param cdef (optional) A python string as would be passed to Keras model.compile(...).
    *         This line should begin with "compile".  Model is implicit here.
    * @see Types.FuncLayerDef
    */
  EXPORT UNSIGNED4 DefineFuncModel(UNSIGNED sess,
                                   DATASET(FuncLayerDef) lDefs,
                                   SET OF STRING inputs,
                                   SET OF STRING outputs,
                                   STRING cdef = '') := FUNCTION
    // Distribute the lDefs to all nodes to make sure that the model is defined on each node
    lDefsRepl := DISTRIBUTE(lDefs, ALL);
    kstatus := ASSERT(Keras.DefineFuncModel(lDefsRepl, sess, inputs, outputs, cdef), LENGTH(text) = 0, 'DefineFuncModel Exception: ' + text);
    status := reduceResults(kstatus);
    // Extract the Keras modelId from the id field of the returned status.  Each node should have the
    // same model id since they are kept in sync.  So we just use the one from our own node.
    modelId := kstatus(nodeId = nodeId)[1].id;
    // We need to generate an GNNI modelId that encompasses both the sequence, and encodes
    // the Keras model id.
    // We multiply the modelId by kerasIdFactor and use that as the basis for our returned modelId token.
    // This allows up to kerasIdFactor operations on each model, which should be enough.
    modelBase := modelId * kerasIdFactor;
    model := IF(LENGTH(status) = 0, getToken(sess + modelBase), 0);
    RETURN model;
  END;
  /**
    * Return a JSON representation of the Keras model.
    *
    * @param mod The model token as previously returned
    *         from DefineModel(...) above.
    * @return A JSON string representing the model definition.
    */
  EXPORT STRING ToJSON(UNSIGNED4 mod) := FUNCTION
    kModelId := mod DIV kerasIdFactor;
    results := Keras.ToJSON(DATASET([], kString), mod, kModelId);
    result := results[1].text;
    RETURN result;
  END;

  /**
    * Create a Keras model from previously saved JSON.
    * <p>Note that this call defines the model, but does not
    * restore the compile definition or the trained model weights.
    * CompileMod(...) should be called after this to define the
    * model compilation parameters.
    *
    * @param sess A session token previously returned from GetSession(..).
    * @param json A JSON string defining the model as previously
    *         returned from ToJSON(...).
    * @return A model token to be used in subsequent GNNI calls.
    */
  EXPORT UNSIGNED4 FromJSON(UNSIGNED4 sess, STRING json) := FUNCTION
    mdefRepl := DATASET(1, TRANSFORM(kString,
                                    SELF.id :=1,
                                    SELF.typ := kStrType.json,
                                    SELF.text := json), LOCAL);
    kstatus := ASSERT(Keras.FromJSON(mdefRepl, sess), LENGTH(text) = 0, 'FromJSON Exception: ' + text, FAIL);
    status := reduceResults(kstatus);
    // Extract the Keras modelId from the id field of the returned status.  Each node should have the
    // same model id since they are kept in sync.  So we just use the one from our own node.
    modelId := kstatus(nodeId = nodeId)[1].id;
    // We need to generate an GNNI modelId that encompasses both the sequence, and encodes
    // the Keras model id.
    // We multiply the modelId by kerasIdFactor and use that as the basis for our returned modelId token.
    // This allows up to kerasIdFactor operations on each model, which should be enough.
    modelBase := modelId * kerasIdFactor;
    model := IF(LENGTH(status) = 0, getToken(sess + modelBase), 0);
    RETURN model;
  END;
  /**
    * Compile a previously defined Keras model.
    *
    * <p>This is an optional call that can be used if you omit the
    * compileDef parameter during DefineModel(...) or if the model
    * was created via FromJSON(...).
    * <p>The compile string uses the same python syntax as using Keras'
    * model.compile(...).  Model is implied in this call, so the line
    * should begin with "compile".
    *
    * <p>The symbol "tf" (for tensorflow) is available for use within
    * the compile string.
    * <p>Example:<ul>
    * <li>'''compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])'''</li></ul>
    * <p>It is convenient to use the triple single quote(''') syntax as
    *     it allows strings to cross line boundaries, and allows
    *     special characters such as siepochNumngle or double quotes without
    *     escaping.
    * <p>There is no need to make this call if the compileDef was provided
    * in the DefineModel(...) call.
    * <p>The returned model token should be used in subsequent calls to
    * GNNI.
    *
    * @param model A model token as returned from DefineModel(...) or
    *         FromJSON(...).
    * @param compileStr A python formatted string defining the Keras
    *         "compile" call and its parameters.
    * @return A new model token that should be used in subsequent GNNI
    *               calls.
    */
  EXPORT UNSIGNED4 CompileMod(UNSIGNED model, STRING compileStr) := FUNCTION
    mdefRepl := DATASET(1, TRANSFORM(kString,
                                    SELF.id :=1,
                                    SELF.typ := kStrType.compile,
                                    SELF.text := compileStr), LOCAL);
    kModelId := model DIV kerasIdFactor;
    kstatus := ASSERT(Keras.CompileMod(mdefRepl, model, kModelId), LENGTH(text) = 0, 'CompileMod Exception: ' + text, FAIL);
    status := reduceResults(kstatus);
    RETURN getToken(model);
  END;
  /**
    * Return the weights currently associated with the model.
    *
    * <p>The weights are returned as a Tensor List containing the
    * weights for each Keras layer as a separate Tensor.
    * <p>The weights from a given layer can be extracted by
    * simply filtering on the work-item (wi).  The first layer
    * will use wi = 1, and the Nth layer uses wi = N.
    * <p>This call is typically made after training the model
    * via Fit(...), but can also be called before Fit(...) to
    * retrieve the initial weights.
    *
    * @param model The model token as returned from DefineModel(...),
    *         CompileMod(...), or Fit(...).
    * @return A t_Tensor dataset representing the weights as a
    *     list of Tensors.
    */
  EXPORT DATASET(t_Tensor) GetWeights(UNSIGNED4 model) := FUNCTION
    // Get the weights from a single node.  Note that weights should
    // be the same on all nodes since they are automatically
    // synchronized between nodes.
    kModelId := model DIV kerasIdFactor;
    dummy := DATASET(1, TRANSFORM(kString, SELF.id := 1, SELF.typ := kStrType.None, SELF.text := ''), LOCAL);
    weights := Keras.GetWeights(dummy, model, kModelId);
    RETURN weights(nodeId=0);
  END;

  /**
    * Set the weights of the model from a list of Tensors.
    *
    * <p>Typically, the weights to be set were originally obtained
    * using GetWeights(...) above.  They must be of the same number
    * and shape as would be returned from GetWeights(...).
    * <p>These will contain one Tensor for each defined Keras layer.
    * The shape of each tensor is determined by the definition of
    * the layer.
    *
    * @param model The model token from the previous step.
    * @param weights The Tensor List containing the desired weights.
    * @return A new model token to be used in subsequent calls.
    */
  EXPORT UNSIGNED4 SetWeights(UNSIGNED4 model, DATASET(t_Tensor) weights) := FUNCTION
    kModelId := model DIV kerasIdFactor;
    weightsD := Tensor.R4.replicate(weights);
    kstatus := ASSERT(Keras.SetWeights(weightsD, model, kModelId), LENGTH(text) = 0, 'SetWeights Exception: ' + text, FAIL);
    status := reduceResults(kstatus);
    mod :=  IF(LENGTH(status) = 0, getToken(model), 0);
    RETURN mod;
  END;
  /**
    * Get the accumulated average loss for the latest epoch.
    * <p>This represents the average per sample loss.
    * @param model The model token as returned from Fit(...).
    * @return The average loss.
    */
  EXPORT REAL GetLoss(UNSIGNED4 model) := FUNCTION
    kModelId := model DIV kerasIdFactor;
    dummy := DATASET(1, TRANSFORM(kString, SELF.id := 1, SELF.typ := kStrType.None, SELF.text := ''), LOCAL);
    trainLosses := Keras.GetLoss(dummy, model, kModelId);
    // Each node provides the average loss across samples in the epoch.
    // We return the average of those averages.
    trainLoss := AVE(trainLosses, loss);
    RETURN trainLoss;
  END;
  /**
    *  Take in a set of weight slices to be updated on this node plus a set of updates (deltas)
    *  to apply to those weights (multiple slices).
    *  Roll up the results by adding the base weights and updates for each slice to produce
    *  a new set of weights for each slice.
    *  Then replicate the weight slices back to all nodes for the next round of processing.
    *  Note: There will typically be one update for each node, plus the shared (i.e. replicated)
    *  base-weights.  Before calling this rollup, update slices should be distributed by sliceId
    *  and sorted (locally) by sliceId.  At the end of this rollup, the updates have been
    *  applied to produce a new set of weights that are replicated to all nodes (i.e.
    *  synchronized.  Sort is always by sliceId locally.
    */
  SHARED DATASET(t_Tensor) rollUpdates(DATASET(t_Tensor) inWeights, DATASET(t_Tensor) updates) := FUNCTION
    combined := SORT(inWeights+updates, wi, sliceId, LOCAL);
    t_Tensor doRollup(t_Tensor l, t_Tensor r) := TRANSFORM
      sumT := Tensor.R4.addSlices(l, r);
      SELF.denseData := sumT.denseData;
      SELF.sparseData := sumT.sparseData;
      SELF := l;
    END;
    outWeights0 := ROLLUP(combined, doRollup(LEFT, RIGHT), wi, sliceId, LOCAL);
    outWeights := Tensor.R4.Replicate(outWeights0);
    RETURN outWeights;
  END;
  /**
    * Train the model using synchronous batch distributed gradient descent.
    * <p>The X tensor represents the independent training data and the Y
    * tensor represents the dependent training data.
    * <p>Both X and Y tensors
    * should be record-oriented tensors, indicated by a first shape
    * component of zero.  These must also be distributed (not replicated)
    * tensors.
    * <p>BatchSize defines how many observations are processed on each node
    * before weights are re-synchronized.  There is an interaction between
    * the number of nodes in the cluster, the batchSize, and the complexity
    * of the model.  A larger batch size will process epochs faster, but
    * the loss reduction may be less per epoch.  As the number of nodes
    * is increased, a smaller batchSize may be required.  The default
    * batchSize of 100 is a good starting point, but may require tuning to
    * increase performance or improve convergence (i.e. loss reduction).
    * Final loss should be used to assess the fit, rather than number of
    * epochs trained.  For example, for a given neural network, a loss of
    * .2 may be the optimal tradeoff between underfit and overfit.  In that case
    * the network should be trained to that level, adjusting number of epochs
    * and batchSize to reach that level.
    *
    * @param model The model token from the previous GNNI call.
    * @param x The independent training data tensor.
    * @param y The dependent training data tensor.
    * @param batchSize The number of records to process on each node before
    *         re-synchronizing weights across nodes.
    * @param numEpochs The number of times to iterate over the full training
    *         set.
    * @return A new model token for use with subsequent GNNI calls.
    */
  EXPORT UNSIGNED4 Fit(UNSIGNED4 model,
                      DATASET(t_Tensor) x,
                      DATASET(t_Tensor) y,
                      UNSIGNED4 batchSize = 100, //RK - now this should be the weight aggregate interval size.
                      UNSIGNED4 numEpochs = 1,
											UNSIGNED4 miniBatch = 32 //RK -  mini batch is the keras batch size. GNN batch should be larger than actual batch for performance reasons
											) := FUNCTION
    kModelId := model DIV kerasIdFactor;
    // Get the initial weights to use
    initWts0 := GetWeights(model);
    // We get the weights from the first node and then copy them to all nodes
    // so that everybody starts with the same weights
    initWts := Tensor.R4.Replicate(initWts0);
    // Align the X and Y tensor lists so that we will get the corresponding records on the same nodes
    // for each input and output tensor.
    maxInputWi := MAX(x, wi);
    // Change the wi's for outputs (y) so that they are after the input wi's
    y1 := PROJECT(y, TRANSFORM(RECORDOF(LEFT), SELF.wi := LEFT.wi + maxInputWi, SELF := LEFT), LOCAL);
    aligned := Tensor.R4.AlignTensors(x + y1);
    // Now change the Y's wi back to the original numbers
    xAl := aligned(wi <= maxInputWi);
    yAl := PROJECT(aligned(wi > maxInputWi), TRANSFORM(RECORDOF(LEFT), SELF.wi := LEFT.wi - maxInputWi, SELF := LEFT), LOCAL);
    totalRecords := Tensor.R4.GetRecordCount(yAl);
    batchesPerEpoch := ROUNDUP(totalRecords / nNodes / batchSize);
    DATASET(t_Tensor) doEpoch(DATASET(t_Tensor) wts1, UNSIGNED epochNum) := FUNCTION
      DATASET(t_Tensor) doBatch(DATASET(t_Tensor) wts2, UNSIGNED batchNum) := FUNCTION
        // Train the model and Get the weight changes from each node
        batchPos := (batchNum-1) * batchSize + 1;
        xBatch := int.TensExtract(xAl, batchPos, batchSize); //this is where the big bottleneck is, from Roger
        yBatch := int.TensExtract(yAl, batchPos, batchSize);
        wtChanges0 := IF(EXISTS(yBatch), Keras.FitBatch(wts2, xBatch, yBatch, obscure(model), epochNum, kModelId, miniBatch), DATASET([], t_Tensor));
				rkLog := Syslog.addWorkunitInformation('RKLOG - Keras.FitBatch iteration: ' + batchNum + ', Epoch: ' + epochNum + ', OldBatchSize(): ' + (string)batchSize);
				//rkLog := OUTPUT(xBatch);
        // Move all the changes for a given wi and slice to the same node.  Each
        // node has a set of wi/sliceIds to roll up.  Note that the original
        // weights are already replicated to all nodes.
        wtChanges := DISTRIBUTE(wtChanges0, wi + sliceId);
        // Sum up the original weights (de-replicated) and all changes for each wi and slice
        newWts := rollUpdates(wts2((wi + sliceId) % nNodes = nodeId), wtChanges);
        // Note: newWts have been replicated to all nodes by rollUpdates.
        // We use obscure to prevent the ECL compiler from treating GetLoss as a
        // constant
        batchLoss := IF(EXISTS(newWts), GetLoss(model + (batchesPerEpoch * (epochNum-1)) + batchNum), 1.0);
        logProgress2 := Syslog.addWorkunitInformation('Training Status (2): ModelId = ' +
                kModelId + ', Epoch = ' + epochNum + ', Batch = ' + batchNum + ', Loss = ' + batchLoss);
        RETURN WHEN(newWts, rkLog);
      END;
      epochWts := LOOP(wts1, batchesPerEpoch, doBatch(ROWS(LEFT), COUNTER));
      epochLoss := IF(EXISTS(epochWts), GetLoss(model + (batchesPerEpoch * (epochNum-1))), 1.0);
      //epochLoss := IF(EXISTS(epochWts), GetLoss(obscure(model)), 1.0);
      logProgress := Syslog.addWorkunitInformation('Training Status: ModelId = ' +
                      kModelId + ', Epoch = ' + epochNum + ', Loss = ' + epochLoss);
      RETURN WHEN(epochWts, logProgress);
    END;
    finalWts := LOOP(initWts, numEpochs, doEpoch(ROWS(LEFT), COUNTER));
    RETURN IF(EXISTS(finalWts), getToken(model + numEpochs * numEpochs), 0);
  END; // Fit
	
	//RK - this NCCL Fit function is a modification of the FIT above. It is to be used with the coresponding GNN.keras changes too
	// this will allow for data parallelism accross mutliple GPUs and use NCCL (direct GPU communication) to aggregate weights
	// Only applicable when with an NVIDIA GPU count >=2 
	// Use ECL to aggregate weights every epoch and NCCL to aggregate weights more often between the two. Don't need to agregate weights
	// via ECL if only using GPU nodes on one machine, the weight updates via NCCL will acomplish the same goal
	// Thus, use batch size such that th
	EXPORT UNSIGNED4 NCCLFit(UNSIGNED4 model,
                      DATASET(t_Tensor) x,
                      DATASET(t_Tensor) y,
                      UNSIGNED4 batchSize = 100, //RK - now this should be the weight aggregate interval size.
                      UNSIGNED4 numEpochs = 1,
											UNSIGNED4 miniBatch = 32 //RK -  mini batch is the keras batch size. GNN batch should be larger than actual batch for performance reasons
											) := FUNCTION
    kModelId := model DIV kerasIdFactor;
    // Get the initial weights to use
    initWts0 := GetWeights(model);
    // We get the weights from the first node and then copy them to all nodes
    // so that everybody starts with the same weights
    initWts := Tensor.R4.Replicate(initWts0);
    // Align the X and Y tensor lists so that we will get the corresponding records on the same nodes
    // for each input and output tensor.
    maxInputWi := MAX(x, wi);
    // Change the wi's for outputs (y) so that they are after the input wi's
    y1 := PROJECT(y, TRANSFORM(RECORDOF(LEFT), SELF.wi := LEFT.wi + maxInputWi, SELF := LEFT), LOCAL);
    aligned := Tensor.R4.AlignTensors(x + y1);
    // Now change the Y's wi back to the original numbers
    xAl := aligned(wi <= maxInputWi);
    yAl := PROJECT(aligned(wi > maxInputWi), TRANSFORM(RECORDOF(LEFT), SELF.wi := LEFT.wi - maxInputWi, SELF := LEFT), LOCAL);
    totalRecords := Tensor.R4.GetRecordCount(yAl);
    batchesPerEpoch := ROUNDUP(totalRecords / nNodes / batchSize);
    DATASET(t_Tensor) doEpoch(DATASET(t_Tensor) wts1, UNSIGNED epochNum) := FUNCTION
      DATASET(t_Tensor) doBatch(DATASET(t_Tensor) wts2, UNSIGNED batchNum) := FUNCTION
        // Train the model and Get the weight changes from each node
        batchPos := (batchNum-1) * batchSize + 1;
        xBatch := int.TensExtract(xAl, batchPos, batchSize);
        yBatch := int.TensExtract(yAl, batchPos, batchSize);
        wtChanges0 := IF(EXISTS(yBatch), Keras.FitBatchNCCL(wts2, xBatch, yBatch, obscure(model), epochNum, kModelId, miniBatch), DATASET([], t_Tensor));
				rkLog := Syslog.addWorkunitInformation('RKLOG - Keras.FitBatch iteration: ' + batchNum + ', Epoch: ' + epochNum + ', OldBatchSize(Batch Size per Computational Node): ' + batchSize);
        // Move all the changes for a given wi and slice to the same node.  Each
        // node has a set of wi/sliceIds to roll up.  Note that the original
        // weights are already replicated to all nodes.
        wtChanges := DISTRIBUTE(wtChanges0, wi + sliceId);
        // Sum up the original weights (de-replicated) and all changes for each wi and slice
        newWts := rollUpdates(wts2((wi + sliceId) % nNodes = nodeId), wtChanges);
        // Note: newWts have been replicated to all nodes by rollUpdates.
        // We use obscure to prevent the ECL compiler from treating GetLoss as a
        // constant
        batchLoss := IF(EXISTS(newWts), GetLoss(model + (batchesPerEpoch * (epochNum-1)) + batchNum), 1.0);
        logProgress2 := Syslog.addWorkunitInformation('Training Status (2): ModelId = ' +
                kModelId + ', Epoch = ' + epochNum + ', Batch = ' + batchNum + ', Loss = ' + batchLoss);
        RETURN WHEN(newWts, rkLog);
      END;
      epochWts := LOOP(wts1, batchesPerEpoch, doBatch(ROWS(LEFT), COUNTER));
      epochLoss := IF(EXISTS(epochWts), GetLoss(model + (batchesPerEpoch * (epochNum-1))), 1.0);
      //epochLoss := IF(EXISTS(epochWts), GetLoss(obscure(model)), 1.0);
      logProgress := Syslog.addWorkunitInformation('Training Status: ModelId = ' +
                      kModelId + ', Epoch = ' + epochNum + ', Loss = ' + epochLoss);
      RETURN WHEN(epochWts, logProgress);
    END;
    finalWts := LOOP(initWts, numEpochs, doEpoch(ROWS(LEFT), COUNTER));
    RETURN IF(EXISTS(finalWts), getToken(model + numEpochs * numEpochs), 0);
  END; // NCCL FIT
	
	
  /**
    * Determine the loss and other metrics in order to evaluate
    * the model.
    * <p>Returns a set of metrics including loss and any other metrics
    * that were defined in the compile definition for a set of provided
    * test data.
    * <p>Both X and Y tensors
    * should be record-oriented tensors, indicated by a first shape
    * component of zero.  These must also be distributed (not replicated)
    * tensors.
    * <p>This is typically used after training the model, using a segregated
    * set of test data, in order to determine the "out of sample" performance
    * (i.e. performance on data outside of the training set).
    * @param model The model token from the previous GNNI call (e.g. Fit).
    * @param x The independent test data tensor.
    * @param y The dependent test data tensor.
    * @return A dataset of metrics indicating the performance of the model.
    * @see Types.metrics
    */
  EXPORT DATASET(Types.metrics) EvaluateMod(UNSIGNED4 model,
                      DATASET(t_Tensor) x,
                      DATASET(t_Tensor) y) := FUNCTION
    kModelId := model DIV kerasIdFactor;
    // Align the X and Y tensor lists so that we will get the corresponding records on the same nodes
    // for each input and output tensor.
    maxInputWi := MAX(x, wi);
    // Change the wi's for outputs (y) so that they are after the input wi's
    y1 := PROJECT(y, TRANSFORM(RECORDOF(LEFT), SELF.wi := LEFT.wi + maxInputWi, SELF := LEFT), LOCAL);
    aligned := Tensor.R4.AlignTensors(x + y1);
    // Now change the Y's wi back to the original number
    xAl := aligned(wi <= maxInputWi);
    yAl := PROJECT(aligned(wi > maxInputWi), TRANSFORM(RECORDOF(LEFT), SELF.wi := LEFT.wi - maxInputWi, SELF := LEFT), LOCAL);
    m0 := Keras.Evaluate(xAl, yAl, model, kModelId);
    m1 := DISTRIBUTE(m0, metricId);
    m2 := TABLE(m1,
                {metricId, metricName, avgVal := AVE(GROUP, value)},
                metricId, metricName, LOCAL);
    metrics := PROJECT(m2, TRANSFORM(Types.metrics,
                SELF.value := LEFT.avgVal,
                SELF := LEFT), LOCAL);
    RETURN metrics;
  END;
	
	EXPORT DATASET(Types.metrics) RocAucScore(UNSIGNED4 model,
                      DATASET(t_Tensor) x,
                      DATASET(t_Tensor) y) := FUNCTION
    kModelId := model DIV kerasIdFactor;
    // Align the X and Y tensor lists so that we will get the corresponding records on the same nodes
    // for each input and output tensor.
    maxInputWi := MAX(x, wi);
    // Change the wi's for outputs (y) so that they are after the input wi's
    y1 := PROJECT(y, TRANSFORM(RECORDOF(LEFT), SELF.wi := LEFT.wi + maxInputWi, SELF := LEFT), LOCAL);
    aligned := Tensor.R4.AlignTensors(x + y1);
    // Now change the Y's wi back to the original number
    xAl := aligned(wi <= maxInputWi);
    yAl := PROJECT(aligned(wi > maxInputWi), TRANSFORM(RECORDOF(LEFT), SELF.wi := LEFT.wi - maxInputWi, SELF := LEFT), LOCAL);
    m0 := Keras.RocAucScore(xAl, yAl, model, kModelId);
    m1 := DISTRIBUTE(m0, metricId);
    m2 := TABLE(m1,
                {metricId, metricName, avgVal := AVE(GROUP, value)},
                metricId, metricName, LOCAL);
    metrics := PROJECT(m2, TRANSFORM(Types.metrics,
                SELF.value := LEFT.avgVal,
                SELF := LEFT), LOCAL);
    RETURN metrics;
  END;
	
	
  /**
    * Predict the results using the trained model.
    * <p>The X tensor represents the independent (input) data
    * for the neural network and the output is returned as
    * a tensor.  Input and output will be Tensor Lists if
    * there is more than one input or output tensor for the NN.
    * <p>The X tensor
    * should be a record-oriented tensor, indicated by a first shape
    * component of zero.  It must also be distributed (not replicated)
    * tensor.
    * @param model A model token as returned from the previous GNNI call
    *           (e.g. Fit).
    * @param x The independent (i.e. input) data tensor.
    * @return The output predicted by the model as a record-oriented
    *       tensor.
    */
  EXPORT DATASET(t_Tensor) Predict(UNSIGNED4 model, DATASET(t_Tensor) x) := FUNCTION
    kModelId := model DIV kerasIdFactor;
    // Align all of the X tensors (in case of multi tensor inputs)
    maxInputWi := MAX(x, wi); // The number of tensors in the input
    aligned := Tensor.R4.AlignTensors(x);
    xAl := IF(maxInputWi > 1, aligned, x); // Only align if multiple tensors in input
    pred := Keras.Predict(xAl, model, kModelId);
    return pred;
  END;
  /**
    * @nodoc
    * Shutdown Keras / Tensorflow and free up any allocated memory.
    *
    * This function is not required at this time but is here for future
    * use.
    * @param model A model token as returned from a previous GNNI call.
    * @return A new model token.
    */
  EXPORT UNSIGNED4 Shutdown(UNSIGNED4 model) := FUNCTION
    dummyDat := DATASET(1, TRANSFORM(initParms,
                                      SELF.nodeId := nodeId,
                                      SELF.nNodes := nNodes,
                                      SELF.maxSliceSize := Tensor.MAX_SLICE), LOCAL);
    dummy := DATASET(1, TRANSFORM(kString, SELF.id := 1, SELF.typ := kStrType.None, SELF.text := ''), LOCAL);
    kstatus := ASSERT(Keras.Shutdown(dummy, model), LENGTH(text) = 0, 'Shutdown Exception: ' + text, FAIL);
    status := reduceResults(kstatus);
    RETURN IF(LENGTH(status) = 0, getToken(model), 0);
  END;
  /**
    * Convert a NumericField matrix dataset to Tensor format.
    */
  SHARED DATASET(t_Tensor) NF2Tensor(DATASET(NumericField) nf) := FUNCTION
    tensDat := PROJECT(nf, TRANSFORM(TensData,
                              SELF.indexes := [LEFT.id, LEFT.number],
                              SELF := LEFT), LOCAL);
    maxNumber := MAX(nf, number);
    tens := Tensor.R4.MakeTensor([0,maxNumber], tensDat);
    RETURN tens;
  END;
  /**
    * Convert a 2-dimensional Tensor to a NumericField matrix dataset.
    */
  SHARED DATASET(NumericField) Tensor2NF(DATASET(t_Tensor) tens) := FUNCTION
    td := Tensor.R4.GetData(tens);
    nf := PROJECT(td, TRANSFORM(NumericField,
                                  SELF.wi := 1, // To do -- add wi to TensData
                                  SELF.id := LEFT.indexes[1],
                                  SELF.number := LEFT.indexes[2],
                                  SELF := LEFT), LOCAL);
    RETURN nf;
  END;
  /**
    * Fit a model with 2 dimensional input and output using NumericField
    * matrices.
    * <p>This is a NumericField wrapper around the Fit function.
    * See Fit (above) for details.
    * @param model The model token from the previous GNNI call.
    * @param x The independent training data.
    * @param y The dependent training data.
    * @param batchSize The number of records to process on each node before
    *         re-synchronizing weights across nodes.
    * @param numEpochs The number of times to iterate over the full training
    *         set.
    * @return A new model token for use with subsequent GNNI calls.
    * @see ML_Core.Types.NumericField
    */
  EXPORT UNSIGNED4 FitNF(UNSIGNED4 model,
                    DATASET(NumericField) x,
                    DATASET(NumericField) y,
                    UNSIGNED4 batchSize = 100,
                    UNSIGNED4 numEpochs = 1) := FUNCTION
    xT := NF2Tensor(x);
    yT := NF2Tensor(y);
    RETURN Fit(model, xT, yT, batchSize, numEpochs);
  END;
  /**
    * Evaluate a model with 2 dimensional input and output using NumericField
    * matrices.
    * <p>This is a NumericField wrapper around the EvaluateMod function.
    * See EvaluateMod (above) for details.
    * @param model The model token from the previous GNNI call.
    * @param x The independent test data.
    * @param y The dependent test data.
    * @return A dataset of metrics indicating the performance of the model.
    * @see Types.metrics
    * @see ML_Core.Types.NumericField
    */
  EXPORT DATASET(Types.metrics) EvaluateNF(UNSIGNED4 model,
                      DATASET(NumericField) x,
                      DATASET(NumericField) y) := FUNCTION
    xT := NF2Tensor(x);
    yT := NF2Tensor(y);
    RETURN EvaluateMod(model, xT, yT);
  END;
  /**
    * Predict the results for a model with 2 dimensional input
    * and output using NumericField matrixes for input and
    * output.
    * <p>This a a NumericField wrapper around the Predict function.
    * See Predict (above) for details.
    *
    * @param model A model token as returned from the previous GNNI call
    *           (e.g. Fit).
    * @param x The independent (i.e. input) data NumericField matrix.
    * @return The output predicted by the model as a NumericField matrix.
    * @see ML_Core.Types.NumericField
    */
  EXPORT DATASET(NumericField) PredictNF(UNSIGNED4 model, DATASET(NumericField) x) := FUNCTION
    xT := NF2Tensor(x);
    td := Predict(model, xT);
    nf := Tensor2NF(td);
    RETURN nf;
  END;
END; // GNNI