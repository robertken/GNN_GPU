IMPORT Python3 as python;

globalscope := 'scope.ecl';

returnType := RECORD
STRING out;
END;

//STREAMED DATASET(kString) DefineModel(STREAMED DATASET(kString) mdef, UNSIGNED4 seqId) := EMBED(Python: globalscope(globalScope), persist('query'), activity)
DATASET(returnType) DefineModel() := EMBED(Python: globalscope(globalScope), persist('query'), activity)
    import traceback as tb
    #Swap the next three lines, temporary TF2 workaround
    import tensorflow as tf
    import os
    #os.environ["CUDA_VISIBLE_DEVICES"]="1" 
    #tf.compat.v1.disable_v2_behavior()
    from tensorflow.keras import layers
    global nextModId
    try:
      # Allocate a new modelId
      # Make sure we do it atomically to avoid conflict with
      # another model running on another thread
      #threadlock.acquire()
      #modId = nextModId
      #nextModId += 1
      #threadlock.release()
      # Create a new keras / tensorflow context.  It sometimes gets lost between calls,
      # so we explicitly restore it before each call that uses it.
      # Note that for each model, we create a new session and new graph under the hood.
      # The graph is stored within the session, so only the session and model are stored,
      # both by model id.
      graph = tf.Graph()
      with tf.device('/device:GPU:0'):
        with graph.as_default():
          config = tf.ConfigProto()
          config.intra_op_parallelism_threads = 1
          config.inter_op_parallelism_threads= 1
          tfSession = tf.Session(config=config)
          #devices = tfSession.list_devices()
          with tfSession.as_default():
            mod = tf.keras.Sequential()
      return [str(mod)]
    except tf.errors.InvalidArgumentError as e:
      # We had an error.  Format the exception and return it in the kString
      return ['error occured: {}'.format(e)]
ENDEMBED; // DefineModel

OUTPUT(DefineModel());
