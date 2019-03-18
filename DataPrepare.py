import tensorflow as tf
import os
import re
model_dir="/Users/yeppy/facenet_model/v1"

files = os.listdir(model_dir)
meta_files = [s for s in files if s.endswith('.meta')]
meta_file = meta_files[0]

meta_files = [s for s in files if '.ckpt' in s]
max_step = -1

for f in files:
    step_str = re.match(r'(^model-[\w\- ]+.ckpt-(\d+))', f)
    if step_str is not None and len(step_str.groups())>=2:
        step = int(step_str.groups()[1])
        if step > max_step:
            max_step = step
            ckpt_file = step_str.groups()[0]
meta_file, ckpt_file
with tf.Graph().as_default():
    with tf.Session() as sess:
        saver = tf.train.import_meta_graph(os.path.join(model_dir, meta_file), input_map=None)
        saver.restore(tf.get_default_session(), os.path.join(model_dir, ckpt_file))
        output_path = "/Users/yeppy/output_model/1"
        builder = tf.saved_model.builder.SavedModelBuilder(output_path)
        # Get input and output tensors
        images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
        embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
        phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
        # Build the signature_def_map.
        prediction_signature = tf.saved_model.signature_def_utils.build_signature_def(inputs={'images': tf.saved_model.utils.build_tensor_info(images_placeholder),'phase': tf.saved_model.utils.build_tensor_info(phase_train_placeholder)},outputs={'embeddings': tf.saved_model.utils.build_tensor_info(embeddings)},method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME)
        legacy_init_op = tf.group(tf.tables_initializer(), name='legacy_init_op')
        sess.run(tf.global_variables_initializer())
        builder.add_meta_graph_and_variables(sess, [tf.saved_model.tag_constants.SERVING],signature_def_map={'calculate_embeddings':prediction_signature,},legacy_init_op=legacy_init_op)
        builder.save()



def freeze_graph(args):

    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        print(tf.global_variables())
        saver = tf.train.Saver(tf.global_variables())
        ckpt = tf.train.get_checkpoint_state(args.save_dir)
        print(ckpt.model_checkpoint_path)
    # We import the meta graph in the current default Graph
        saver = tf.train.import_meta_graph(ckpt.model_checkpoint_path + '.meta', clear_devices=True)
        # We restore the weights
        saver.restore(sess, ckpt.model_checkpoint_path)
        # We use a built-in TF helper to export variables to constants
        print(len([n.name for n in tf.get_default_graph().as_graph_def().node]))
        output_graph_def = tf.graph_util.convert_variables_to_constants(
            sess, # The session is used to retrieve the weights
            tf.get_default_graph().as_graph_def(), # The graph_def is used to retrieve the nodes 
            args.output_node_names.split(",") # The output node names are used to select the usefull nodes
        ) 
        # Finally we serialize and dump the output graph to the filesystem
        with tf.gfile.GFile(args.output_graph + "model.pb", "wb") as f:
            f.write(output_graph_def.SerializeToString())
        print("%d ops in the final graph." % len(output_graph_def.node))


saved_model_cli show --dir output_model/1


TESTDATA="$(pwd)"
docker run --name rest -dt -p 8501:8501 -v "$TESTDATA/output_model:/models/facenet" -e MODEL_NAME=facenet tensorflow/serving

curl http://localhost:8501/v1/models/facenet

TESTDATA="$(pwd)"
docker run -dt -p 8500:8500 -v "$TESTDATA/output_model:/models/facenet" -e MODEL_NAME=facenet tensorflow/serving


GET http://localhost:8500/v1/models/facenet/metadata

# 在服务器本机测试模型是否正常工作，这里需要注意，源码中half_plus_two的模型版本是00000123，但在访问时也必须输入v1而不是v000000123
curl -d '{"instances": [1.0, 2.0, 5.0]}' -X POST http://localhost:8501/v1/models/facenet:predict
#


curl get http://localhost:8501/v1/models/facenet/metadata




