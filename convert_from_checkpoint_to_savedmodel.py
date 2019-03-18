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
