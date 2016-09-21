import tensorflow as tf

def read_and_decode_single_example(filename):
    # first construct a queue containing a list of filenames.
    # this lets a user split up there dataset in multiple files to keep
    # size down
    filename_queue = tf.train.string_input_producer([filename],
                                                    num_epochs=None)
    # Unlike the TFRecordWriter, the TFRecordReader is symbolic
    reader = tf.TFRecordReader()
    # One can read a single serialized example from a filename
    # serialized_example is a Tensor of type string.
    _, serialized_example = reader.read(filename_queue)
    # The serialized example is converted back to actual values.
    # One needs to describe the format of the objects to be returned
    features = tf.parse_single_example(
        serialized_example,
        features={
            # We know the length of both fields. If not the
            # tf.VarLenFeature could be used
            'img_feats': tf.FixedLenFeature([4096], tf.float32),
            'sentence': tf.VarLenFeature(tf.int64)
        })
    # now return the converted data
    img_feats = features['img_feats']
    sentence = features['sentence']
    return img_feats, sentence

if __name__ == "__main__":
  filename = "train.tfrec"
  print "importing from "+filename
  feats, sent = read_and_decode_single_example(filename)
  batched_data = tf.train.batch(
    [feats, sent], batch_size=128,
    capacity=2000,
    dynamic_pad=True)
  sess = tf.Session()
  init = tf.initialize_all_variables()
  sess.run(init)
  tf.train.start_queue_runners(sess)
  res = sess.run(batched_data)
