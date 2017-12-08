from deepspeech.utils import audioToInputVector
from DeepSpeech import create_inference_graph, initialize_globals
import tensorflow as tf
import scipy.io.wavfile as wav

model_path = "./data/models/output_graph.pb"


def load_graph(frozen_graph_filename):
    # We load the protobuf file from the disk and parse it to retrieve the
    # unserialized graph_def
    with tf.gfile.GFile(frozen_graph_filename, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    # Then, we import the graph_def into a new Graph and returns it
    with tf.Graph().as_default() as graph:
        # The name var will prefix every op/nodes in your graph
        # Since we load everything in a new graph, this is not needed
        tf.import_graph_def(graph_def, name="prefix")
    return graph


def infer_audio(input_file_path):
    initialize_globals()
    n_input = 26
    n_context = 9

    graph = load_graph(model_path)

    # We access the input and output nodes
    inp = graph.get_tensor_by_name('prefix/input_node:0')
    inp_len = graph.get_tensor_by_name('prefix/input_lengths:0')
    logits = graph.get_tensor_by_name('prefix/logits:0')
    op = graph.get_tensor_by_name('prefix/output_node:0')

    with tf.Session(graph=graph) as session:
        fs, audio = wav.read(input_file_path)
        mfcc = audioToInputVector(audio, fs, n_input, n_context)

        output = session.run(logits, feed_dict={
            inp: [mfcc],
            inp_len: [len(mfcc)],
        })

    return output
