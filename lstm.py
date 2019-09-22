import tensorflow as tf
import numpy as np
import time

NUM_LSTM=1
LSTM_UNIT=512
LENGTH=1000
INPUT_DIM=2*LSTM_UNIT

def build_graph():


    inputs = tf.placeholder(dtype=tf.float32,shape=[None,None,INPUT_DIM])
    lengths= tf.placeholder(dtype=tf.int64,shape=[None])


    x=inputs

    for i in range(NUM_LSTM):
        fw_cell = tf.contrib.rnn.LSTMBlockFusedCell(num_units=LSTM_UNIT,dtype=tf.float32)

        x_fw,_=fw_cell(x,dtype=tf.float32)

        x_reverse=tf.reverse_sequence(x,seq_lengths=lengths,seq_axis=1,batch_axis=0)
        bw_cell = tf.contrib.rnn.LSTMBlockFusedCell(num_units=LSTM_UNIT,dtype=tf.float32)

        x_bw,_=bw_cell(x_reverse,dtype=tf.float32)
        x_bw_reverse=tf.reverse_sequence(x_bw,seq_lengths=lengths,seq_axis=1,batch_axis=0)

        x=tf.concat([x_fw,x_bw_reverse],axis=-1)

    output=x

    tf.add_to_collection("inputs",inputs)
    tf.add_to_collection("inputs",lengths)
    tf.add_to_collection("outputs",output)



def run_graph():

    inputs=tf.get_collection("inputs")
    outputs=tf.get_collection("outputs")

    length=LENGTH
    x=np.random.rand(1,length,INPUT_DIM)
    length=np.array(length).reshape([-1])
    

    init_op=tf.initialize_all_variables()

    config = tf.ConfigProto(intra_op_parallelism_threads=1, 
                            inter_op_parallelism_threads=1,
                            allow_soft_placement=True)

    with tf.Session(config=config) as sess:
    #with tf.Session(config=config) as sess:
        sess.run(init_op)
        start_time=time.time()
        sess.run(outputs,feed_dict=dict(zip(inputs,[x,length])))
        print("warmup, time cost: {}".format(time.time()-start_time))

        for i in range(1,20):
            start_time=time.time()
            length=LENGTH+i
            x=np.random.rand(1,length,INPUT_DIM)
            length=np.array(length).reshape([-1])
            sess.run(outputs,feed_dict=dict(zip(inputs,[x,length])))
            print("step {}, time cost: {}".format(i,time.time()-start_time))



if __name__=="__main__":
    build_graph()
    run_graph()






