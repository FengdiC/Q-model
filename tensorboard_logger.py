import tensorflow as tf
import time
import json

#Tensorboard instruction:
#tensorboard --logdir=log --port=6006
#If results from remote, download log directory
#go to localhost:6006
class tensorflowboard_logger:
    def __init__(self, figure_dir, sess, args):
        self.scalar = tf.placeholder(tf.float32, shape=[])
        print("Writing logs to ", figure_dir + "/" + str(int(time.time())))
        self.sess = sess
        self.directory = figure_dir + "/" + str(int(time.time()))
        self.writer = tf.summary.FileWriter(self.directory, graph=sess.graph, flush_secs=30)
        self.logged_scalar_dict = {}

        #dump the initial settings in the directory as well ...
        jsonStr = json.dumps(args.__dict__)
        file = open(self.directory + "/args.json", "w")
        file.write(jsonStr)
        file.close()


    def log_scalar(self, scalar_name, scalar_value, step):
        if not scalar_name in self.logged_scalar_dict:
            self.logged_scalar_dict[scalar_name] = tf.summary.scalar(name=scalar_name, tensor=self.scalar)
        scalar_summary = self.logged_scalar_dict[scalar_name]
        summary = self.sess.run(scalar_summary, feed_dict={self.scalar: scalar_value})
        self.writer.add_summary(summary, step)




