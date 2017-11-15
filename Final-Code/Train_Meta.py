import os
import matplotlib.pyplot as plt
from pylab import savefig
from timeit import default_timer as timer
import numpy as np
import tensorflow as tf

import MetaLearner
import util

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

flags = tf.flags
FLAGS = flags.FLAGS
flags.DEFINE_integer("num_epochs", 70, "Number of training epochs.") # 10000
flags.DEFINE_integer("log_period", 10, "Log period.") # 100
flags.DEFINE_integer("evaluation_period", 20, "Evaluation period.")#1000
flags.DEFINE_integer("evaluation_epochs", 20, "Number of evaluation epochs.")

flags.DEFINE_string("problem", "cifar10", "Type of problem.")
flags.DEFINE_integer("num_steps", 100,
                     "Number of optimization steps per epoch.") # 100
flags.DEFINE_integer("unroll_length", 20, "Meta-optimizer unroll length.")
flags.DEFINE_integer('num_layer',2,"Number of LSTM layer")
flags.DEFINE_integer('hidden_size',20,"Number of hidden layer in each LSTM")
flags.DEFINE_float('lr',0.001,"Initial learning rate")

logs_path = '/Users/kalyanb/PycharmProjects/MetaLearning/MetaLog/'
save_path = '/Users/kalyanb/PycharmProjects/MetaLearning/MetaOpt/model.ckpt'

def main(_):
  # Configuration.
  num_iter = FLAGS.num_steps // FLAGS.unroll_length

  # Problem.
  config = {'hidden_size': FLAGS.hidden_size, 'num_layer': FLAGS.num_layer, 'unroll_nn': FLAGS.unroll_length,'lr': FLAGS.lr}
  problem = FLAGS.problem()
  optimizer = MetaLearner.MetaOpt(**config)
  step, loss_opt, update, reset, cost_tot, cost_op, arraycost, _ = optimizer.metaoptimizer(problem)

  with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
      graph_writer = tf.summary.FileWriter(logs_path, sess.graph)
      sess.run(tf.global_variables_initializer())
      best_evaluation = float("inf")
      start = timer()
      losstrain = []
      losseval = []
      plotlosstrain = []
      plotlosseval = []
      # Training
      for e in range(FLAGS.num_epochs):
          cost, trainloss = util.run_epoch(sess, num_iter, arraycost, cost_op, [step, update], reset)
          print(cost)
          losstrain.append(cost)
          util.print_stats("Training Epoch {}".format(e), trainloss, timer() - start)
          saver = tf.train.Saver()
          if (e + 1) % FLAGS.logging_period == 0:
              plotlosstrain.append(cost)
          if (e + 1) % FLAGS.evaluation_period == 0:
              for _ in range(FLAGS.evaluation_epochs):
                  evalcost, evaloss = util.run_epoch(sess, num_iter, arraycost, cost_op, [update], reset)
                  losseval.append(evalcost)
                  if save_path is not None and evaloss < best_evaluation:
                      print("Saving meta-optimizer to {}".format(save_path))
                      saver.save(sess, save_path, global_step=0)
                      best_evaluation = evaloss
                      plotlosseval.append(evalcost)
      slengths = np.arange(FLAGS.num_steps)
      plt.figure(figsize=(8, 5))
      plt.plot(slengths, np.mean(plotlosstrain, 0), 'r-', label='Training Loss')
      plt.xlabel('Epoch')
      plt.ylabel('Training Loss')
      plt.legend()
      savefig('Training.png')
      plt.close()
      plt.figure(figsize=(8, 5))
      plt.plot(slengths, np.mean(plotlosseval, 0), 'b-', label='Validation Loss')
      plt.xlabel('Epoch')
      plt.ylabel('Validation Loss')
      plt.legend()
      savefig('Validation.png')
      plt.close()
      graph_writer.close()


if __name__ == "__main__":
  tf.app.run()