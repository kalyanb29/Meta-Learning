from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import matplotlib.pyplot as plt
from pylab import savefig
from timeit import default_timer as timer
import numpy as np
import tensorflow as tf


import meta_ori
import util_ori

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

flags = tf.flags
logging = tf.logging


FLAGS = flags.FLAGS

flags.DEFINE_integer("num_epochs", 20, "Number of training epochs.") # 10000
flags.DEFINE_integer("logging_period", 10, "Log period.") # 100
flags.DEFINE_integer("evaluation_period", 20, "Evaluation period.")#1000
flags.DEFINE_integer("evaluation_epochs", 20, "Number of evaluation epochs.")

flags.DEFINE_string("problem", "segmentation", "Type of problem.")
flags.DEFINE_integer("num_steps", 8000,
                     "Number of optimization steps per epoch.") # 100
flags.DEFINE_integer("unroll_length", 20, "Meta-optimizer unroll length.")
flags.DEFINE_float("learning_rate", 0.001, "Learning rate.")
flags.DEFINE_boolean("second_derivatives", False, "Use second derivatives.")

logs_path = os.getcwd() + '/Log/' + FLAGS.problem + '/OriMetaLog'
save_path = os.getcwd() + '/Save/' + FLAGS.problem + '/OriMetaSave'

if not os.path.exists(logs_path):
    os.makedirs(logs_path)

if not os.path.exists(save_path):
    os.makedirs(save_path)

def main(_):
  # Configuration.
  num_unrolls = FLAGS.num_steps // FLAGS.unroll_length

  # Problem.
  problem, net_config, net_assignments = util_ori.get_config(FLAGS.problem)

  problem = problem['Opt_loss']

  # Optimizer setup.
  optimizer = meta_ori.MetaOptimizer(**net_config)
  minimize = optimizer.meta_minimize(
      problem, FLAGS.unroll_length,
      learning_rate=FLAGS.learning_rate,
      net_assignments=net_assignments,
      second_derivatives=FLAGS.second_derivatives)

  step, loss, update, reset, cost_op, farray, lropt, _ = minimize
  with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
    # Prevent accidental changes to the graph.
    graph_writer = tf.summary.FileWriter(logs_path, sess.graph)
    sess.run(tf.global_variables_initializer())
    best_evaluation = float("inf")
    start = timer()
    losstrain = []
    lrtrain = []
    losseval = []
    plotlosstrain = []
    plotlrtrain = []
    plotlosseval = []
    for e in range(FLAGS.num_epochs):
      cost, trainloss, lropttrain = util_ori.run_epoch(sess, cost_op, farray, lropt, [step, update], reset, num_unrolls)
      print(cost)
      losstrain.append(cost)
      lrtrain.append(lropttrain)
      util_ori.print_stats("Training Epoch {}".format(e), trainloss, timer() - start)
      saver = tf.train.Saver()
      if (e + 1) % FLAGS.logging_period == 0 and (e + 1) > int(FLAGS.num_epochs / 2):
          plotlosstrain.append(cost)
          plotlrtrain.append(lropttrain)

      if (e + 1) % FLAGS.evaluation_period == 0:
        for _ in range(FLAGS.evaluation_epochs):
          evalcost, evaloss, _ = util_ori.run_epoch(sess, cost_op, farray, lropt, [update], reset, num_unrolls)
          losseval.append(evalcost)
        if save_path is not None and evaloss < best_evaluation:
          print("Saving meta-optimizer to {}".format(save_path))
          saver.save(sess, save_path + '/model.ckpt', global_step=e + 1)
          best_evaluation = evaloss
          plotlosseval.append(evalcost)
    slengths = np.arange(FLAGS.num_steps)
    slengthlr = np.arange(FLAGS.num_steps - num_unrolls)
    np.savetxt(save_path + '/plotlosstrain.out', plotlosstrain, delimiter=',')
    np.savetxt(save_path + '/plotlrtrain.out', plotlrtrain, delimiter=',')
    np.savetxt(save_path + '/plotlosseval.out', plotlosseval, delimiter=',')
    np.savetxt(save_path + '/losstrain.out', losstrain, delimiter=',')
    np.savetxt(save_path + '/lrtrain.out', plotlosstrain, delimiter=',')
    np.savetxt(save_path + '/losseval.out', losseval, delimiter=',')
    plt.figure(figsize=(8, 5))
    plt.plot(slengths, np.mean(plotlosstrain, 0), 'r-', label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Training Loss')
    plt.legend()
    savefig(save_path + '/Training.png')
    plt.close()
    plt.figure(figsize=(8, 5))
    plt.plot(slengths, np.mean(plotlosseval, 0), 'b-', label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Validation Loss')
    plt.legend()
    savefig(save_path + '/Validation.png')
    plt.close()
    plt.figure(figsize=(8, 5))
    plt.plot(slengthlr, np.mean(plotlrtrain, 0), 'r-', label='Learning Rate')
    plt.xlabel('Epoch')
    plt.ylabel('Average Learning Rate')
    plt.legend()
    savefig(save_path + '/LearningRate.png')
    plt.close()
    graph_writer.close()

if __name__ == "__main__":
  tf.app.run()
