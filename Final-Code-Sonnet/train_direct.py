from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import matplotlib.pyplot as plt
from pylab import savefig
from timeit import default_timer as timer
import numpy as np
import tensorflow as tf

import util

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

flags = tf.flags
logging = tf.logging


FLAGS = flags.FLAGS
flags.DEFINE_integer("num_epochs", 20, "Number of training epochs.") # 10000
flags.DEFINE_integer("logging_period", 10, "Log period.") # 100

flags.DEFINE_string("problem", "segmentation", "Type of problem.")
flags.DEFINE_integer("num_steps", 8000,
                     "Number of optimization steps per epoch.") # 100
flags.DEFINE_float('lr',0.001,"Initial learning rate")
flags.DEFINE_float('momentum',0.9,"Initial momentum value")
flags.DEFINE_string('optimizer',"ADAM","Type of optimizer")

logs_path = os.getcwd() + '/Log/' + FLAGS.problem + '/DirectLog/' + FLAGS.optimizer
save_path = os.getcwd() + '/Save/' + FLAGS.problem + '/DirectSave/' + FLAGS.optimizer

if not os.path.exists(logs_path):
    os.makedirs(logs_path)

if not os.path.exists(save_path):
    os.makedirs(save_path)

def main(_):
  # Configuration.
  problem, net_config, net_assignments = util.get_config(FLAGS.problem)
  problem = problem['Opt_loss']
  loss = problem()
  if FLAGS.optimizer == "SGD":
     optimizer = tf.train.GradientDescentOptimizer(FLAGS.lr)
  elif FLAGS.optimizer == "ADAM":
     optimizer = tf.train.AdamOptimizer(FLAGS.lr)
  elif FLAGS.optimizer == "RMSprop":
      optimizer = tf.train.RMSPropOptimizer(FLAGS.lr)
  elif FLAGS.optimizer == "Momentum":
      optimizer = tf.train.MomentumOptimizer(FLAGS.lr,FLAGS.momentum)

  gvs = optimizer.compute_gradients(loss)
  grads, tvars = zip(*gvs)
  step = optimizer.apply_gradients(zip(grads, tvars))

  reset = tf.variables_initializer(tvars)
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
          sess.run(reset)
          costall = []
          evalcostall = []
          saver = tf.train.Saver()
          for _ in range(FLAGS.num_steps):
             cost,_ = sess.run([loss,step])
             costall.append(np.log10(cost))
             evalcost = sess.run(loss)
             evalcostall.append(np.log10(evalcost))
             if save_path is not None and evalcost < best_evaluation:
                 print("Saving meta-optimizer to {}".format(save_path))
                 saver.save(sess, save_path + '/model.ckpt', global_step=e + 1)
                 best_evaluation = evalcost
          losstrain.append(np.reshape(costall, -1))
          losseval.append(np.reshape(evalcostall, -1))
          util.print_stats("Training Epoch {}".format(e), cost, timer() - start)
          if (e + 1) % FLAGS.logging_period == 0:
              plotlosstrain.append(costall)
              plotlosseval.append(evalcostall)
      slengths = np.arange(FLAGS.num_steps)
      np.savetxt(save_path + '/plotlosstrain.out', plotlosstrain, delimiter=',')
      np.savetxt(save_path + '/plotlosseval.out', plotlosseval, delimiter=',')
      np.savetxt(save_path + '/losstrain.out', losstrain, delimiter=',')
      np.savetxt(save_path + '/losseval.out', losseval, delimiter=',')
      plt.figure(figsize=(8, 5))
      plt.plot(slengths, np.mean(plotlosstrain, 0), 'r-', label='Training Loss')
      plt.xlabel('Epoch')
      plt.ylabel('Training Loss')
      plt.legend()
      savefig(save_path + '/Training_' + FLAGS.optimizer + '.png')
      plt.close()
      plt.figure(figsize=(8, 5))
      plt.plot(slengths, np.mean(plotlosseval, 0), 'b-', label='Validation Loss')
      plt.xlabel('Epoch')
      plt.ylabel('Validation Loss')
      plt.legend()
      savefig(save_path + '/Validation_' + FLAGS.optimizer + '.png')
      plt.close()
      graph_writer.close()


if __name__ == "__main__":
  tf.app.run()