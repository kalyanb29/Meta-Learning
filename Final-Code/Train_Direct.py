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
flags.DEFINE_integer("num_epochs", 100, "Number of training epochs.") # 10000
flags.DEFINE_integer("logging_period", 10, "Log period.") # 100
flags.DEFINE_integer("evaluation_period", 1, "Evaluation period.")#1000

flags.DEFINE_string("problem", "quadratic", "Type of problem.")
flags.DEFINE_integer("num_steps", 100,
                     "Number of optimization steps per epoch.") # 100
flags.DEFINE_float('lr',0.001,"Initial learning rate")
flags.DEFINE_float('momentum',0.9,"Initial momentum value")
flags.DEFINE_string('optimizer',"Momentum","Type of optimizer")

logs_path = '/Users/kalyanb/PycharmProjects/Final-Code//Log/'
save_path = '/Users/kalyanb/PycharmProjects/Final-Code/Save/'

def main(_):
  # Configuration.
  problem = util.get_problem_config(FLAGS.problem)
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
      graph_writer = tf.summary.FileWriter(logs_path + FLAGS.optimizer + 'Log/', sess.graph)
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
          for _ in range(FLAGS.num_steps):
             cost,_ = sess.run([loss,step])
             costall.append(np.log10(cost))
          losstrain.append(np.reshape(costall, -1))
          util.print_stats("Training Epoch {}".format(e), cost, timer() - start)
          saver = tf.train.Saver()
          if (e + 1) % FLAGS.logging_period == 0:
              plotlosstrain.append(costall)
          if (e + 1) % FLAGS.evaluation_period == 0:
              evalcost = sess.run(loss)
              losseval.append(evalcost)
              plotlosseval.append(np.log10(evalcost))
              if save_path is not None and evalcost < best_evaluation:
                 print("Saving meta-optimizer to {}".format(save_path))
                 saver.save(sess, save_path + FLAGS.optimizer + 'Save/model.ckpt', global_step=0)
                 best_evaluation = evalcost
      slengths = np.arange(FLAGS.num_steps)
      plt.figure(figsize=(8, 5))
      plt.plot(slengths, np.mean(plotlosstrain, 0), 'r-', label='Training Loss')
      plt.xlabel('Epoch')
      plt.ylabel('Training Loss')
      plt.legend()
      savefig('Training_' + FLAGS.optimizer + '.png')
      plt.close()
      plt.figure(figsize=(8, 5))
      plt.plot(slengths, np.reshape(plotlosseval,-1), 'b-', label='Validation Loss')
      plt.xlabel('Epoch')
      plt.ylabel('Validation Loss')
      plt.legend()
      savefig('Validation_' + FLAGS.optimizer + '.png')
      plt.close()
      graph_writer.close()


if __name__ == "__main__":
  tf.app.run()