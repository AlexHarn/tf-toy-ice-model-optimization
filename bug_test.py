#!/usr/bin/env python
# encoding: utf-8

from __future__ import division, print_function
import numpy as np
import tensorflow as tf

FLOAT_PRECISION = tf.float32

def main():

    #--------------------------
    # Settings
    #--------------------------
    learning_rate = 1
    num_steps = 100
    true_iters = 10.
    initial_pred_iters = 1.
    initial_loop_var = 20.

    '''
    Test Settings:
        1: Do plain counting in loop.
           The number of loop iterations is defined by a trainable 
           parameter named loop_var.
           This will fail, since no gradients exist

        2: Artificially add gradient by adding a zero
           This will work (by chance).
           Possible Explanation: the artificial 0 is not exact and therefore
                                 loop_var does in fact provide a gradient.
                                 This gradient defines the direction, whereas
                                 the magnitude is defined by the loss. 
                                 Therefore, convergence is achieved, since the
                                 magnitude drops

        3. Artificially add gradient by adding a zero, but this time in 
           "different direction".
           Similar to test 2, but now the zero is added the other way around.
           As a result, the gradient points in the other direction.
    '''    
    test_setting = 3
    
    #--------------------------
    # create true distribution
    #--------------------------
    true_iters = tf.constant(true_iters)
    loop_var = tf.Variable(initial_loop_var, trainable=True)

    #--------------------------
    # create trainable distribution
    #--------------------------
    def body(iter_var):

            # count iterations

            if test_setting == 1:
                ## This will Fail: [No Gradient available] 
                iter_var += tf.ones_like(iter_var)

            elif test_setting == 2:
                ## This will work: [Gradient over Artificial 0] 
                iter_var += tf.ones_like(iter_var) + \
                            loop_var / tf.stop_gradient(loop_var) - 1

            elif test_setting == 3:
                ## This will go in wrong direction:
                iter_var += tf.ones_like(iter_var) - \
                            loop_var / tf.stop_gradient(loop_var) + 1

            else:
                raise ValueError('Not a valid test setting: {}'.format(
                                                                test_setting))

            return [iter_var]

    pred_iters =  tf.while_loop(
                    lambda iter_var: tf.less(iter_var, loop_var),
                    lambda iter_var: body(iter_var),
                    [initial_pred_iters],
                    parallel_iterations=1)
    print('pred_iters',pred_iters)


    #--------------------------
    # Define Loss and Minimizer
    #--------------------------
    loss = tf.squared_difference(true_iters, pred_iters)
    optimizer = tf.train.AdamOptimizer(
                                    learning_rate=learning_rate).minimize(loss)
        
    #--------------------------------------
    # Start Tensorflow
    #--------------------------------------
    print('Now Starting Tensorflow')

    config = tf.ConfigProto(gpu_options = tf.GPUOptions(
                                            allow_growth = True,
                                            ))
        
    # create tensorflow session
    session = tf.Session(config=config)

    # initialize variables
    session.run(tf.global_variables_initializer())


    #--------------------------------------
    # Perform Training
    #--------------------------------------
    for i in range(num_steps):
        result = session.run({ 'optimizer' : optimizer, 
                               'true_iters' : true_iters,
                               'pred_iters' : pred_iters,
                               'loop_var' : loop_var,
                               })

        print('[{:08d}] true: {true_iters:2.2f} pred: {pred_iters:2.2f}'
              ' loop var: {loop_var:2.2f}'.format( i,**result))


if __name__ == '__main__':
    main()