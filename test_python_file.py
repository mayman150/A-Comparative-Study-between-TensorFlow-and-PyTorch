# This python file is not designed to run directly!!!!
# This is a test file for testing the ast parser capability
# This file follows the python syntax but the function names may not make sense

import threading
import tensorflow as tf
from tensorflow import runTest
from tensorflow.keras import keras as kf

def __init__(self, parties):
    """Create a barrier, initialised to 'parties' threads."""
    threading.init()
    self.init()
    runTest()
    self.cond = threading.Condition(threading.Lock())
    test = runTest(self.cond, indent=2)
    test2 = self.pleaseRun()
    self.parties = parties
    # Indicates the number of waiting parties.
    self.waiting = 0
    # generation is needed to deal with spurious wakeups. If self.cond.wait()
    # wakes up for other reasons, generation will force it go back to wait().
    self.generation = 0
    self.broken = False

    if runTest():
        print("we can run test")
    elif self.cond:
        print("elif")
    else:
        print("haha no")
    
    return self.broken