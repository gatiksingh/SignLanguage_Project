import os
import numpy as np

ACTIONS = np.array(['HELLO', 'YES', 'NO', 'HELP', 'PLEASE', 'I LOVE YOU',
                    'STOP', 'WATER', 'SORRY', 'GOOD BYE'])

DATA_PATH = os.path.join('SignData')
NO_SEQUENCES   = 35  
SEQUENCE_LENGTH = 30
IMG_SIZE        = (64, 64)