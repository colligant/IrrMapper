PROJECT = 'GEE-IrrMapper'
BUCKET = 'ee-irrigation-mapping'
REMOTE_OR_LOCAL = 'local'
if REMOTE_OR_LOCAL == 'remote':
    FOLDER = 'models/0.01wd-nondvi-july28-2mparams-final-split-balanced'
    JOB_DIR = 'gs://' + BUCKET + '/' + FOLDER + '/trainer'
    MODEL_DIR = JOB_DIR + '/model'
    LOGS_DIR = JOB_DIR + '/logs'
else:
    FOLDER = '/home/thomas/models/shared-weights'
    JOB_DIR = FOLDER + '/trainer'
    MODEL_DIR = JOB_DIR + '/model'
    LOGS_DIR = JOB_DIR + '/logs'

TRAIN_BASE = 'train-data-aug25/'
TEST_BASE = 'test-data-aug24/'
TEST_SIZE = 393 * 9
BATCH_SIZE = 16

EPOCHS = 300
STEPS_PER_EPOCH = 400
BUFFER_SIZE = 1
N_CLASSES = 3
