PROJECT = 'GEE-IrrMapper'
BUCKET = 'ee-irrigation-mapping'
REMOTE_OR_LOCAL = 'local'
if REMOTE_OR_LOCAL == 'remote':
    FOLDER = 'models/0.01wd-nondvi-july28-2mparams-final-split-balanced'
    JOB_DIR = 'gs://' + BUCKET + '/' + FOLDER + '/trainer'
    MODEL_DIR = JOB_DIR + '/model'
    LOGS_DIR = JOB_DIR + '/logs'
else:
    FOLDER = '/home/thomas/models/aug28lr0.001'
    JOB_DIR = FOLDER + '/trainer'
    MODEL_DIR = JOB_DIR + '/model'
    LOGS_DIR = JOB_DIR + '/logs'

TRAIN_BASE = 'train-data-aug25/'
TEST_BASE = 'test-data-aug24/'
TEST_SIZE = 3492
BATCH_SIZE = 16
TRAIN_STEPS = 1000

EPOCHS = 1000
STEPS_PER_EPOCH = 400
BUFFER_SIZE = 1
N_CLASSES = 3
