PROJECT = 'GEE-IrrMapper'
BUCKET = 'ee-irrigation-mapping'
FOLDER = 'models/0.01wd-nondvi-july28-2mparams-final-split-balanced'
JOB_DIR = 'gs://' + BUCKET + '/' + FOLDER + '/trainer'
MODEL_DIR = JOB_DIR + '/model'
LOGS_DIR = JOB_DIR + '/logs'

TRAIN_BASE = 'train-data-july23/'
TEST_BASE = 'test-data-july23/'
TEST_SIZE = 8673 #19446 
BATCH_SIZE = 32

EPOCHS = 300
STEPS_PER_EPOCH = 400
BUFFER_SIZE = 10
N_CLASSES = 5
