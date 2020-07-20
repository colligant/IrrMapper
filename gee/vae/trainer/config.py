# What to do?
# export GOOGLE_APPLICATION_CREDENTIALS=/home/thomas/Downloads/GEE-IrrMapper-10626e1135d5.json
# && gcloud auth login
# gsutil config?

PROJECT = 'GEE-IrrMapper'
BUCKET = 'ee-irrigation-mapping'
REMOTE_OR_LOCAL = 'remote'
FOLDER = '{}-july17-2mparams-weight-decay'.format(REMOTE_OR_LOCAL)
JOB_DIR = 'gs://' + BUCKET + '/' + FOLDER + '/trainer'
MODEL_DIR = JOB_DIR + '/model'
LOGS_DIR = JOB_DIR + '/logs'

DATA_BUCKET = ''
TRAIN_BASE = 'train-data-july9_1-578'
TEST_BASE = 'test-data-june30-578/'
TRAIN_SIZE = 1000
TEST_SIZE = 19446 # idk what this really is anymore... (why TFRecord?!)

if REMOTE_OR_LOCAL == 'remote':
    BATCH_SIZE = 32
else:
    BATCH_SIZE = 32

EPOCHS = 200
STEPS_PER_EPOCH = 600
BUFFER_SIZE = 10
N_CLASSES = 3
