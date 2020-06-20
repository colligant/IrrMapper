# What to do?
# export GOOGLE_APPLICATION_CREDENTIALS=/home/thomas/Downloads/GEE-IrrMapper-10626e1135d5.json
# && gcloud auth login
# gsutil config?

PROJECT = 'GEE-IrrMapper'
BUCKET = 'ee-irrigation-mapping'
REMOTE_OR_LOCAL = 'remote'
FOLDER = 'fcnn-{}-train-june19'.format(REMOTE_OR_LOCAL)
JOB_DIR = 'gs://' + BUCKET + '/' + FOLDER + '/trainer'
MODEL_DIR = JOB_DIR + '/model'
LOGS_DIR = JOB_DIR + '/logs'

DATA_BUCKET = ''
TRAIN_BASE = 'training-data-june18-578'
TEST_BASE = 'test-data-june18-578/'
TRAIN_SIZE = 1000
TEST_SIZE = 8969

if REMOTE_OR_LOCAL == 'remote':
    BATCH_SIZE = 64
else:
    BATCH_SIZE = 32

EPOCHS = 100
STEPS_PER_EPOCH = 140
BUFFER_SIZE = 15
N_CLASSES = 3
