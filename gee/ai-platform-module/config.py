# What to do?
# export GOOGLE_APPLICATION_CREDENTIALS=/home/thomas/Downloads/GEE-IrrMapper-10626e1135d5.json
# && gcloud auth login
# gsutil config?
# I've given irrigation-mapping ADMIN priviledges to dis buccccckkkkkeeeeeet.

PROJECT = 'GEE-IrrMapper'
BUCKET = 'ee-irrigation-mapping'
FOLDER = 'fcnn-try1-june-10'
JOB_DIR = 'gs://' + BUCKET + '/' + FOLDER + '/trainer'
MODEL_DIR = JOB_DIR + '/model'
LOGS_DIR = JOB_DIR + '/logs'

DATA_BUCKET = 'training-data'
TRAIN_BASE = 'train'
TEST_BASE = 'test'

TRAIN_SIZE = 1000
TEST_SIZE = 200

BATCH_SIZE = 16
EPOCHS = 100
STEPS_PER_EPOCH = 200
VAL_STEPS = 100
BUFFER_SIZE = 500
OPTIMIZER = 'Adam'
LOSS = 'cross_entropy'
N_CLASSES = 3
