# What to do?
# export GOOGLE_APPLICATION_CREDENTIALS=/home/thomas/Downloads/GEE-IrrMapper-10626e1135d5.json
# && gcloud auth login
# gsutil config?
# I've given irrigation-mapping ADMIN priviledges to dis buccccckkkkkeeeeeet.

PROJECT = 'GEE-IrrMapper'
BUCKET = 'ee-irrigation-mapping'
FOLDER = 'fcnn-local-train-june17'
JOB_DIR = 'gs://' + BUCKET + '/' + FOLDER + '/trainer'
MODEL_DIR = JOB_DIR + '/model'
LOGS_DIR = JOB_DIR + '/logs'

DATA_BUCKET = ''
TRAIN_BASE = 'training-data-june16'
TEST_BASE = 'test-data-june16'

TRAIN_SIZE = 1000
TEST_SIZE = 6096

BATCH_SIZE = 32
EPOCHS = 100
STEPS_PER_EPOCH = 250
VAL_STEPS = 20
BUFFER_SIZE = 1
N_CLASSES = 3
