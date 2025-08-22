#------------------config.py------------------

PRETRAINED_MODEL_PATH = 'weights/congait-small.pth'
SEGMENT_LENGTH = 1000  # 10 seconds at 100Hz 
NUM_CLASSES = 4  # Healthy, Stage 2, Stage 2.5, Stage 3
CLASS_NAMES = ["Healthy","Stage 2","Stage 2.5","Stage 3"]
SENSOR_NAMES = [f"Left VGRF-{i}" for i in range(1,9)] + [f"Right VGRF-{i}" for i in range(1,9)]


