from easydict import EasyDict as edict

__C = edict()

cfg = __C

__C.Mode = "predict"  # run mode: train / test / predict
__C.ModelName = "MRDFCN"
__C.global_image_size = 512
__C.global_image_channels = 3
__C.global_label_classes = 2

__C.PATH = edict()
__C.PATH.TestDir = "DataSet/test_a"
__C.PATH.TestCSV = "DataSet/test_a_samplesubmit.csv"

__C.PATH.ResultSaveDir = "results/"
__C.PATH.ResultCSV = "results/CSV/result.csv"

__C.PATH.Weights = "logs/MRDFCN_final_256.hdf5"

__C.PATH.TrainDir = "DataSet/train_a"
__C.PATH.TrainDict = {"image": "train", "label": "train_mark"}
__C.PATH.TrainCSV = "DataSet/train_mask.csv"
