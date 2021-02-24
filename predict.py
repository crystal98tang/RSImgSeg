from utils.config import cfg
from utils.utils import testGenerator, vary, rle_encode
import os
import pandas as pd
import imageio
from postprocess.Morphological import morph


model = None
model_name = cfg.ModelName
if model_name == "MRDFCN":
    from models.MRDFCN import *
    model = MRDFCN_4s()
elif model_name == "MRDFCN_fix":
    from models.MRDFCN import *
    model = MRDFCN_4s_fix()

model.load_weights(cfg.PATH.Weights)
test_mask = pd.read_csv(cfg.PATH.TestCSV, sep='\t', names=['name', 'mask'])

#
eval_p = {'Prescision': [], 'Recall': [],  'F_measure': [], 'IoU': [], 'Dice': []}
eval_oc = {'Prescision': [], 'Recall': [],  'F_measure': [], 'IoU': [], 'Dice': []}
eval_co = {'Prescision': [], 'Recall': [],  'F_measure': [], 'IoU': [], 'Dice': []}
eval_crf = {'Prescision': [], 'Recall': [],  'F_measure': [], 'IoU': [], 'Dice': []}
eval_crf_co = {'Prescision': [], 'Recall': [],  'F_measure': [], 'IoU': [], 'Dice': []}
#####
#
imagelist = test_mask['name'].iloc[:].tolist()
#
results_csv = []
batch = 10
cnt = 0
itr = imagelist.__len__() // batch // 250
for i in range(itr):
    testGene = testGenerator(imagelist, i * batch, cfg.PATH.TestDir, batch)
    results = model.predict_generator(testGene, batch, verbose=2)
    cnt = 0
    for mark in results:
        th = 0.6
        mark = mark[:, :, 1]
        # # 单CRF(仅linux
        # mark = crf.CRFs(image, mark, tr_size)
        # # CRF + 闭开
        # mark = morph(mark, operation='co', vary=True, th=th)
        #############################
        # # 形态学开闭
        # res_mor_oc = morph(mark, operation='oc', vary=True, th=th)    # 结果不好
        # #############################
        # # 形态学闭开
        res_mor_co = morph(mark, operation='co', vary=True, th=th)
        #############################
        res = vary(mark, th)
        name = imagelist[i * batch + cnt]
        imageio.imwrite(os.path.join(cfg.PATH.ResultSaveDir, name), res)
        cnt += 1
        rle_res = rle_encode(res)
        results_csv.append(name + rle_res)

res_pd = pd.DataFrame(data=results_csv)
res_pd.to_csv(cfg.PATH.ResultCSV, header=False, index=False)
