import torch
from dataloader import ViPCDataLoader
from tqdm import tqdm
from torch.utils.data import DataLoader
from config import params
from model import Network
from decoder.utils.utils import *

opt = params()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
ViPCDataset_test = ViPCDataLoader('test_list2.txt', data_path=opt.dataroot, status= "test", category = opt.cat)
test_loader = DataLoader(ViPCDataset_test,
                            batch_size=32,
                            num_workers=opt.nThreads,
                            shuffle=True,
                            drop_last=True)



model = Network().to(device)
model.load_state_dict(torch.load("")['model_state_dict'], strict=True)
loss_eval = L2_ChamferEval_1000()
loss_f1 = F1Score()

loss_de = Loss_Density()

with torch.no_grad():
    model.eval()
    i = 0
    Loss = 0
    Loss_de_complete = 0
    Loss_de_final = 0
    f1_final = 0
    for data in tqdm(test_loader):

        i += 1
        image = data[0].to(device)
        partial = data[2].to(device)
        gt = data[1].to(device)  

        partial = farthest_point_sample(partial, 2048)
        gt = farthest_point_sample(gt, 2048)
        partial = partial.permute(0, 2, 1)
        complete, final, pc_generate, pred_pcds = model(partial, image)

        #Compute the eval loss
        loss = loss_eval(complete, gt)
        f1, _, _  = loss_f1(complete, gt)
        f1 = f1.mean()

        Loss += loss
        f1_final += f1

    Loss = Loss/i
    f1_final = f1_final/i


print(f"The evaluation loss for {opt.cat} is :{Loss}")
print(f"The F1-score for {opt.cat} is :{f1_final}")