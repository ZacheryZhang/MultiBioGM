import os
import numpy as np
import time
import datetime
import torch
import torchvision
from torch import optim
from torch.autograd import Variable
import torch.nn.functional as F
from evaluation import *
from network import U_Net,R2U_Net,AttU_Net,R2AttU_Net
import csv
import cv2
from torchvision import utils as vutils
import math

class BCEFocalLoss(torch.nn.Module):    
    def __init__(self, gamma=10, alpha=0.25, reduction='elementwise_mean'):
        super().__init__()
        self.gamma = math.log(gamma+1)
        self.alpha = alpha
        self.reduction = reduction
 
    def forward(self, _input, target):
        pt = torch.sigmoid(_input)
        alpha = self.alpha
        loss = - alpha * (1 - pt) ** self.gamma * target * torch.log(pt) - \
               (1 - alpha) * pt ** self.gamma * (1 - target) * torch.log(1 - pt)
        if self.reduction == 'elementwise_mean':
            loss = torch.mean(loss)
        elif self.reduction == 'sum':
            loss = torch.sum(loss)
        return loss

class Solver(object):
    def __init__(self, config, train_loader, valid_loader, test_loader,infer_loader):

        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.test_loader = test_loader
        self.infer_loader=infer_loader

        self.unet = None
        self.optimizer = None
        self.img_ch = config.img_ch
        self.output_ch = config.output_ch
        self.criterion = BCEFocalLoss()
        self.augmentation_prob = config.augmentation_prob
        self.test_model_path=config.test_model_path

        self.lr = config.lr
        self.alpha1 = 1
        self.alpha2 = 1
        self.alpha3 = 1
        self.beta1 = config.beta1
        self.beta2 = config.beta2
        self.image_size=config.image_size

        self.num_epochs = config.num_epochs
        self.num_epochs_decay = config.num_epochs_decay
        self.batch_size = config.batch_size

        self.log_step = config.log_step
        self.val_step = config.val_step

        self.model_path = config.model_path
        self.result_path = config.result_path
        self.mode = config.mode
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_type = config.model_type
        self.t = config.t
        self.build_model()

    def build_model(self):
        """Build generator and discriminator."""
        if self.model_type =='U_Net':
            self.unet = U_Net(img_ch=3,output_ch=1)
        elif self.model_type =='R2U_Net':
            self.unet = R2U_Net(img_ch=3,output_ch=1,t=self.t)
        elif self.model_type =='AttU_Net':
            self.unet = AttU_Net(img_ch=3,output_ch=3)
        elif self.model_type == 'R2AttU_Net':
            self.unet = R2AttU_Net(img_ch=3,output_ch=1,t=self.t)
            

        self.optimizer = optim.Adam(list(self.unet.parameters()),
                                      self.lr, [self.beta1, self.beta2])
        self.unet.to(self.device)


    def print_network(self, model, name):
        num_params = 0
        for p in model.parameters():
            num_params += p.numel()
        print(model)
        print(name)
        print("The number of parameters: {}".format(num_params))

    def to_data(self, x):
        """Convert variable to tensor."""
        if torch.cuda.is_available():
            x = x.cpu()
        return x.data

    def update_lr(self, g_lr, d_lr):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def reset_grad(self):
        """Zero the gradient buffers."""
        self.unet.zero_grad()
    
    def compute_accuracy(self,SR,GT):
        SR_flat = SR.view(-1)
        GT_flat = GT.view(-1)

        acc = GT_flat.data.cpu()==(SR_flat.data.cpu()>0.5)

    def tensor2img(self,x):
        img = (x[:,0,:,:]>x[:,1,:,:]).float()
        img = img*255
        return img
    def save_image_tensor(self,input_tensor: torch.Tensor, filename):
        input_tensor = input_tensor.clone().detach()
        input_tensor = input_tensor.to(torch.device('cpu'))
        vutils.save_image(input_tensor.float(), filename)
    def get_Several_MinMax_Array(np_arr, several):
        if several > 0:
            several_min_or_max = np_arr[np.argpartition(np_arr,several)[:several]]
        else:
            several_min_or_max = np_arr[np.argpartition(np_arr, several)[several:]]
        return np.mean(several_min_or_max)

    def train(self):
        """Train encoder, generator and discriminator."""

        #====================================== Training ===========================================#
        #===========================================================================================#
        
        unet_path = os.path.join(self.model_path, '%s-%d-%.4f-%d-%.4f.pkl' %(self.model_type,self.num_epochs,self.lr,self.num_epochs_decay,self.augmentation_prob))

        if os.path.isfile(unet_path):
            self.unet.load_state_dict(torch.load(unet_path))
            print('%s is Successfully Loaded from %s'%(self.model_type,unet_path))
        else:
            lr = self.lr
            best_unet_score = 0.
            
            for epoch in range(self.num_epochs):

                self.unet.train(True)
                epoch_loss = 0
                
                acc = 0.    # Accuracy
                SE = 0.     # Sensitivity (Recall)
                SP = 0.     # Specificity
                PC = 0.     # Precision
                F1 = 0.     # F1 Score
                JS = 0.     # Jaccard Similarity
                DC = 0.     # Dice Coefficient
                length = 0

                for i, (images, GT,_) in enumerate(self.train_loader):

                    images = images.to(self.device)
                    t=images.cpu().detach().numpy()[0]
                    cv2.imwrite("t1.png",t[0]*255)
                    cv2.imwrite("t2.png",t[1]*255)
                    cv2.imwrite("t3.png",t[2]*255)
                    
                    GT = GT.to(self.device)
                    SR = self.unet(images)
                    SR_probs = torch.sigmoid(SR)
                    t=SR_probs.cpu().detach().numpy()[0]
                    cv2.imwrite("t1_.png",t[0]*255)
                    cv2.imwrite("t2_.png",t[1]*255)
                    cv2.imwrite("t3_.png",t[2]*255)
                    
                    SR_flat = SR_probs.view(SR_probs.size(0),-1)
                    
                    GT_flat = GT.view(GT.size(0),-1)

                    div=GT.size(2)*GT.size(3)
                    GT_flats=torch.split(GT_flat,div,dim=1)
                    top1s=torch.topk(GT_flats[0],k=10, dim=1)
                    top2s=torch.topk(GT_flats[1],k=10, dim=1)
                    top3s=torch.topk(GT_flats[2],k=10, dim=1)
                    p1=torch.mean(top1s[0])
                    p2=torch.mean(top2s[0])
                    p3=torch.mean(top3s[0])
                    maxid=1
                    maxp=p1
                    if maxp<p2:
                      maxid=2
                      maxp=p2
                    if maxp<p3:
                      maxid=3
                      maxp=p3

                    p=1+(1-maxp)/maxp
                    t=[torch.clone(GT_flats[0]),torch.clone(GT_flats[1]),torch.clone(GT_flats[2])]
                    t[maxid-1]=t[maxid-1]*p;
                    if maxid==1:
                      top1s=torch.topk(t[0],k=20, dim=1)
                    elif maxid==2:
                      top2s=torch.topk(t[1],k=20, dim=1)
                    elif maxid==3:
                      top3s=torch.topk(t[2],k=20, dim=1)
                    p1=torch.mean(top1s[0])
                    p2=torch.mean(top2s[0])
                    p3=torch.mean(top3s[0])
                    if maxid==1:
                      torch.where(t[1]>torch.mean(t[1]),(p1/p2)*t[1],t[1])
                      torch.where(t[2]>torch.mean(t[2]),(p1/p3)*t[2],t[2])
                    if maxid==2:
                      torch.where(t[0]>torch.mean(t[0]),(p2/p1)*t[0],t[0])
                      torch.where(t[2]>torch.mean(t[2]),(p2/p3)*t[2],t[2])
                    if maxid==3:
                      torch.where(t[0]>torch.mean(t[0]),(p3/p1)*t[0],t[0])
                      torch.where(t[1]>torch.mean(t[1]),(p3/p2)*t[1],t[1])
                    GT_flat=torch.cat((t[0],t[1],t[2]),dim=1)
                    t=GT.cpu().detach().numpy()[0]
                    cv2.imwrite("t1__.png",t[0]*255)
                    cv2.imwrite("t2__.png",t[1]*255)
                    cv2.imwrite("t3__.png",t[2]*255)
                    
                    loss = self.criterion(SR_flat,GT_flat)

                    epoch_loss += loss.item()

                    self.reset_grad()
                    loss.backward()
                    self.optimizer.step()

                    acc += get_accuracy(SR,GT)
                    SE += get_sensitivity(SR,GT)
                    SP += get_specificity(SR,GT)
                    PC += get_precision(SR,GT)
                    F1 += get_F1(SR,GT)
                    JS += get_JS(SR,GT)
                    DC += get_DC(SR,GT)
                    length += 1

                acc = acc/length
                SE = SE/length
                SP = SP/length
                PC = PC/length
                F1 = F1/length
                JS = JS/length
                DC = DC/length

                print('Epoch [%d/%d], Loss: %.4f, \n[Training] Acc: %.4f, SE: %.4f, SP: %.4f, PC: %.4f, F1: %.4f, JS: %.4f, DC: %.4f' % (
                      epoch+1, self.num_epochs, \
                      epoch_loss,\
                      acc,SE,SP,PC,F1,JS,DC))

            

                if (epoch+1) > (self.num_epochs - self.num_epochs_decay):
                    lr -= (self.lr / float(self.num_epochs_decay))
                    for param_group in self.optimizer.param_groups:
                        param_group['lr'] = lr
                    print ('Decay learning rate to lr: {}.'.format(lr))
                
                
                #===================================== Validation ====================================#
                self.unet.train(False)
                self.unet.eval()

                acc = 0.    # Accuracy
                SE = 0.     # Sensitivity (Recall)
                SP = 0.     # Specificity
                PC = 0.     # Precision
                F1 = 0.     # F1 Score
                JS = 0.     # Jaccard Similarity
                DC = 0.     # Dice Coefficient
                length=0
                for i, (images, GT,_) in enumerate(self.valid_loader):

                    images = images.to(self.device)
                    GT = GT.to(self.device)
                    SR = torch.sigmoid(self.unet(images))
                    acc += get_accuracy(SR,GT)
                    SE += get_sensitivity(SR,GT)
                    SP += get_specificity(SR,GT)
                    PC += get_precision(SR,GT)
                    F1 += get_F1(SR,GT)
                    JS += get_JS(SR,GT)
                    DC += get_DC(SR,GT)
                        
                    length += 1
                    
                acc = acc/length
                SE = SE/length
                SP = SP/length
                PC = PC/length
                F1 = F1/length
                JS = JS/length
                DC = DC/length
                unet_score = JS + DC

                print('[Validation] Acc: %.4f, SE: %.4f, SP: %.4f, PC: %.4f, F1: %.4f, JS: %.4f, DC: %.4f'%(acc,SE,SP,PC,F1,JS,DC))
                
                '''
                torchvision.utils.save_image(images.data.cpu(),
                                            os.path.join(self.result_path,
                                                        '%s_valid_%d_image.png'%(self.model_type,epoch+1)))
                torchvision.utils.save_image(SR.data.cpu(),
                                            os.path.join(self.result_path,
                                                        '%s_valid_%d_SR.png'%(self.model_type,epoch+1)))
                torchvision.utils.save_image(GT.data.cpu(),
                                            os.path.join(self.result_path,
                                                        '%s_valid_%d_GT.png'%(self.model_type,epoch+1)))
                '''


                if unet_score > best_unet_score:
                    best_unet_score = unet_score
                    best_epoch = epoch
                    best_unet = self.unet.state_dict()
                    print('Best %s model score : %.4f'%(self.model_type,best_unet_score))
                    torch.save(best_unet,unet_path)
    def test(self):
            #===================================== Test ====================================#
            self.build_model()
            self.unet.load_state_dict(torch.load(self.test_model_path))
            
            self.unet.train(False)
            self.unet.eval()

            acc = 0.    # Accuracy
            SE = 0.     # Sensitivity (Recall)
            SP = 0.     # Specificity
            PC = 0.     # Precision
            F1 = 0.     # F1 Score
            JS = 0.     # Jaccard Similarity
            DC = 0.     # Dice Coefficient
            length=0
            for i, (images, GT) in enumerate(self.test_loader):

                images = images.to(self.device)
                GT = GT.to(self.device)
                SR = torch.sigmoid(self.unet(images))
                acc += get_accuracy(SR,GT)
                SE += get_sensitivity(SR,GT)
                SP += get_specificity(SR,GT)
                PC += get_precision(SR,GT)
                F1 += get_F1(SR,GT)
                JS += get_JS(SR,GT)
                DC += get_DC(SR,GT)

                threshold=0.5
                threshold2=0.8
                RES=(SR > threshold).long()
                RES2=(GT > threshold2).long()
                for j in range(self.batch_size):
                    self.save_image_tensor(RES[j],os.path.join(self.result_path,str(i)+str(j)+"_1.png"))
                    self.save_image_tensor(RES2[j],os.path.join(self.result_path,str(i)+str(j)+"_2.png"))
                    self.save_image_tensor(images[j],os.path.join(self.result_path,str(i)+str(j)+"_3.png"))
                length += 1
                    
            acc = acc/length
            SE = SE/length
            SP = SP/length
            PC = PC/length
            F1 = F1/length
            JS = JS/length
            DC = DC/length
            unet_score = JS + DC

            print("model_type:%s;acc:%.4lf;SE:%.4lf;SP:%.4lf;PC:%.4lf;F1:%.4lf;JS:%.4lf;DC:%4lf"%(self.model_type,acc,SE,SP,PC,F1,JS,DC))
    def infer(self):
            #===================================== Infer ====================================#
            self.build_model()
            self.unet.load_state_dict(torch.load(self.test_model_path))
            
            self.unet.train(False)
            self.unet.eval()
            import time
            T1=time.perf_counter()
            for i, (images, imagespath) in enumerate(self.infer_loader):
                #print(imagespath)
                images = images.to(self.device)
                SR = torch.sigmoid(self.unet(images))
                
                pp1=os.path.join(self.result_path,"fingerprint")
                pp2=os.path.join(self.result_path,"fingervein")
                pp3=os.path.join(self.result_path,"palmprint")
                if not os.path.exists(pp1):
                    os.mkdir(pp1)
                if not os.path.exists(pp2):
                    os.mkdir(pp2)
                if not os.path.exists(pp3):
                    os.mkdir(pp3)
                for j in range(self.batch_size):
                  t1=imagespath[j].split("/")[-1]
                  t2=imagespath[j].split("/")[-2]
                  p1=os.path.join(pp1,t2)
                  p2=os.path.join(pp2,t2)
                  p3=os.path.join(pp3,t2)
                  if not os.path.exists(p1):
                    os.mkdir(p1)
                  if not os.path.exists(p3):
                    os.mkdir(p3)
                  if not os.path.exists(p2):
                    os.mkdir(p2)
                  t=SR.cpu().detach().numpy()[0]
                  cv2.imwrite(os.path.join(p1,t1),t[0]*255)
                  cv2.imwrite(os.path.join(p2,t1),t[1]*255)
                  cv2.imwrite(os.path.join(p3,t1),t[2]*255)
            T2=time.perf_counter()
            print("total time %d ms"%((T2-T1)*1000))       
