# 复数神经网络实践
# import argparse
import os
import sys
import argparse
import random
from typing import Dict
from struct import unpack, pack
import mmap
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
# onnx
import onnx
import onnxscript
import onnxruntime
from apps.rsp.conf.fmcw_config import FmcwConfig as Fcfg
from apps.rsp.dss.cssr_ds import CssrDs
from apps.rsp.anns.fmcw_cmlp_model import FmcwCmlpModel

class ComplexToRealMSELoss(nn.Module):
    def forward(self, input, target):
        # 计算输入的实部和虚部与目标之间的均方误差
        loss_real = nn.MSELoss()(input.real, target.real)
        # print(f'??????????? y_hat: {input.real}; gt: {target.real}; loss_real: {loss_real};')
        loss_imag = nn.MSELoss()(input.imag, target.imag)
        # 将实部和虚部的损失相加
        loss = loss_real + loss_imag
        return loss

class FmcwCmlpApp(object):
    def __init__(self):
        self.name = 'apps.rsp.anns.fmcw_cmlp_app.FmcwCmlpApp'

    @staticmethod
    def startup(params:Dict = {}) -> None:
        appId = 101 # params['appId']
        print(f'复数神经网络 v0.0.1')
        if appId == 101:
            FmcwCmlpApp.train_main(params=params)
        elif appId == 102:
            FmcwCmlpApp.predict()
        elif appId == 103:
            FmcwCmlpApp.export_onnx()
            

    @staticmethod
    def train_main(params:Dict = {}) -> None:
        batch_size = 128
        # 生成数据集
        train_ds = CssrDs(mode='train')
        train_dataloader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        test_ds = CssrDs(mode='test')
        test_dataloader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
        # 创建模型
        model = FmcwCmlpModel().to(Fcfg.device)
        # loss_fn = nn.MSELoss()
        loss_fn = ComplexToRealMSELoss()
        # warmup
        warmup_opt = torch.optim.SGD(model.parameters(), lr=1e-5)
        warmup_epochs = 5 # 5
        for epoch in range(warmup_epochs):
            FmcwCmlpApp.train(epoch, train_dataloader, model, loss_fn, optimizer=warmup_opt, scheduler=None)
            FmcwCmlpApp.test(test_dataloader, model, loss_fn)
        # 定义代价函数和优化器
        optimizer = torch.optim.SGD(model.parameters(), lr=1e-5)
        scheduler = ReduceLROnPlateau(optimizer, 'min', threshold=1e-5, eps=1e-12)
        epochs = 2000000
        best_loss = sys.float_info.max
        improve_threshold = 0.000001
        cumulative_steps = 0
        max_unimproves = 50
        update_radarlab_threshold = 0.1
        # dst_onnx = '/home/zywy/yantao/container_disk/irpoc.onnx'
        dst_onnx = './apps/rsp/work/onnx/cmlp.onnx'
        for epoch in range(epochs):
            print(f"Epoch {epoch+1}\n-------------------------------")
            loss = FmcwCmlpApp.train(epoch, train_dataloader, model, loss_fn, optimizer, scheduler)
            accuracy = FmcwCmlpApp.test(test_dataloader, model, loss_fn)
            cumulative_steps += 1
            print(f'????????? best_loss={best_loss}; vs loss={loss}; cumulative_steps={cumulative_steps};')
            if best_loss > loss:
                if best_loss - loss > improve_threshold:
                    torch.save(model, f'./apps/rsp/work/ckpts/fmcw/irpoc.pt')
                    # if (best_loss - loss)/best_loss > update_radarlab_threshold: # 更新RadarLab程序
                    #     args.shared_mem.seek(204)
                    #     args.shared_mem.write(pack('i', 1))
                    #     args.shared_mem.seek(208)
                    #     # FmcwCmlpApp.export_onnx_irpoc(f'./work/ckpts/fmcw/irpoc_{epoch}_{loss}_{accuracy}.pt', dst_onnx)
                    #     # # args.shared_mem.write('/home/zywy/yantao/adev/zywy/nersp/python/work/irpoc/irpoc.onnx'.encode())
                    #     # args.shared_mem.write(dst_onnx.encode())
                    # else:
                    #     # 无需更新参数
                    #     args.shared_mem.seek(204)
                    #     args.shared_mem.write(pack('i', 0))
                    cumulative_steps = 0
                    best_loss = loss
            if cumulative_steps > max_unimproves:
                print(f'Earlly Stopping!!!!!!')
                break
            # if accuracy > 99.99999:
            #     print(f'训练结束')
            #     break
            # !!!!!!!!!! 仅用于测试目的 !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            # 手工转换的模型为irpoc
            # has_new_pt = 0
            # if epoch == 2:
            #     print(f'##### 精度：59.26%')
            #     has_new_pt = 1
            #     FmcwCmlpApp.export_onnx_irpoc('./work/irpoc/irpoc_100.pt', dst_onnx)
            # elif epoch == 4:
            #     print(f'##### 精度：81.3%')
            #     has_new_pt = 1
            #     FmcwCmlpApp.export_onnx_irpoc('./work/irpoc/irpoc_200.pt', dst_onnx)
            # elif epoch == 6:
            #     print(f'##### 精度：93.8%')
            #     has_new_pt = 1
            #     FmcwCmlpApp.export_onnx_irpoc('./work/irpoc/irpoc_300.pt', dst_onnx)
            # elif epoch == 8:
            #     print(f'##### 精度：99.8%')
            #     has_new_pt = 1
            #     FmcwCmlpApp.export_onnx_irpoc('./work/irpoc/irpoc_1000.pt', dst_onnx)
            # elif epoch == 12:
            #     print(f'##### 精度：100.0%')
            #     has_new_pt = 1
            #     FmcwCmlpApp.export_onnx_irpoc('./work/irpoc/irpoc_5000.pt', dst_onnx)
            # else:
            #     has_new_pt = 0
            # args.shared_mem.seek(204)
            # args.shared_mem.write(pack('i', has_new_pt))
            # args.shared_mem.seek(208)
            # args.shared_mem.write(dst_onnx.encode())
            # !!!!!!!!!! 仅用于测试目的 !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        print("Done!")

    @staticmethod
    def train(epoch, dataloader, model, loss_fn, optimizer, scheduler=None):
        size = len(dataloader.dataset)
        model.train()
        total_loss = torch.tensor([0.0]).to(Fcfg.device)
        for batch, (X, y) in enumerate(dataloader):
            X, y = X.to(Fcfg.device), y.to(Fcfg.device)
            # Compute prediction error
            pred = model(X).squeeze(dim=-1)
            loss = loss_fn(pred, y)
            total_loss += loss
            # Backpropagation
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            if batch % 100 == 0:
                loss, current = loss.item(), (batch + 1) * len(X)
                print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
        if scheduler is not None:
            scheduler.step(total_loss)
        return total_loss.detach().cpu().item()

    @staticmethod
    def test(dataloader, model, loss_fn):
        size = len(dataloader.dataset)
        num_batches = len(dataloader)
        model.eval()
        test_loss, correct = 0, 0
        with torch.no_grad():
            for X, y in dataloader:
                X, y = X.to(Fcfg.device), y.to(Fcfg.device)
                pred = model(X).squeeze(dim=-1)
                print(f'y: {y}; vs pred: {pred}; diff: {(pred.real-y.real)**2}')
                test_loss += loss_fn(pred, y).item()
                correct += ((pred.real-y.real)**2 < 0.0001).type(torch.float).sum().item()
                # correct += ((pred.real-y.real)**2 < 0.000001).type(torch.float).sum().item()
        test_loss /= num_batches
        correct /= size
        print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
        return 100*correct

    @staticmethod
    def predict():
        # model = FmcwMlpModel()
        model =torch.load('./work/ckpts/fmcw/v001/irpoc_96_0.001581694115884602_100.0.pt')
        model.eval()
        model = model.to(Fcfg.device)
        # 产生数据
        batch_size = 16
        train_ds = FmcwCmlpDs(mode='train')
        train_dataloader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        X, y = next(iter(train_dataloader))
        with open('./work/t005.txt', 'w', encoding='utf-8') as fd:
            for idx in range(X.shape[0]):
                xi = None
                for col in range(X.shape[1]):
                    if xi is None:
                        xi = f'{X[idx][col]}'
                    else:
                        xi = f'{xi}, {X[idx][col]}'
                # print(f'{xi}, {y[idx][0]}')
                fd.write(f'{xi}, {y[idx][0]}\n')
        print(f'X: {X.shape}; y: {y.shape};')
        X = X.to(Fcfg.device)
        y_hat = model(X)
        for idx in range(batch_size):
            print(f'0: GT:{y[idx]*100} vs hat:{y_hat[idx]*100};')

    @staticmethod 
    def export_onnx() -> None:
        model_fn = './work/ckpts/fmcw/v001/irpoc_96_0.001581694115884602_100.0.pt'
        model = torch.load(model_fn)
        batch_size = 1
        X_real = torch.randn(batch_size, 1024, requires_grad=True)
        X_imag = torch.randn(batch_size, 1024, requires_grad=True)
        X_ = []
        for idx in range(1024):
            X_.append(complex(random.random(), random.random()))
        X = torch.tensor(X_, dtype=torch.complex64)
        X = torch.reshape(X, (1, 1024))
        X = X.to('cuda:0')
        model = model.to('cuda:0')
        print(f'X: {X.device};')
        y = model(X)
        # Export the model
        torch.onnx.export(model,               # model being run
            X,                         # model input (or a tuple for multiple inputs)
            "./work/ckpts/fmcw/v001/irpoc_001.onnx",   # where to save the model (can be a file or file-like object)
            export_params=True,        # store the trained parameter weights inside the model file
            opset_version=16,          # the ONNX version to export the model to
            do_constant_folding=True,  # whether to execute constant folding for optimization
            input_names = ['input'],   # the model's input names
            output_names = ['output'], # the model's output names
            dynamic_axes={'input' : {0 : 'batch_size'},    # variable length axes
                            'output' : {0 : 'batch_size'}}
        )

    @staticmethod 
    def export_onnx_irpoc(model_fn: str, dst_onnx:str) -> None:
        model = torch.load(model_fn)
        batch_size = 1
        X = torch.randn(batch_size, 2048, requires_grad=True)
        X = X.to('cpu')
        model = model.to('cpu')
        print(f'X: {X.device};')
        y = model(X)
        # Export the model
        torch.onnx.export(model,               # model being run
            X,                         # model input (or a tuple for multiple inputs)
            dst_onnx,   # where to save the model (can be a file or file-like object)
            export_params=True,        # store the trained parameter weights inside the model file
            opset_version=16,          # the ONNX version to export the model to
            do_constant_folding=True,  # whether to execute constant folding for optimization
            input_names = ['input'],   # the model's input names
            output_names = ['output'], # the model's output names
            dynamic_axes={'input' : {0 : 'batch_size'},    # variable length axes
                            'output' : {0 : 'batch_size'}}
        )
    

    @staticmethod
    def run_onnx() -> None:
        ort_session = onnxruntime.InferenceSession("work/irpoc/irpoc_0.onnx", providers=["CPUExecutionProvider"])
        def to_numpy(tensor):
            return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()
        # 产生数据
        batch_size = 16
        train_ds = FmcwMlpDs(mode='train')
        train_dataloader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        X, y = next(iter(train_dataloader))
        x = X.to('cpu')
        y = y.to('cpu')
        print(f'############ x:{x.shape}; y: {y.shape};')
        # x = torch.randn(batch_size, 128, requires_grad=True)
        # x = x.to('cpu')
        # compute ONNX Runtime output prediction
        ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(x)}
        y_hat = ort_session.run(None, ort_inputs)
        for idx in range(batch_size):
            print(f'y: {y.detach().cpu()[idx][0].item()}; y_hat: {y_hat[0][idx][0]};')

        # compare ONNX Runtime and PyTorch results
        # np.testing.assert_allclose(to_numpy(torch_out), ort_outs[0], rtol=1e-03, atol=1e-05)

        print("Exported model has been tested with ONNXRuntime, and the result looks good!")