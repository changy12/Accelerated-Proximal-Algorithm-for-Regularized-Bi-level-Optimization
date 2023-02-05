import os
import numpy as np      #In /Users/chenziyi/anaconda3/lib/python3.7/site-packages
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn
import torch.optim
import argparse
import hypergrad as hg
import time
from itertools import repeat
from itertools import cycle
from torch.nn import functional as F
from torchvision import datasets
from stocBiO import *
from ARGS import ARGS
import matplotlib.pyplot as plt
import pdb

def get_data_loaders(args):   
    kwargs = {'num_workers': 0, 'pin_memory': True}
    dataset = datasets.MNIST(root=args.data_path, train=True, download=True,
                        transform=transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize((0.1307,), (0.3081,))
                        ]))    #<class 'torchvision.datasets.mnist.MNIST'>

    dataset=[data for data in dataset]
    
    num = args.noise_rate*(args.training_size)
    num = int(num)
    randint = torch.randint(1, 10, (num,))   
    index = torch.randperm(args.training_size)[:num]
    i=0
    for k in index:
        dataset[k] = (dataset[k][0], (dataset[k][1]+randint[i].item())%args.num_classes)
        i+=1

    train_sampler = torch.utils.data.sampler.SequentialSampler(dataset)  #Samples elements sequentially, always in the same order.
    train_loader = torch.utils.data.DataLoader(dataset, sampler=train_sampler,
        batch_size=args.batch_size, **kwargs)
    test_loader = torch.utils.data.DataLoader(datasets.MNIST(root=args.data_path, train=False,
                        download=True, transform=transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize((0.1307,), (0.3081,))
                        ])), batch_size=args.test_size)    
    
    return train_loader, test_loader

def prox_lambda_hardreg(lambda0,outreg_bounds):
    lambda1=lambda0.clone()
    half=(outreg_bounds[0]+outreg_bounds[1])/2
    lambda1[(lambda0>outreg_bounds[0]) & (lambda0<half)]=outreg_bounds[0]
    lambda1[(lambda0>=half) & (lambda0<outreg_bounds[1])]=outreg_bounds[1]
    return lambda1

def prox_lambda(lambda0,outer_lr,outreg_coeff,outreg_thr):  
    #return argmin_x ||x-lambda0||^2-outer_lr*outreg_coeff*sum_i min(|xi|,outreg_thr)
    lambda1=lambda0.clone()
    c1=outer_lr*outreg_coeff
    c2=outreg_thr-c1
    
    i=(lambda0>-outreg_thr) & (lambda0<=-c2)
    lambda1[i]=-outreg_thr
    i=(lambda0<outreg_thr) & (lambda0>=c2)
    lambda1[i]=outreg_thr
    
    i=(lambda0<0) & (lambda0>-c2)
    lambda1[i]-=c1
    i=(lambda0>=0) & (lambda0<c2)
    lambda1[i]+=c1
    
    return lambda1

def train_model(args, train_loader, test_loader, parameters0):
    parameters=parameters0.clone()
    lambda_x = torch.zeros((args.training_size), requires_grad=True).to(device)
    
    if args.save_paras:
        para_np=np.zeros((args.epochs+1,)+parameters.shape)
        hpara_np=np.zeros((args.epochs+1,args.training_size))
        para_np[0]=parameters.detach().clone()
        hpara_np[0]=lambda_x.detach().clone()
                
    loss_time_results = np.zeros((args.epochs+1, 8))
    batch_num = args.training_size//args.batch_size
    train_loss_avg = loss_train_avg(train_loader, parameters, device, batch_num)
    val_loss_avg = loss_val_avg(train_loader, parameters, device)
    test_loss_avg, test_acc_avg = loss_test_avg(test_loader, parameters, device)
    if args.outreg_bounds is None:   #Soft regularizer
        reg_val=-np.minimum(np.abs(lambda_x.detach().requires_grad_(False)),args.outreg_thr).sum()*args.outreg_coeff
    else:
        reg_val=1e+100
    loss_time_results[0, :] = [train_loss_avg, test_loss_avg, (0.0), (0.0), (0), reg_val, val_loss_avg, test_acc_avg]
    print('Epoch: {:d} Val Loss: {:.4f} Test Loss: {:.4f}'.format(0, val_loss_avg, test_loss_avg))
   
    images_list, labels_list = [], []       #list of 3 sets of training samples    #The 1st set is used for evaluating the train_loss_avg to be plotted.
    for index, (images, labels) in enumerate(train_loader):
        images_list.append(images)
        labels_list.append(labels)
        if index>=2:
            break

    # setting for reverse, fixed_point & CG
    def loss_inner(parameters, weight, data_all):
        data = data_all[0]
        labels = data_all[1]
        data = torch.reshape(data, (data.size()[0],-1)).to(device)
#        labels_cp = nositify(labels, args.noise_rate, args.num_classes).to(device)
        output = torch.matmul(data, torch.t(parameters[0][:, 0:784]))+parameters[0][:, 784]
#        loss = F.cross_entropy(output, labels_cp, reduction='none')
        loss = F.cross_entropy(output, labels, reduction='none')
        loss_regu = torch.mean(torch.mul(loss, torch.sigmoid(weight[0]))) + args.inreg_coeff*torch.pow(torch.norm(parameters[0]),2)
        return loss_regu

    def loss_outer(parameters, lambda_x):   #Unregularized
        images, labels = images_list[-1], labels_list[-1]
        images = torch.reshape(images, (images.size()[0],-1)).to(device)
        images_temp, labels_temp = images[0:args.validation_size,:], labels[0:args.validation_size]
        # images = torch.cat([images_temp]*(args.training_size // args.validation_size))
        # labels = torch.cat([labels_temp]*(args.training_size // args.validation_size)) 
        output = torch.matmul(images, torch.t(parameters[0][:, 0:784]))+parameters[0][:, 784]
        loss = F.cross_entropy(output, labels)
        return loss

    def out_f(data, parameters):
        output = torch.matmul(data, torch.t(parameters[0][:, 0:784]))+parameters[0][:, 784]
        return output

    def reg_f(params, hparams, loss):
        loss_regu = torch.mean(torch.mul(loss, torch.sigmoid(hparams))) + 0.001*torch.pow(torch.norm(params[0]),2)
        return loss_regu

    tol = 1e-12
    warm_start = True
    params_history = []
    train_iterator = repeat([images_list[0], labels_list[0]])  
    inner_opt = hg.GradientDescent(loss_inner, args.inner_lr, data_or_iter=train_iterator)
    inner_opt_cg = hg.GradientDescent(loss_inner, 1., data_or_iter=train_iterator)
    outer_opt = torch.optim.SGD(lr=args.outer_lr, params=[lambda_x])
    
    start_time = time.time() 
    lambda_index_outer = 0
    for epoch in range(args.epochs):
        grad_norm_inner = 0.0
        if args.alg == 'stocBiO' or args.alg == 'HOAG':
            train_index_list = torch.randperm(batch_num)
            for index in range(args.iterations):
                index_rn = train_index_list[index%batch_num]
                images, labels = images_list[index_rn], labels_list[index_rn]
                images = torch.reshape(images, (images.size()[0],-1)).to(device)
#                labels_cp = nositify(labels, args.noise_rate, args.num_classes).to(device)
                weight = lambda_x[index_rn*args.batch_size: (index_rn+1)*args.batch_size]
                output = out_f(images, [parameters])
                inner_update = gradient_gy(args, labels_cp, parameters, images, weight, output, reg_f)
                parameters = parameters - args.inner_lr*inner_update
        
        if args.alg == 'stocBiO':
            val_index = torch.randperm(args.validation_size//args.batch_size)
            val_data_list = build_val_data(args, val_index, images_list, labels_list, device)
            hparams = lambda_x[lambda_index_outer: lambda_index_outer+args.batch_size]
            outer_update = stocbio([parameters], hparams, val_data_list, args, out_f, reg_f)
        
        elif args.alg == 'HOAG':
            images, labels = images_list[-1], labels_list[-1]
            images = torch.reshape(images, (images.size()[0],-1)).to(device)
            images_temp, labels_temp = images[0:args.validation_size,:], labels[0:args.validation_size]
            images = torch.cat([images_temp]*(args.training_size // args.validation_size))
            labels = torch.cat([labels_temp]*(args.training_size // args.validation_size))
            
            labels = labels.to(device)
#            labels_cp = nositify(labels, args.noise_rate, args.num_classes).to(device)
            val_images_list, val_labels_list = [images,images,images], [labels,labels_cp,labels_cp]         
            val_data_list = [val_images_list, val_labels_list]
            outer_update = stocbio([parameters], lambda_x, val_data_list, args, out_f, reg_f)

        elif args.alg == 'BSA' or args.alg == 'TTSA':
            train_index_list = torch.randperm(args.training_size)
            random_list = np.random.uniform(size=[args.training_size])
            noise_rate_list = np.where((random_list>args.noise_rate), 0, 1)
            for index in range(args.iterations):
                images, labels = images_list[train_index_list[index]], labels_list[train_index_list[index]]
                images = torch.reshape(images, (images.size()[0],-1)).to(device)
#                labels_cp = nositify(labels, noise_rate_list[index], args.num_classes).to(device)
                weight = lambda_x[train_index_list[index]: train_index_list[index]+1]
                output = out_f(images, [parameters])
                inner_update = gradient_gy(args, labels_cp, parameters, images, weight, output, reg_f)
                parameters = parameters - args.inner_lr*inner_update

            val_index = -torch.randperm(args.validation_size)
            random_list = np.random.uniform(size=[args.hessian_q+2])
            noise_rate_list = np.where((random_list>args.noise_rate), 0, 1)

            val_images_list, val_labels_list = [], []
            images, labels = images_list[val_index[1]], labels_list[val_index[1]]
            images = torch.reshape(images, (images.size()[0],-1)).to(device)
            val_images_list.append(images)
            val_labels_list.append(labels.to(device))
            images, labels = images_list[val_index[2]], labels_list[val_index[2]]
            images = torch.reshape(images, (images.size()[0],-1)).to(device)
#            labels_cp = nositify(labels, noise_rate_list[1], args.num_classes).to(device)
            val_images_list.append(images)
            val_labels_list.append(labels)
            images, labels = images_list[val_index[3]], labels_list[val_index[3]]
            images = torch.reshape(images, (images.size()[0],-1)).to(device)
#            labels_cp = nositify(labels, noise_rate_list[2], args.num_classes).to(device)
            val_images_list.append(images)
            val_labels_list.append(labels)
            val_data_list = [val_images_list, val_labels_list]

            hyparams = lambda_x[lambda_index_outer: lambda_index_outer+args.batch_size]
            outer_update = stocbio([parameters], hyparams, val_data_list, args, out_f, reg_f)

        else:
            if params_history:
                params_history = [params_history[-1]]
            else:
                params_history = [[parameters]]
            for index in range(args.iterations):
                params_new=inner_opt(params_history[-1], [lambda_x], create_graph=False)
                if args.momentum_coeff>0:
                    params_new=[new+args.momentum_coeff*(new-old) for old,new in zip(params_history[-1],params_new)]
                params_history.append(params_new)

            final_params = params_history[-1]
            outer_opt.zero_grad()
            if 'reverse' in args.alg:
                hg.reverse(params_history[-args.hessian_q-1:], [lambda_x], [inner_opt]*args.hessian_q, loss_outer)
            elif args.alg == 'AID-FP':
                hg.fixed_point(final_params, [lambda_x], args.hessian_q, inner_opt, loss_outer, stochastic=False, tol=tol)
            elif 'AID-CG' in args.alg:
                hg.CG(final_params[:len(parameters)], [lambda_x], args.hessian_q, inner_opt_cg, loss_outer, stochastic=False, tol=tol)    
            outer_update = lambda_x.grad
            weight = lambda_x[lambda_index_outer: lambda_index_outer+args.batch_size]
        
        outer_update = torch.squeeze(outer_update)
        if args.outreg_bounds is None:   #Soft regularizer
            if 'prox' in args.alg: 
                with torch.no_grad():
                    weight = weight - args.outer_lr*outer_update
                    if args.outreg_coeff!=0:
                        weight=prox_lambda(weight,args.outer_lr,args.outreg_coeff,args.outreg_thr)
            else:
                with torch.no_grad():
                    i=(torch.abs(weight)<args.outreg_thr)
                    outer_update[i]=outer_update[i]-args.outreg_coeff*torch.sign(weight[i])
                    weight = weight - args.outer_lr*outer_update
        else:   #Hard regularizer
            if 'prox' in args.alg:  
                with torch.no_grad():
                    weight = weight - args.outer_lr*outer_update
                    weight=prox_lambda_hardreg(weight,args.outreg_bounds)
                    
        lambda_x = torch.zeros((args.training_size), requires_grad=False)
        lambda_x[lambda_index_outer: lambda_index_outer+args.batch_size]=weight  #Added by me     
        lambda_x.requires_grad=True
        lambda_x = lambda_x.to(device)
        
        lambda_index_outer = (lambda_index_outer+args.batch_size) % args.training_size

        if ('reverse' in args.alg) or ('AID-CG' in args.alg) or (args.alg == 'AID-FP'):
            train_loss_avg = loss_train_avg(train_loader, final_params[0], device, batch_num)
            test_loss_avg, test_acc_avg = loss_test_avg(test_loader, final_params[0], device)
            val_loss_avg = loss_val_avg(train_loader, final_params[0], device)
        else:
            train_loss_avg = loss_train_avg(train_loader, parameters, device, batch_num)
            test_loss_avg, test_acc_avg = loss_test_avg(test_loader, parameters, device)
            val_loss_avg = loss_val_avg(train_loader, parameters, device)
        
        if args.outreg_bounds is None:   #Soft regularizer
            reg_val=-np.minimum(np.abs(weight),args.outreg_thr).sum()*args.outreg_coeff
        else:
            reg_val=1e+100
        end_time = time.time()

        loss_time_results[epoch+1, 0] = train_loss_avg
        loss_time_results[epoch+1, 1] = test_loss_avg
        loss_time_results[epoch+1, 2] = (end_time-start_time)
        loss_time_results[epoch+1, 3] = grad_norm_inner         #Always 0
        if args.outreg_bounds is not None:   #Use hard regularizer
            loss_time_results[epoch+1, 4] = 1 if torch.all((weight<=args.outreg_bounds[0])|(weight>=args.outreg_bounds[1])) else 0       
            print('Epoch: {:d} Val Loss: {:.4f} Test Loss: {:.4f} Time: {:.4f} Lambda all outside [{},{}]: {}'.format(epoch+1, val_loss_avg, test_loss_avg,
                            (end_time-start_time), args.outreg_bounds[0],args.outreg_bounds[1],loss_time_results[epoch+1, 4]))
        else:
            loss_time_results[epoch+1, 4] = 1 if torch.all(torch.abs(weight)>=args.outreg_thr) else 0     
            print('Epoch: {:d} Val Loss: {:.4f} Test Loss: {:.4f} Time: {:.4f} Lambda all outside [{},{}]: {}'.format(epoch+1, val_loss_avg, test_loss_avg,
                            (end_time-start_time),-args.outreg_thr,args.outreg_thr,loss_time_results[epoch+1, 4]))
        loss_time_results[epoch+1, 5]=reg_val
        loss_time_results[epoch+1, 6]=val_loss_avg
        loss_time_results[epoch+1, 7]=test_acc_avg
        
        if args.save_paras:
            hpara_np[epoch+1]=weight
            if ('reverse' in args.alg) or ('AID-CG' in args.alg) or (args.alg == 'AID-FP'):
                para_np[epoch+1]=final_params[0].detach().clone()
            else:
                para_np[epoch+1]=parameters.detach().clone()

    print(loss_time_results)
    file_name = str(args.seed)+'.npy'
    file_addr = os.path.join(args.save_folder, file_name)
    with open(file_addr, 'wb') as f:
        np.save(f, loss_time_results)
            
    if args.save_paras:
        file_name = str(args.seed)+'para.npy'
        file_addr = os.path.join(args.save_folder, file_name)
        with open(file_addr, 'wb') as f:
            np.save(f, para_np)
        
        file_name = str(args.seed)+'hpara.npy'
        file_addr = os.path.join(args.save_folder, file_name)
        with open(file_addr, 'wb') as f:
            np.save(f, hpara_np)
        
        return loss_time_results, para_np, hpara_np
    else:
        return loss_time_results

def loss_train_avg(data_loader, parameters, device, batch_num):
    loss_avg, num = 0.0, 0
    for index, (images, labels) in enumerate(data_loader):
        if index>= batch_num:
            break
        else:
            images = torch.reshape(images, (images.size()[0],-1)).to(device)   #Strecth into 20000 training samples * 784 dim
            labels = labels.to(device)
            loss = loss_f_function(labels, parameters, images)
            loss_avg += loss 
            num += 1
    loss_avg = loss_avg/num
    return loss_avg.detach()

def loss_val_avg(data_loader, parameters, device):
    loss_avg, num = 0.0, 0
    for index, (images, labels) in enumerate(data_loader):
        if index==0:
            continue
        
        if index>= 2:
            break
        else:
            images = torch.reshape(images, (images.size()[0],-1)).to(device)   #Strecth into 20000 validation samples * 784 dim
            labels = labels.to(device)
            loss = loss_f_function(labels, parameters, images)
            loss_avg += loss 
            num += 1
    loss_avg = loss_avg/num
    return loss_avg.detach()

def loss_test_avg(data_loader, parameters, device):
    loss_avg, num, acc_avg = 0.0, 0, 0.0
    for _, (images, labels) in enumerate(data_loader):
        images = torch.reshape(images, (images.size()[0],-1)).to(device)  #Strecth into 1000 test samples * 784 dim
        # images = torch.cat((images, torch.ones(images.size()[0],1)),1)
        labels = labels.to(device)
        loss, acc = loss_f_function(labels, parameters, images, True)
        loss_avg += loss 
        acc_avg += acc
        num += 1
    loss_avg = loss_avg/num
    acc_avg = acc_avg/num
    return loss_avg.detach(),acc.detach()

def loss_f_function(labels, parameters, data, is_acc=False):
    output = torch.matmul(data, torch.t(parameters[:, 0:784]))+parameters[:, 784]
    loss = F.cross_entropy(output, labels)
    if is_acc:
        acc = ((torch.argmax(output,1)==labels)+0.0).mean()
        return loss, acc
    else:
        return loss

def nositify(labels, noise_rate, n_class):
    num = noise_rate*(labels.size()[0])
    num = int(num)
    randint = torch.randint(1, 10, (num,))   
    index = torch.randperm(labels.size()[0])[:num]
    labels=labels.clone()
    labels[index] = (labels[index]+randint) % n_class
    return labels

def build_val_data(args, val_index, images_list, labels_list, device):
    val_index = -(val_index)
    val_images_list, val_labels_list = [], []
    
    images, labels = images_list[val_index[0]], labels_list[val_index[0]]
    images = torch.reshape(images, (images.size()[0],-1)).to(device)
    labels = labels.to(device)
    val_images_list.append(images)
    val_labels_list.append(labels)

    images, labels = images_list[val_index[1]], labels_list[val_index[1]]
    images = torch.reshape(images, (images.size()[0],-1)).to(device)
    labels_cp = nositify(labels, args.noise_rate, args.num_classes).to(device)
    val_images_list.append(images)
    val_labels_list.append(labels_cp)

    images, labels = images_list[val_index[2]], labels_list[val_index[2]]
    images = torch.reshape(images, (images.size()[0],-1)).to(device)
    labels_cp = nositify(labels, args.noise_rate, args.num_classes).to(device)
    val_images_list.append(images)
    val_labels_list.append(labels_cp)

    return [val_images_list, val_labels_list]

def plot_txt(algs,legends,colors,linestyles,markers,plot_folder,seeds,outreg_coeff,noise_rate,txt_name=None,fig_names=\
             ['ValLoss_unreg','TestLoss_unreg','reg_val','ValLoss_reg','TestLoss_reg']):
    if txt_name is not None:
        hyp_txt=open(txt_name,'w')
    for alg,color,linestyle,marker,legend in zip(algs,colors,linestyles,markers,legends):
        seed_i=0
        violation_percent=np.zeros(len(seeds))
        time_avg=0.0
        for seed in seeds:
            if 'AID-CG' in alg:
                hessian_q=hessian_q_AID
            else:
                hessian_q=hessian_q_ITD
            
            if 'momentum' in alg:
                if 'AID-CG' in alg:
                    momentum_coeff=momentum_coeff_AID
                else:
                    momentum_coeff=momentum_coeff_ITD
            else:
                momentum_coeff=0.0
            
            if 'unreg' in alg:
                args=ARGS(noise_rate=noise_rate,outreg_coeff=0.0,hessian_q=hessian_q,momentum_coeff=momentum_coeff,seed=seed,alg=alg)
            else:
                args=ARGS(noise_rate=noise_rate,hessian_q=hessian_q,momentum_coeff=momentum_coeff,seed=seed,alg=alg,outreg_coeff=outreg_coeff)
            
            print('\n folder: '+args.save_folder)
            file_name = str(seed)+'.npy'
            file_addr = os.path.join(args.save_folder, file_name)
            loss_time_results=np.load(file_addr)
            
            if seed_i==0:
                train_loss=np.zeros((len(seeds),loss_time_results.shape[0]))
                val_loss=np.zeros((len(seeds),loss_time_results.shape[0]))
                test_loss=np.zeros((len(seeds),loss_time_results.shape[0]))
                reg_val=np.zeros((len(seeds),loss_time_results.shape[0]))
                test_acc=np.zeros(len(seeds))
            train_loss[seed_i]=loss_time_results[:,0]
            val_loss[seed_i]=loss_time_results[:,6]
            test_loss[seed_i]=loss_time_results[:,1]
            reg_val[seed_i]=loss_time_results[:,5]
            test_acc[seed_i]=loss_time_results[-1,7]
            tmp=loss_time_results[:,4].copy()
            violation_percent[seed_i]=1-tmp[1:len(tmp)].mean()
            time_avg+=loss_time_results[-1,2]
            seed_i+=1
        time_avg/=seed_i
    
        if txt_name is not None:
            args_totxt=['inreg_coeff','outreg_bounds','outreg_coeff','outreg_thr','epochs',\
                 'iterations','outer_lr','inner_lr','momentum_coeff','data_path','save_folder']
            toprint=alg+': violation percent:'
            for ele in violation_percent:
                toprint+=str(ele)+', '
            toprint+='avg='+str(violation_percent.mean())+', std='+str(violation_percent.std())+'.\n'
            hyp_txt.write(toprint)
            hyp_txt.write('Time consumption: '+str(time_avg/60)+'minutes. \n')
            
            toprint='test acc (last iterate)='
            for ele in test_acc:
                toprint+=str(ele)+', '
            toprint+='avg='+str(test_acc.mean())+', std='+str(test_acc.std())+'.\n'
            hyp_txt.write(toprint)
            
            toprint='test loss (last iterate)='
            test_loss_final=test_loss[:,-1].copy()
            for ele in test_loss_final:
                toprint+=str(ele)+', '
            toprint+='avg='+str(test_loss_final.mean())+', std='+str(test_loss_final.std())+'.\n'
            hyp_txt.write(toprint)
            
            for feature_name in args_totxt:
                hyp_txt.write(feature_name+':'+str(getattr(args,feature_name))+',  ')
            hyp_txt.write('.\n\n')
                
        x=[k*args.iterations for k in range(args.epochs+1)]
        if fig_names[0] is not None:
            plt.figure(1)
            upper_loss = np.percentile(val_loss, percentile, axis=0)  
            lower_loss = np.percentile(val_loss, 100 - percentile, axis=0)
            avg_loss = np.mean(val_loss, axis=0)
            plt.plot(x,avg_loss,color=color,linestyle=linestyle,marker=marker,label=legend)
            plt.fill_between(x,lower_loss,upper_loss, color=color,alpha=0.3)
            plt.xscale('log')
            plt.yscale('log')
            
        if fig_names[1] is not None:
            plt.figure(2)
            upper_loss = np.percentile(test_loss, percentile, axis=0)  
            lower_loss = np.percentile(test_loss, 100 - percentile, axis=0)
            avg_loss = np.mean(test_loss, axis=0)
            plt.plot(x,avg_loss,color=color,linestyle=linestyle,marker=marker,label=legend)
            plt.fill_between(x,lower_loss,upper_loss, color=color,alpha=0.3)
            # plt.xscale('log')
            # plt.yscale('log')
        
        if fig_names[2] is not None:
            plt.figure(3)
            upper_loss = np.percentile(reg_val, percentile, axis=0)  
            lower_loss = np.percentile(reg_val, 100 - percentile, axis=0)
            avg_loss = np.mean(reg_val, axis=0)
            plt.plot(x,avg_loss,color=color,linestyle=linestyle,marker=marker,label=legend)
            plt.fill_between(x,lower_loss,upper_loss, color=color,alpha=0.3)
            # plt.xscale('log')
            # plt.yscale('log')
    
        if fig_names[3] is not None:
            plt.figure(4)
            val_loss+=reg_val
            upper_loss = np.percentile(val_loss, percentile, axis=0)  
            lower_loss = np.percentile(val_loss, 100 - percentile, axis=0)
            avg_loss = np.mean(val_loss, axis=0)
            plt.plot(x,avg_loss,color=color,linestyle=linestyle,marker=marker,label=legend)
            plt.fill_between(x,lower_loss,upper_loss, color=color,alpha=0.3)
            plt.xscale('log')
            plt.yscale('log')
    
        if fig_names[4] is not None:
            plt.figure(5)
            test_loss+=reg_val
            upper_loss = np.percentile(test_loss, percentile, axis=0)  
            lower_loss = np.percentile(test_loss, 100 - percentile, axis=0)
            avg_loss = np.mean(test_loss, axis=0)
            plt.plot(x,avg_loss,color=color,linestyle=linestyle,marker=marker,label=legend)
            plt.fill_between(x,lower_loss,upper_loss, color=color,alpha=0.3)
            # plt.xscale('log')
            # plt.yscale('log')
    
#    print(alg,train_loss[:,-1].mean(),test_loss[:,-1].mean())
    if not os.path.isdir(plot_folder):
        os.makedirs(plot_folder)
    
    lgdsize=18
    labelsize=16
    ticksize=12
    titlesize=16
    bottom_size=0.17
    left_size=0.1
    
    if fig_names[0] is not None:
        plt.figure(1)
        plt.gcf().subplots_adjust(bottom=bottom_size)
        plt.gcf().subplots_adjust(left=left_size)
        plt.legend(prop={'size':lgdsize},loc=1)
        plt.xlabel('Computational complexity', fontsize=labelsize)
        plt.ylabel('Validation loss', fontsize=labelsize)
        plt.xticks(fontsize=ticksize)
        plt.yticks(fontsize=ticksize)
        plt.suptitle('p='+str(noise_rate), fontsize=titlesize)
        plt.savefig(plot_folder+'/'+fig_names[0]+'.eps',format='eps',dpi=400,bbox_inches='tight')
        plt.clf()    
    
    if fig_names[1] is not None:
        plt.figure(2)
        plt.gcf().subplots_adjust(bottom=bottom_size)
        plt.gcf().subplots_adjust(left=left_size)
        plt.legend(prop={'size':lgdsize},loc=1)
        plt.xlabel('Computational complexity', fontsize=labelsize)
        plt.ylabel('Test loss', fontsize=labelsize)
        plt.xticks(fontsize=ticksize)
        plt.yticks(fontsize=ticksize)
        plt.suptitle('p='+str(noise_rate), fontsize=titlesize)
        plt.savefig(plot_folder+'/'+fig_names[1]+'.eps',format='eps',dpi=400,bbox_inches='tight')
        plt.clf()    

    if fig_names[2] is not None:
        plt.figure(3)
        plt.gcf().subplots_adjust(bottom=bottom_size)
        plt.gcf().subplots_adjust(left=left_size)
        plt.legend(prop={'size':lgdsize},loc=1)
        plt.xlabel('Computational complexity', fontsize=labelsize)
        plt.ylabel('Regularizer value', fontsize=labelsize)
        plt.xticks(fontsize=ticksize)
        plt.yticks(fontsize=ticksize)
        plt.suptitle('p='+str(noise_rate), fontsize=titlesize)
        plt.savefig(plot_folder+'/'+fig_names[2]+'.eps',format='eps',dpi=400,bbox_inches='tight')
        plt.clf()    
    
    if fig_names[3] is not None:
        plt.figure(4)
        plt.gcf().subplots_adjust(bottom=bottom_size)
        plt.gcf().subplots_adjust(left=left_size)
        plt.legend(prop={'size':lgdsize},loc=1)
        plt.xlabel('Computational complexity', fontsize=labelsize)
        plt.ylabel('Validation loss', fontsize=labelsize)
        plt.xticks(fontsize=ticksize)
        plt.yticks(fontsize=ticksize)
        plt.suptitle('p='+str(noise_rate), fontsize=titlesize)
        plt.savefig(plot_folder+'/'+fig_names[3]+'.eps',format='eps',dpi=400,bbox_inches='tight')
        plt.clf()    
    
    if fig_names[4] is not None:
        plt.figure(5)
        plt.gcf().subplots_adjust(bottom=bottom_size)
        plt.gcf().subplots_adjust(left=left_size)
        plt.legend(prop={'size':lgdsize},loc=1)
        plt.xlabel('Computational complexity', fontsize=labelsize)
        plt.ylabel('Test loss', fontsize=labelsize)
        plt.xticks(fontsize=ticksize)
        plt.yticks(fontsize=ticksize)
        plt.suptitle('p='+str(noise_rate), fontsize=titlesize)
        plt.savefig(plot_folder+'/'+fig_names[4]+'.eps',format='eps',dpi=400,bbox_inches='tight')
        plt.clf()    
        
    if txt_name is not None:
        hyp_txt.close()

num_classes=10
momentum_coeff_AID=1.0
momentum_coeff_ITD=1.0
hessian_q_AID=200
hessian_q_ITD=3
percentile=95
noise_rates=[0.1,0.2,0.4]
# outreg_coeffs=[c/5000 for c in [0.5,5.0]]
outreg_coeffs=[c/20000 for c in [20.0,2000.0,2e+6]]

#Initialize for all algorithms
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
    
if 'parameters0' in dir():
    del parameters0
if 'train_loader' in dir():
    del train_loader
seeds=list(range(1))
parameters0=[0]*len(seeds)
torch.manual_seed(911)
for k in range(len(seeds)):
    parameters0[k] = torch.randn((num_classes, 785), requires_grad=True)
    parameters0[k] = nn.init.kaiming_normal_(parameters0[k], mode='fan_out').to(device) 
    
dir_results='./save_tb_results'
if not os.path.isdir(dir_results):
    os.makedirs(dir_results)

with open('./save_tb_results/test_results.txt','a') as TestResults_txt:
    for noise_rate in noise_rates:
        k=0
        TestResults_txt.write('\n noise rate='+str(noise_rate)+':\n')
        torch.manual_seed(1)
        for seed in seeds:
            algs=['AID-CG-unreg','AID-CG-unreg-momentum','AID-CG-prox','AID-CG-prox-momentum','reverse-unreg','reverse-unreg-momentum','reverse-prox','reverse-prox-momentum']
            num_algs=len(algs)
            for alg in algs:
                if 'AID-CG' in alg:
                    hessian_q=hessian_q_AID
                else:
                    hessian_q=hessian_q_ITD
                
                if 'momentum' in alg:
                    if 'AID-CG' in alg:
                        momentum_coeff=momentum_coeff_AID
                    else:
                        momentum_coeff=momentum_coeff_ITD
                else:
                    momentum_coeff=0.0
                
                if 'unreg' in alg:
                    args=ARGS(noise_rate=noise_rate,outreg_coeff=0.0,hessian_q=hessian_q,momentum_coeff=momentum_coeff,seed=seed,alg=alg)                
                    if 'train_loader' not in dir():
                        train_loader, test_loader = get_data_loaders(args)
                    print('\n folder: '+args.save_folder+'; the '+str(k)+'-th seed')
                    loss_time_results, _, _=train_model(args, train_loader, test_loader, parameters0[k])
                    TestResults_txt.write(str(alg)+': '+'Test accuracy='+str(loss_time_results[-1,7])+', Test loss='+str(loss_time_results[-1,1])+'\n')
                else:
                    for outreg_coeff in outreg_coeffs:
                        args=ARGS(noise_rate=noise_rate,hessian_q=hessian_q,momentum_coeff=momentum_coeff,seed=seed,alg=alg,outreg_coeff=outreg_coeff)
                        if 'train_loader' not in dir():
                            train_loader, test_loader = get_data_loaders(args)
                        print('\n folder: '+args.save_folder+'; the '+str(k)+'-th seed')
                        loss_time_results, _, _=train_model(args, train_loader, test_loader, parameters0[k])
                        TestResults_txt.write(str(alg)+', outer regularizer coefficient='+str(outreg_coeff*20000)+'/20000'+': '+'Test accuracy='+str(loss_time_results[-1,7])+', Test loss='+str(loss_time_results[-1,1])+'\n')
            k+=1
            
            for outreg_coeff in outreg_coeffs:
                folder='save_tb_results/noiseRate'+str(noise_rate)+'_outregCoeff'+str(outreg_coeff)+'/'
                if not os.path.isdir(folder):
                    os.makedirs(folder)
        
                txt_name=folder+'AIDs.txt'
                algs=['AID-CG-unreg','AID-CG-unreg-momentum','AID-CG-prox','AID-CG-prox-momentum']
                legends=['BiO-AID','BiO-AIDm','Proximal BiO-AID','Proximal BiO-AIDm']
                colors=['red','black','lime','blue']
                linestyles=[':','-','--','-.']
                markers=['x','o','v','d']
                plot_folder=folder+'complete_AID_figures'
                plot_txt(algs,legends,colors,linestyles,markers,plot_folder,seeds,outreg_coeff,noise_rate,txt_name)
                
                txt_name=folder+'ITDs.txt'
                algs=['reverse-unreg','reverse-unreg-momentum','reverse-prox','reverse-prox-momentum']
                legends=['BiO-ITD','BiO-ITDm','Proximal BiO-ITD','Proximal BiO-ITDm']
                plot_folder=folder+'complete_ITD_figures'
                plot_txt(algs,legends,colors,linestyles,markers,plot_folder,seeds,outreg_coeff,noise_rate,txt_name)
                
                algs=['AID-CG-unreg','AID-CG-unreg-momentum','AID-CG-prox','AID-CG-prox-momentum']
                legends=['BiO-AID','BiO-AIDm','Proximal Bio-AID','Proximal Bio-AIDm']
                plot_folder=folder+'desired_figures'
                fig_names=[None,'TestLoss_unreg_AID',None,None,None]
                plot_txt(algs,legends,colors,linestyles,markers,plot_folder,seeds,outreg_coeff,noise_rate,None,fig_names)
                
                algs=['reverse-unreg','reverse-unreg-momentum','reverse-prox','reverse-prox-momentum']
                legends=['BiO-ITD','BiO-ITDm','Proximal Bio-ITD','Proximal Bio-ITDm']
                fig_names=[None,'TestLoss_unreg_ITD',None,None,None]
                plot_txt(algs,legends,colors,linestyles,markers,plot_folder,seeds,outreg_coeff,noise_rate,None,fig_names)
                
                algs=['AID-CG-prox','AID-CG-prox-momentum']
                legends=['Proximal Bio-AID','Proximal Bio-AIDm']
                fig_names=[None,None,None,'ValLoss_reg_AID',None]
                plot_txt(algs,legends,colors,linestyles,markers,plot_folder,seeds,outreg_coeff,noise_rate,None,fig_names)
                
                algs=['reverse-prox','reverse-prox-momentum']
                legends=['Proximal Bio-ITD','Proximal Bio-ITDm']
                fig_names=[None,None,None,'ValLoss_reg_ITD',None]
                plot_txt(algs,legends,colors,linestyles,markers,plot_folder,seeds,outreg_coeff,noise_rate,None,fig_names)
                
                algs=['AID-CG-unreg','AID-CG-unreg-momentum']
                legends=['Bio-AID','Bio-AIDm']
                fig_names=['ValLoss_unreg_AID',None,None,None,None]
                plot_txt(algs,legends,colors,linestyles,markers,plot_folder,seeds,outreg_coeff,noise_rate,None,fig_names)
    
                algs=['reverse-unreg','reverse-unreg-momentum']
                legends=['Bio-ITD','Bio-ITDm']
                fig_names=['ValLoss_unreg_ITD',None,None,None,None]
                plot_txt(algs,legends,colors,linestyles,markers,plot_folder,seeds,outreg_coeff,noise_rate,None,fig_names)
        del train_loader
