# -*- coding: utf-8 -*-
import os
class ARGS:
#    def __init__(self,num_classes=10,batch_size=1000,test_size=1000,inreg_coeff=0.001,\
#                 outreg_bounds=None,outreg_coeff=1e-2,outreg_thr=20,epochs=50,iterations=5,\
#                 outer_lr=0.1,inner_lr=0.1,momentum_coeff=0.0,\
#                 data_path='data/',training_size=20000,validation_size=5000,\
#                 noise_rate=0.1,hessian_q=10,save_folder='',model_name='',seed=1,alg='stocBiO',
#                 save_paras=True):    

    def __init__(self,num_classes=10,batch_size=1000,test_size=1000,inreg_coeff=0.001,\
                 outreg_bounds=None,outreg_coeff=0.0001,outreg_thr=20,epochs=50,iterations=5,\
                 outer_lr=0.5,inner_lr=0.1,momentum_coeff=0.0,\
                 data_path='data/',training_size=20000,validation_size=5000,\
                 noise_rate=0.1,hessian_q=10,save_folder='',model_name='',seed=1,alg='stocBiO',
                 save_paras=True):
        self.num_classes=num_classes
        self.batch_size=batch_size
        self.test_size=test_size
        self.inreg_coeff=inreg_coeff
        if outreg_bounds is not None:  #Hard regularizer
            assert outreg_bounds[0]<0, "outreg_bounds[0] should <0"
            assert outreg_bounds[1]>0, "outreg_bounds[1] should >0"
        self.outreg_bounds=outreg_bounds  #Hard regularizer
        self.outreg_coeff=outreg_coeff    #Soft regularizer
        self.outreg_thr=outreg_thr        #Soft regularizer
        self.epochs=epochs
        self.iterations=iterations
        self.outer_lr=outer_lr
        self.inner_lr=inner_lr
        self.momentum_coeff=momentum_coeff
        self.data_path=data_path
        self.training_size=training_size
        self.validation_size=validation_size
        self.noise_rate=noise_rate
        self.hessian_q=hessian_q
        self.save_folder=save_folder
        self.model_name=model_name
        self.seed=seed
        self.alg=alg
        self.save_paras=save_paras
#        assert self.alg in ['stocBiO', 'HOAG', 'TTSA', 'BSA', 'reverse', 'AID-CG', 'AID-FP'], \
#            "alg should be in ['stocBiO', 'HOAG', 'TTSA', 'BSA', 'reverse', 'AID-CG', 'AID-FP']"
        
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--num_classes', default=10, type=int)
    # parser.add_argument('--batch_size', type=int, default=1000)
    # parser.add_argument('--test_size', type=int, default=1000)
    # parser.add_argument('--epochs', type=int, default=50, help='K')
    # parser.add_argument('--iterations', type=int, default=200, help='T')
    # parser.add_argument('--outer_lr', type=float, default=0.1, help='beta')
    # parser.add_argument('--inner_lr', type=float, default=0.1, help='alpha')
    # parser.add_argument('--eta', type=float, default=0.5, help='used in Hessian')
    # parser.add_argument('--data_path', default='data/', help='The temporary data storage path')
    # parser.add_argument('--training_size', type=int, default=20000)
    # parser.add_argument('--validation_size', type=int, default=5000)
    # parser.add_argument('--noise_rate', type=float, default=0.1)
    # parser.add_argument('--hessian_q', type=int, default=3)
    # parser.add_argument('--save_folder', type=str, default='', help='path to save result')
    # parser.add_argument('--model_name', type=str, default='', help='Experiment name')
    # parser.add_argument('--seed', type=int, default=1)
    # parser.add_argument('--alg', type=str, default='stocBiO', choices=['stocBiO', 'HOAG', 'TTSA', 'BSA', 
    #                                                     'reverse', 'AID-CG', 'AID-FP'])
    # self = parser.parse_self()

        if self.alg == 'stocBiO':
            self.batch_size = self.batch_size
        elif self.alg == 'BSA':
            self.batch_size = 1
        elif self.alg == 'TTSA':
            self.batch_size = 1
            self.iterations = 1
        else:
            self.batch_size = self.training_size
    
        if not self.save_folder:
            self.save_folder = './save_tb_results'
        if outreg_bounds is not None:  #Hard regularizer
            self.model_name = '{}_{}_bs_{}_olr_{}_ilr_{}_eta_{}_momentum_{}_noiser_{}_q_{}_ite_{}_lambda[{},{}]'.format(self.alg, 
                               self.training_size, self.batch_size, self.outer_lr, self.inner_lr, self.eta, self.momentum_coeff, 
                               self.noise_rate, self.hessian_q, self.iterations, self.outreg_bounds[0], outreg_bounds[1])
        else:
            self.model_name = '{}_{}_bs_{}_olr_{}_ilr_{}_momentum_{}_noiser_{}_q_{}_ite_{}_reg{}min(|x|,{})'.format(self.alg, 
                               self.training_size, self.batch_size, self.outer_lr, self.inner_lr, self.momentum_coeff, 
                               self.noise_rate, self.hessian_q, self.iterations, self.outreg_coeff, outreg_thr)
        self.save_folder = os.path.join(self.save_folder, self.model_name)
        if not os.path.isdir(self.save_folder):
            os.makedirs(self.save_folder)

