# Accelerated-Proximal-Algorithm-for-Regularized-Bi-level-Optimization



The code is adapted from that of the following paper:

http://proceedings.mlr.press/v139/ji21c.html 



Simply run mnist_exp.py. 

The file save_tb_results/test_results.txt implies Table 1 of the above paper where "reverse" means "ITD", and outer regularizer coefficient=gamma/20000=0 (unregularized), 0.001, 0.1 and 100 with 20000 validation samples. 

The subfigures of corruption rate p=0.1 (first row), 0.2 (second row), 0.4 (third row) of Figure 1 with in the above paper are respectively given by the following folders ("outregCoeff0.001" means outer regularizer coefficient=gamma/20000=0.001 with 20000 validation samples):

save_tb_results/noiseRate0.1_outregCoeff0.001/desired_figures
save_tb_results/noiseRate0.2_outregCoeff0.001/desired_figures
save_tb_results/noiseRate0.4_outregCoeff0.001/desired_figures

In each above folder, 
ValLoss_unreg_AID: Unregularized AID with outer regularizer coefficient=gamma/20000=0
ValLoss_unreg_ITD: Unregularized ITD with outer regularizer coefficient=gamma/20000=0
ValLoss_reg_AID: Regularized AID with outer regularizer coefficient=gamma/20000=0.001
ValLoss_reg_ITD: Regularized ITD with outer regularizer coefficient=gamma/20000=0.001
