# Accelerated-Proximal-Algorithm-for-Regularized-Bi-level-Optimization
This submission is for the following paper:

Chen, Z., Kailkhura, B., & Zhou, Y. (2022). A Fast and Convergent Proximal Algorithm for Regularized Nonconvex and Nonsmooth Bi-level Optimization. arXiv preprint arXiv:2203.16615.

https://arxiv.org/abs/2203.16615


Simply run mnist_exp.py. 

You will obtain Table 1 in the above figure given by test_results.txt, and Figure 1 in the above paper given by the following folders ("outregCoeff0.001" and "outregCoeff0.0001" correspond to gamma=0.001*5000=5 and gamma=0.0001*5000=0.5 respectively with 5000 validation samples.):

save_tb_results/noiseRate0.1_outregCoeff0.001/desired_figures

save_tb_results/noiseRate0.1_outregCoeff0.0001/desired_figures

save_tb_results/noiseRate0.2_outregCoeff0.001/desired_figures

save_tb_results/noiseRate0.2_outregCoeff0.0001/desired_figures

save_tb_results/noiseRate0.4_outregCoeff0.001/desired_figures

save_tb_results/noiseRate0.4_outregCoeff0.0001/desired_figures
