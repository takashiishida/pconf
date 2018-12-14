"""A simple demo to visualize synthetic experiments for Pconf classification."""

from components import *
import matplotlib.pyplot as plt
import matplotlib

def plot_graph(i, mu2, input1, input2, input3, input4, r, clf_osvm_1, x_test, y_test):
    axarr[0,i].plot(input1, input2, 'r-', label='Pconf (proposed)', linewidth=3)
    axarr[0,i].plot(input1, input3, 'g:', label='Weighted (baseline)', linewidth=4)
    axarr[0,i].plot(input1, np.ones(len(input1))*1000, 'k-', label='One Class SVM (baseline)')
    axarr[0,i].plot(input1, input4, 'b--', label='Supervised', linewidth=2.5)
    if i==2:
        axarr[0,i].set_xlim([-12, 12])
        axarr[0,i].set_ylim([-9, 15])
        axarr[0,i].xaxis.set_ticks([-12, -4, 4, 12])
        axarr[0,i].yaxis.set_ticks([-9, -1, 7, 15])
        axarr[0,i].scatter(x_test[y_test == 1, 0], x_test[y_test == 1, 1], c='r', marker='x', s=10, lw=1)
        axarr[0,i].scatter(x_test[y_test == -1, 0], x_test[y_test == -1, 1], c='b', marker='.', s=15, lw=0)
        axarr[1,i].set_xlim([-12, 12])
        axarr[1,i].set_ylim([-9, 15])
        xx, yy = np.meshgrid(np.linspace(-12, 12, 500), np.linspace(-9, 15, 500))
    elif i==1:
        axarr[0,i].set_xlim([-9, 9])
        axarr[0,i].set_ylim([-8, 10])
        axarr[0,i].xaxis.set_ticks([-9, -3, 3, 9])
        axarr[0,i].yaxis.set_ticks([-8, -2, 4, 10])
        axarr[0,i].scatter(x_test[y_test == 1, 0], x_test[y_test == 1, 1], c='r', marker='x', s=10, lw=1)
        axarr[0,i].scatter(x_test[y_test == -1, 0], x_test[y_test == -1, 1], c='b', marker='.', s=15, lw=0)
        axarr[1,i].set_xlim([-9, 9])
        axarr[1,i].set_ylim([-8, 10])
        xx, yy = np.meshgrid(np.linspace(-9, 9, 500), np.linspace(-8, 10, 500))
    else:
        axarr[0,i].set_xlim([-9, 9])
        axarr[0,i].set_ylim([-8, 10])
        axarr[0,i].xaxis.set_ticks([-9, -3, 3, 9])
        axarr[0,i].yaxis.set_ticks([-8, -2, 4, 10])
        axarr[0,i].scatter(x_test[y_test == 1, 0], x_test[y_test == 1, 1], c='r', marker='x', s=10, lw=1)
        axarr[0,i].scatter(x_test[y_test == -1, 0], x_test[y_test == -1, 1], c='b', marker='.', s=15, lw=0)
        axarr[1,i].set_xlim([-9, 9])
        axarr[1,i].set_ylim([-8, 10])
        xx, yy = np.meshgrid(np.linspace(-9, 9, 500), np.linspace(-8, 10, 500))
    Z_1 = clf_osvm_1.decision_function(np.c_[xx.ravel(), yy.ravel()])
    Z_1 = Z_1.reshape(xx.shape)
    axarr[0,i].contour(xx, yy, Z_1, levels=[0], linewidths=2, colors='black')
    if i==2:
        axarr[0,i].xaxis.set_ticks([-12, -4, 4, 12])
        axarr[0,i].yaxis.set_ticks([-9, -1, 7, 15])
    if i==1:
        axarr[0,i].xaxis.set_ticks([-9, -3, 3, 9])
        axarr[0,i].yaxis.set_ticks([-8, -2, 4, 10])
    else:
        axarr[0,i].xaxis.set_ticks([-9, -3, 3, 9])
        axarr[0,i].yaxis.set_ticks([-9, -3, 3, 9])
    axarr[0,i].tick_params(axis='both', which='major', labelsize=18)
    axarr[0,i].scatter(x_test[y_test == 1, 0], x_test[y_test == 1, 1], c='r', marker='x', s=10, lw=1)
    axarr[0,i].scatter(x_test[y_test == -1, 0], x_test[y_test == -1, 1], c='b', marker='.', s=15, lw=0)
    # histogram
    axarr[1,i].hist(r, bins = 20)
    axarr[1,i].set_xlim([0,1])
    axarr[1,i].set_ylim([0,500])
    axarr[1,i].yaxis.set_ticks([0, 250, 500])
    axarr[1,i].xaxis.set_ticks([0, 0.25, 0.50, 0.75, 1.0])
    if i == 0:
        axarr[1,i].set_ylabel("Counts", fontsize=20)
    axarr[1,i].set_xlabel("Pconf", fontsize=20)
    axarr[1,i].tick_params(axis='both', which='major', labelsize=20)
    if i == 0:
        axarr[0,i].set_title(r'Setup A', fontsize=20)
    if i == 1:
        axarr[0,i].legend(loc="upper center", bbox_to_anchor=(1.13, 1.4), ncol=4, fontsize=17)
        axarr[0,i].set_title(r'Setup B', fontsize=20)
    if i == 2:
        axarr[0,i].set_title(r'Setup C', fontsize=20)
    if i == 3:
        axarr[0,i].set_title(r'Setup D', fontsize=20)

def find_boundary(params_pconf, params_naive, params_sup):
    p_w1 = params_pconf[0][0][0].data.numpy()
    p_w2 = params_pconf[0][0][1].data.numpy()
    p_b = params_pconf[1][0].data.numpy()
    n_w1 = params_naive[0][0][0].data.numpy()
    n_w2 = params_naive[0][0][1].data.numpy()
    n_b = params_naive[1][0].data.numpy()
    s_w1 = params_sup[0][0][0].data.numpy()
    s_w2 = params_sup[0][0][1].data.numpy()
    s_b = params_sup[1][0].data.numpy()
    input1 = np.linspace(-10, 10)
    input2 = -float(p_b) / p_w2 - float(p_w1) / p_w2 * input1
    input3 = -float(n_b) / n_w2 - float(n_w1) / n_w2 * input1
    input4 = -float(s_b) / s_w2 - float(s_w1) / s_w2 * input1
    return input1, input2, input3, input4

if __name__ == "__main__":
    np.random.seed(0); torch.manual_seed(0)
    num_epochs = 5000
    n_positive, n_negative = 500, 500  # number of training samples for P and N classes
    n_positive_test, n_negative_test = 1000, 1000  # number of test samples for P and N classes
    lr = 0.001  # learning rate for vanilla gradient descent
    # mu1 and cov1 are mean and covariance matrix of P Gaussian distribution
    mu1 = np.array([0, 0])
    cov1_candidates = [[[7, -6], [-6, 7]], [[5, 3], [3, 5]], [[7, -6], [-6, 7]], [[4, 0], [0, 4]]]
    # mu2 and cov2 are mean and covariance matrix of P Gaussian distribution
    mu2_candidates = np.array([[-2,5], [0,4], [0,8], [0, 4]])
    cov2_candidates = [[[2, 0], [0, 2]], [[5, -3], [-3, 5]], [[7, 6], [6, 7]], [[1, 0], [0, 1]]]
    matplotlib.rc('text', usetex=True)
    matplotlib.rcParams['text.latex.preamble']=[r"\usepackage{amsmath}"]
    _, axarr = plt.subplots(2, mu2_candidates.shape[0])
    plt.subplots_adjust(wspace=0.3)
    for i in range(len(mu2_candidates)):
        mu2 = mu2_candidates[i,:]
        cov1, cov2 = cov1_candidates[i], cov2_candidates[i]
        r, R, x_naive, x_train, x_train_p, x_test, y_naive, y_train, y_test = generateData(mu1=mu1, mu2=mu2, cov1=cov1, cov2=cov2, n_positive=n_positive, n_negative=n_negative, n_positive_test=n_positive_test, n_negative_test=n_negative_test)
        print('working on graph '+str(i+1))
        print('Start Pconf...')
        params_pconf, _ = pconfClassification(num_epochs=num_epochs, lr=lr, x_train_p=x_train_p, x_test=x_test, y_test=y_test, r=r)
        print('start Naive...')
        params_naive, _ = naiveClassification(num_epochs=num_epochs, lr=lr, x_naive=x_naive, y_naive=y_naive, y_test=y_test, x_test=x_test, R=R)
        print('start Supervised...')
        params_sup, _ = supervisedClassification(num_epochs=num_epochs, lr=lr, x_train=x_train, x_test=x_test, y_train=y_train, y_test=y_test)
        print('start OSVM...')
        clf_osvm_1, _ = osvmClassification(nu=0.05, x_train_p=x_train_p, x_test=x_test, y_train=y_train, y_test=y_test)
        input1, input2, input3, input4 = find_boundary(params_pconf, params_naive, params_sup)
        plot_graph(i=i, mu2=mu2, input1=input1, input2=input2, input3=input3, input4=input4, r=r, clf_osvm_1=clf_osvm_1, x_test=x_test, y_test=y_test)
    plt.show()