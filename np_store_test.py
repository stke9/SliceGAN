import numpy as np
from matplotlib import pyplot as plt

array_np = np.random.uniform(1, 100, 100)
array_np_average_every_10_elements = np.mean(array_np.reshape(-1, 10), axis=1)
np.save('array.npy', array_np_average_every_10_elements)


wass_log = np.load('Trained_Generators/Hyperparameter_tuning_binary_beta1_0.2_beta2_0.9/_wass_log.npy')
wass_log_aver_30 = np.mean(wass_log.reshape(-1, 30), axis=1)

print(wass_log.shape)
print(wass_log_aver_30.shape)


def graph_plot(data_loss_real, data_loss_fake,labels_real, labels_fake,pth,name):
    """
    simple plotter for all the different graphs
    :param data: a list of data arrays
    :param labels: a list of plot labels
    :param pth: where to save plots
    :param name: name of the plot figure
    :return:
    """

    for datum, lbl in zip(data_loss_real, labels_real):
        x = np.arange(0, 9000, 30)
        plt.plot(x, datum, label=lbl)

    for datum, lbl in zip(data_loss_fake, labels_fake):
        x = np.arange(0, 9000, 30)
        plt.plot(x, datum, 'm--', label=lbl)

    print('plotting...')
    plt.legend()
    plt.savefig(pth + '_' + name)
    plt.close()


def calc_beta1_disc_loss():
    beta1_values = [0, .2, .5, .8, .9]
    losses_real = []
    losses_fake = []
    labels_real = []
    labels_fake = []

    for beta1 in beta1_values:
        loss_real_log = np.load('Trained_Generators/Hyperparameter_tuning_binary_beta1_' + str(beta1) + '_beta2_0.9/_disc_real_log.npy')
        loss_fake_log = np.load('Trained_Generators/Hyperparameter_tuning_binary_beta1_' + str(beta1) + '_beta2_0.9/_disc_fake_log.npy')

        loss_real_log_avrg = np.mean(loss_real_log.reshape(-1, 30), axis=1)
        loss_fake_log_avrg = np.mean(loss_fake_log.reshape(-1, 30), axis=1)

        losses_real.append(loss_real_log_avrg)
        losses_fake.append(loss_fake_log_avrg)
        labels_real.append('loss_real_beta1_' + str(beta1))
        labels_fake.append('loss_fake_beta1_' + str(beta1))

    graph_plot(losses_real, losses_fake,  labels_real, labels_fake, '', 'Loss_Graph')


def calc_beta_2_disc_loss():
    beta2_values = [.1, .3, .5, .9]

    losses_real = []
    losses_fake = []
    labels_real = []
    labels_fake = []

    for beta2 in beta2_values:
        loss_real_log = np.load('Trained_Generators/Hyperparameter_tuning_binary_beta1_0_beta2_' + str(beta2) + '/_disc_real_log.npy')
        loss_fake_log = np.load('Trained_Generators/Hyperparameter_tuning_binary_beta1_0_beta2_' + str(beta2) + '/_disc_fake_log.npy')

        loss_real_log_avrg = np.mean(loss_real_log.reshape(-1, 30), axis=1)
        loss_fake_log_avrg = np.mean(loss_fake_log.reshape(-1, 30), axis=1)

        losses_real.append(loss_real_log_avrg)
        losses_fake.append(loss_fake_log_avrg)
        labels_real.append('loss_real_beta2_' + str(beta2))
        labels_fake.append('loss_fake_beta2_' + str(beta2))

    graph_plot(losses_real, losses_fake,  labels_real, labels_fake, 'beta2', 'Loss_Graph_fixed')


def calc_wass_loss_beta_2():
    beta2_values = [.1, .3, .5, .9]

    wass_losses = []
    labels = []

    for beta2 in beta2_values:
        loss_real_log = np.load('Trained_Generators/Hyperparameter_tuning_binary_beta1_0_beta2_'
                                + str(beta2) + '/_wass_log.npy')
        loss_real_log_avrg = np.mean(loss_real_log.reshape(-1, 30), axis=1)

        wass_losses.append(loss_real_log_avrg)
        # losses_fake.append(loss_fake_log_avrg)
        labels.append('Was loss_beta2_' + str(beta2))
        # labels_fake.append('loss_fake_beta1_' + str(beta2))

    graph_plot(wass_losses, [],  labels, [], 'wass', 'Loss_Graph')


def calc_some_loss():
    beta1_values = [0, .2, .5, .8, .9]
    losses_real = []
    losses_fake = []
    labels_real = []
    labels_fake = []

    for beta1 in beta1_values:
        loss_real_log = np.load('Trained_Generators/Hyperparameter_tuning_binary_beta1_' + str(beta1) + '_beta2_0.9/_disc_real_log.npy')

        loss_real_log_avrg = np.mean(loss_real_log.reshape(-1, 30), axis=1)
        #loss_fake_log_avrg = np.mean(loss_fake_log.reshape(-1, 30), axis=1)

        losses_real.append(loss_real_log_avrg)
        #losses_fake.append(loss_fake_log_avrg)
        labels_real.append('loss_real_beta1_' + str(beta1))
        labels_fake.append('loss_fake_beta1_' + str(beta1))

    graph_plot(losses_real, losses_fake,  labels_real, labels_fake, '', 'Loss_Graph')