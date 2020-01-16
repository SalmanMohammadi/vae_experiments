import matplotlib.pyplot as plt
import numpy as np

def plot_results(classifier_sizes, results, num_repetitions):
    classification_labels = ["KNN", "Softmax"]
    regression_labels = ["KNNReg", "LinReg", "Sigmoid"]
    
    classification_latents = ["Shape"]
    regression_latents = ['Scale', 'Orientation', 'PosX', 'PosY']
    
    z_labels = ["all", "cons", "uncons"]
    results = tuple([np.array(x) for x in results])

    mu_regression, std_regression, mu_classification, std_classification, mu_metrics, std_metrics = results

    # Plot regressions
    mu_regression = np.swapaxes(mu_regression, 3, 2)
    std_regression = np.swapaxes(std_regression, 3, 2)

    fig, axes = plt.subplots(len(regression_labels), len(regression_latents),
                            sharex='col')
    print(mu_regression.shape)
    for i, regressor in enumerate(regression_labels):
        for j, latent in enumerate(regression_latents):
            for x, label in enumerate(z_labels):
                axes[i, j].errorbar(classifier_sizes, mu_regression[:, j, x, i],
                            yerr=std_regression[:, j, x, i], label=label,
                            marker="x", capsize=5)
            axes[i, j].set_title(regressor + " regression on " + latent)
            axes[i, j].set_xticks(classifier_sizes)
    plt.legend(loc='center left', bbox_to_anchor=(1, 1.7))
    fig.text(0.5, 0.04, 'Classified Dimensions', ha='center')
    fig.text(0.09, 0.5, 'Mean 4-Fold MSE', va='center', rotation='vertical')
    fig.tight_layout()
    # plt.show()

    # Plot classifications
    print(mu_classification.shape)
    print(std_classification.shape)
    mu_classification = np.swapaxes(mu_classification, 2, 1)
    std_classification = np.swapaxes(std_classification, 2, 1)
    fig, axes = plt.subplots(len(classification_labels), len(classification_latents),
                            sharex='col', sharey='row')
    for i, classifier in enumerate(classification_labels):
        for x, label in enumerate(z_labels):
            axes[i].errorbar(classifier_sizes, mu_classification[:, x, i],
                        yerr=std_classification[:, x, i], label=label,
                        marker="x", capsize=5)
            axes[i].set_title(classifier + " classification on Shape")
            axes[i].set_xticks(classifier_sizes)
    plt.legend(loc='center left', bbox_to_anchor=(1, 1))
    fig.text(0.5, 0.04, 'Classified Dimensions', ha='center')
    fig.text(0.04, 0.5, 'Mean 4-Fold Classification Accuracy', va='center', rotation='vertical')
    fig.tight_layout()
    
    # Plot metrics
    mu_metrics, std_metrics = np.array(mu_metrics).T, np.array(std_metrics).T
    print(std_metrics.shape)
    print(std_metrics)
    metrics_labels = ["loss", "r_loss", "kl_loss", "mse", "scaled_ms"]
    fig, axes = plt.subplots(len(metrics_labels), 1, sharex='col')
    for i, (label, metric, std) in enumerate(zip(metrics_labels, mu_metrics, std_metrics)):
        axes[i].errorbar(classifier_sizes, metric, yerr=std, marker="x",
                         capsize=5)
        axes[i].set_title("Mean " + label + " across " + str(num_repetitions) + " models")
        axes[i].set(xlabel="", ylabel=label)
        axes[i].set_xticks(classifier_sizes)
    fig.text(0.5, 0.04, 'Classified Dimensions', ha='center')
    plt.show()

results_all = ([], [], [], [], [], [])

#0

regression_results = [[[0.02791387,0.02792888,0.0279264],
  [0.0860605, 0.08605987,0.08605883],
  [0.01957853,0.01832972,0.01863993]],

 [[0.01583035,0.01592064,0.01597782],
  [0.06258602,0.06258987,0.06258968],
  [0.03155226,0.03321747,0.03155074]],

 [[0.01200906,0.01197721,0.01201217],
  [0.19021635,0.19021876,0.19021744],
  [0.04155466,0.04114730,0.03946338]],

 [[0.01295578,0.01295249,0.0129534,],
  [0.18734586,0.18735182,0.18734995],
  [0.04019876,0.03876274,0.03987034]]]
regression_std = [[[5.80818916e-04,5.25452429e-04,4.61738062e-04],
  [1.75938907e-03,1.76051911e-03,1.75885984e-03],
  [2.88471091e-03,7.51911255e-04,2.29996722e-03]],

 [[2.55989693e-02,2.57847887e-02,2.58851517e-02],
  [1.01416796e-01,1.01423465e-01,1.01424292e-01],
  [2.20542960e-02,2.20528878e-02,2.18312163e-02]],

 [[1.79306924e-04,1.48223029e-04,1.64514408e-04],
  [7.20394030e-03,7.20791891e-03,7.21259881e-03],
  [1.65850145e-03,3.39879608e-03,4.30156104e-03]],

 [[3.68308501e-05,4.64867953e-05,5.15113097e-05],
  [6.02295995e-03,6.02776324e-03,6.02566963e-03],
  [2.38662655e-03,2.79599428e-03,2.74380064e-03]]]
classification_results = [[0.99916446,0.9992051, 0.99916995],
                        [0.44638672,0.44594184,0.4441732]]
classification_std = [[1.0242978e-04,7.9362129e-05,1.4416428e-04],
                        [3.4239709e-03,5.9800623e-03,6.8120426e-03]]
metrics =[4.84939767e+01,3.34218591e+01,1.50721177e+01,1.09878922e-03,2.04466050e-01]
metrics_std = [0.18945832,0.1765132, 0.1016197, 0.00043347,0.03964964]

cur_results = [regression_results, regression_std, classification_results, classification_std, metrics, metrics_std]
for list_, elem in zip(results_all, cur_results):
    list_.append(elem)
#1
regression_results =[[[2.8199514e-02,1.7829773e-01,3.2185942e-02],
  [8.5873082e-02,1.6077037e-01,1.0169139e-01],
  [1.7707000e-02,3.0161411e-02,2.0755224e-02]],

 [[1.6602895e-06,7.1407944e-06,1.8721906e-06],
  [6.3830994e-06,6.4047285e-06,6.3849898e-06],
  [2.4838131e-02,2.3750346e-02,2.5507748e-02]],

 [[1.2025872e-02,3.1324086e-01,1.7128937e-02],
  [1.8123238e-01,2.8543231e-01,1.9476387e-01],
  [3.7478406e-02,8.4315836e-02,4.1871667e-02]],

 [[1.3006618e-02,3.0503586e-01,1.9029154e-02],
  [1.9098525e-01,2.7694362e-01,2.1621871e-01],
  [4.2239450e-02,7.9505689e-02,5.3622574e-02]]]
regression_std =[[[8.03885516e-04,4.19094833e-03,1.75507832e-03],
  [1.27101142e-03,3.81076639e-03,5.30256797e-03],
  [1.24316965e-03,2.91797891e-03,2.08632904e-03]],

 [[2.69311067e-06,1.15677412e-05,3.05112303e-06],
  [1.03430539e-05,1.03812290e-05,1.03465436e-05],
  [6.80216996e-04,2.63438880e-04,1.06291240e-03]],

 [[1.98611087e-04,1.49739766e-02,1.57868722e-03],
  [3.18541331e-03,1.26761915e-02,1.42900897e-02],
  [1.91724522e-03,8.66424572e-03,6.11442421e-03]],

 [[5.00887982e-05,3.25116962e-02,2.95844511e-03],
  [3.61141050e-03,2.51326505e-02,2.74274088e-02],
  [2.85520754e-03,1.28898332e-02,1.32229812e-02]]]
classification_results = [[0.9991428, 0.34752876,0.99474275],
                        [0.44099393,0.37626955,0.43892145]]
classification_std = [[9.7590855e-05,2.8545067e-03,4.7581739e-04],
                        [4.6408270e-03,1.1789496e-02,7.7905743e-03]]
metrics =[4.83734781e+01,3.33326672e+01,1.50408109e+01,3.80818994e-12,8.79688366e-06]
metrics_std = [2.98978277e-01,2.55775582e-01,1.16718882e-01,5.32337381e-12,8.54143727e-06]

cur_results = [regression_results, regression_std, classification_results, classification_std, metrics, metrics_std]
for list_, elem in zip(results_all, cur_results):
    list_.append(elem)

#2
regression_results =[[[2.8035179e-02,1.7315741e-01,3.5974253e-02],
  [8.7523177e-02,1.5718679e-01,1.0548612e-01],
  [1.9141642e-02,2.9436216e-02,2.1580640e-02]],

 [[1.6747606e-10,7.1840583e-10,2.2734033e-10],
  [6.5325406e-10,6.5416439e-10,6.5336542e-10],
  [2.4881773e-02,2.4026355e-02,2.4934385e-02]],

 [[1.2103039e-02,3.0314070e-01,2.5853867e-02],
  [1.8203373e-01,2.8216076e-01,1.9888854e-01],
  [3.8346905e-02,8.4854767e-02,4.5473687e-02]],

 [[1.2923618e-02,2.9818013e-01,2.6830077e-02],
  [1.9313231e-01,2.7905923e-01,2.2051759e-01],
  [4.1392457e-02,8.4094509e-02,5.4415196e-02]]]
regression_std =[[[2.3926415e-04,6.3747130e-03,3.8530957e-03],
  [1.2890528e-03,4.7391769e-03,7.7524502e-03],
  [1.5093178e-03,2.6883336e-03,1.7371677e-03]],

 [[2.7183583e-10,1.1652370e-09,3.6911960e-10],
  [1.0589669e-09,1.0604335e-09,1.0591004e-09],
  [3.7491074e-04,1.1646324e-03,7.6234114e-04]],

 [[2.5289450e-04,2.0589070e-02,8.5629011e-03],
  [8.1283953e-03,1.5507759e-02,1.8663855e-02],
  [2.4117001e-03,7.9433573e-03,1.0055854e-02]],

 [[2.1411436e-04,1.7162912e-02,6.7035360e-03],
  [5.8043045e-03,1.1544189e-02,1.2393500e-02],
  [3.0082795e-03,7.7751623e-03,6.9675744e-03]]]
classification_results = [[0.9990696, 0.3728163, 0.9842855],
                    [0.4412652, 0.39159074,0.4349718]]
classification_std = [[7.9817655e-05,2.0092663e-03,2.1481230e-03],
                    [6.0845837e-03,8.2525481e-03,6.6790073e-03]]
metrics =[4.84893502e+01,3.34358279e+01,1.50535222e+01,1.85863296e-13,1.41358619e-06]
metrics_std = [2.81530400e-01,3.40813257e-01,1.08610841e-01,3.65823763e-13,2.31070614e-06]

cur_results = [regression_results, regression_std, classification_results, classification_std, metrics, metrics_std]
for list_, elem in zip(results_all, cur_results):
    list_.append(elem)

#4
# Config(dataset={'test_index': 3}, model={'model': <class 'DSprites.RotDSpritesVAE'>, 'epochs': 20, 'lr': 0.001, 'batch_size': 512}, hparams={'z_size': 10, 'classifier_size': 4, 'include_loss': True})
regression_results =[[[0.02637523,0.12638536,0.06629657],
  [0.08780919,0.13886866,0.12284992],
  [0.01898649,0.02579949,0.02499859]],

 [[0.03325542,0.05672854,0.04371557],
  [0.05797627,0.05801874,0.05799309],
  [0.03037625,0.03155281,0.03010072]],

 [[0.01280521,0.1905226,0.09436417],
  [0.18865071,0.24534567,0.25348264],
  [0.04035532,0.06536425,0.07016528]],

 [[0.01203088,0.21767728,0.08449073],
  [0.18370163,0.264243,0.21647993],
  [0.03723441,0.07567922,0.0508311]]]
regression_std =[[[4.9968035e-04,6.1722584e-03,4.8433859e-03],
  [1.4147612e-03,7.7305222e-03,4.5584664e-03],
  [8.4907596e-04,2.8644870e-03,3.0117123e-03]],

 [[5.3830735e-02,9.1759555e-02,7.0978731e-02],
  [9.3976788e-02,9.4049126e-02,9.4002210e-02],
  [1.8003451e-02,1.9492645e-02,2.0860098e-02]],

 [[2.1133176e-04,1.5706355e-02,1.3734763e-02],
  [7.5750528e-03,1.2101293e-02,1.4942903e-02],
  [4.6131411e-03,5.5111004e-03,8.1710350e-03]],

 [[4.6162280e-05,1.7302699e-02,1.9032719e-02],
  [9.4274851e-03,2.0596163e-02,2.1741079e-02],
  [4.3976908e-03,1.1876298e-02,9.6302973e-03]]]
classification_results = [[0.9991129, 0.60411245,0.88898927],
                        [0.44403213,0.41547307,0.4310113]]
classification_std = [[0.00014774,0.01601211,0.00685746],
                        [0.00488675,0.01135493,0.01167822]]
metrics =[4.70985348e+01,3.21037918e+01,1.49947088e+01,3.43880601e-05,1.93204089e-02]
metrics_std = [2.83750687e-01,2.84179022e-01,5.37567734e-02,6.66649960e-05,3.13736832e-02]

cur_results = [regression_results, regression_std, classification_results, classification_std, metrics, metrics_std]
for list_, elem in zip(results_all, cur_results):
    list_.append(elem)

#6

regression_results =[[[0.02750647,0.07272442,0.11950256],
  [0.08851029,0.1315095, 0.13570337],
  [0.02090598,0.02583892,0.0259414,]],

 [[0.03500427,0.04982144,0.06923676],
  [0.07513765,0.07530373,0.07534794],
  [0.03995391,0.03825118,0.04009828]],

 [[0.01257438,0.09292303,0.19064492],
  [0.1852775, 0.24480872,0.24393812],
  [0.0372014, 0.06545644,0.06487377]],

 [[0.01287437,0.08874692,0.19255814],
  [0.1943972, 0.24983235,0.24506815],
  [0.04256432,0.0676502, 0.06649999]]]
regression_std =[[[5.23566676e-04,5.45962108e-03,7.72380969e-03],
  [1.67635223e-03,7.52218673e-03,7.13487342e-03],
  [2.51375907e-03,3.68195539e-03,2.24598078e-03]],

 [[5.66234887e-02,8.15317929e-02,1.12063922e-01],
  [1.21813975e-01,1.22038715e-01,1.22187160e-01],
  [3.45247835e-02,3.32002081e-02,3.41635682e-02]],

 [[6.53729003e-05,1.46877682e-02,2.30168384e-02],
  [1.04852412e-02,3.40572633e-02,2.90890932e-02],
  [4.52555902e-03,1.73134711e-02,1.51878707e-02]],

 [[1.48970459e-04,2.27625370e-02,2.17200443e-02],
  [6.80207973e-03,1.61602888e-02,2.14880724e-02],
  [4.02053259e-03,8.87398794e-03,1.33211073e-02]]]
classification_results = [[0.99912655,0.89389104,0.6127902],[0.44722223,0.43155378,0.42010632]]
classification_std = [[0.00015359,0.00966019,0.01889587],
                    [0.00565771,0.01030179,0.01467621]]
metrics =[4.70232515e+01,3.19477098e+01,1.50755015e+01,4.05122533e-05,2.10137536e-02]
metrics_std = [2.67310863e-01,3.37684075e-01,9.72855986e-02,7.84688554e-05,3.40261931e-02]

cur_results = [regression_results, regression_std, classification_results, classification_std, metrics, metrics_std]
for list_, elem in zip(results_all, cur_results):
    list_.append(elem)

#8
regression_results =[[[2.7962968e-02,3.8664263e-02,1.6829236e-01],
  [8.6544842e-02,1.1298746e-01,1.5434341e-01],
  [1.8791828e-02,2.4987102e-02,2.9957309e-02]],

 [[3.6272127e-06,3.8462308e-06,8.5094371e-06],
  [7.6713750e-06,7.6732704e-06,7.7072782e-06],
  [2.4631042e-02,2.5626916e-02,2.3577703e-02]],

 [[1.2637806e-02,3.4469809e-02,2.8632456e-01],
  [1.9125709e-01,2.2342217e-01,2.7065983e-01],
  [4.1040890e-02,5.5254329e-02,7.9938129e-02]],

 [[1.2713778e-02,2.8760672e-02,2.8803101e-01],
  [1.8615732e-01,2.1199532e-01,2.7278107e-01],
  [3.9561488e-02,4.9892012e-02,8.0474511e-02]]]
regression_std =[[[5.10678685e-04,3.76165565e-03,7.10371695e-03],
  [8.30823206e-04,8.30359478e-03,6.33141771e-03],
  [9.63182421e-04,3.13703483e-03,2.31284159e-03]],

 [[5.86795750e-06,6.20525998e-06,1.38338546e-05],
  [1.24323597e-05,1.24341805e-05,1.24940598e-05],
  [9.13436117e-04,1.78761652e-03,6.00856147e-04]],

 [[1.41430777e-04,7.07049528e-03,2.10182425e-02],
  [6.96487026e-03,2.02641152e-02,1.65901985e-02],
  [2.95486557e-03,9.54906736e-03,7.97260925e-03]],

 [[1.08831577e-04,8.58831499e-03,3.34089175e-02],
  [7.34541658e-03,2.80353148e-02,3.03851720e-02],
  [2.20026891e-03,1.36200814e-02,1.21173421e-02]]]
classification_results = [[0.9991428, 0.9846843, 0.3724745],
                    [0.44266492,0.44026694,0.38538408]]
classification_std = [[0.00015137,0.00142778,0.00583057],
                    [0.00299633,0.00427261,0.00922734]]
metrics =[4.70046259e+01,3.20456046e+01,1.49590212e+01,1.30831367e-12,4.00024068e-06]
metrics_std = [2.00433964e-01,2.08744886e-01,4.79238120e-02,2.49380070e-12,5.97061368e-06]

cur_results = [regression_results, regression_std, classification_results, classification_std, metrics, metrics_std]
for list_, elem in zip(results_all, cur_results):
    list_.append(elem)

#9
regression_results =[[[2.7753526e-02,3.1615458e-02,1.7877163e-01],
  [8.7025836e-02,1.0090097e-01,1.6044812e-01],
  [1.9553756e-02,2.2503575e-02,3.0367449e-02]],

 [[3.7162581e-10,3.8400083e-10,8.7710755e-10],
  [7.8343543e-10,7.8377377e-10,7.8702972e-10],
  [2.5231222e-02,2.4335463e-02,2.3563677e-02]],

 [[1.2533168e-02,1.7531935e-02,3.1118286e-01],
  [1.8198855e-01,2.0261402e-01,2.8332788e-01],
  [3.6365658e-02,4.4345133e-02,8.4819339e-02]],

 [[1.2764183e-02,1.6536262e-02,3.1042036e-01],
  [1.9376889e-01,2.1146163e-01,2.8162250e-01],
  [4.0894315e-02,4.9533106e-02,8.1370935e-02]]]
regression_std =[[[5.77632512e-04,2.09250441e-03,5.05408458e-03],
  [1.62446767e-03, 7.72509119e-03, 4.79469774e-03],
  [1.88534393e-03,2.08899984e-03,3.05343955e-03]],

 [[6.01721284e-10,6.21908802e-10,1.42227985e-09],
  [1.26960287e-09,1.27019129e-09,1.27573196e-09],
  [6.57748547e-04,8.10921774e-04,6.46597531e-04]],

 [[1.02701524e-04,2.41611735e-03,1.59553457e-02],
  [4.96620033e-03,1.76695343e-02,1.60713010e-02],
  [2.90476531e-03,7.11155543e-03,1.12082912e-02]],

 [[9.87015228e-05,2.66540051e-03,1.72171313e-02],
  [5.78325149e-03,1.46541586e-02,1.59636680e-02],
  [1.95680768e-03,6.24688715e-03,9.12445970e-03]]]
classification_results = [[0.999273,  0.99517953,0.34513074],
 [0.44142795,0.44283858,0.37659502]]
classification_std= [[0.00015992,0.00046392,0.00269405],
 [0.00283525,0.0049674, 0.00339275]]
metrics =[4.72968278e+01,3.22470841e+01,1.50497437e+01,2.66110999e-14,7.25090315e-07]
metrics_std = [2.82845890e-01,2.63315942e-01,6.24556661e-02,4.24216059e-14,7.24436436e-07]

##

cur_results = [regression_results, regression_std, classification_results, classification_std, metrics, metrics_std]
for list_, elem in zip(results_all, cur_results):
    list_.append(elem)

results_all = [np.array(x) for x in results_all]
# [print(x.shape) for x in results_all]
classifier_sizes = [0, 1, 2, 4, 6, 8, 9]
plot_results(classifier_sizes, results_all, 5)