from typing import Dict, List

import matplotlib.pyplot as plt


def graph_loss_vs_epoch(title, losses: Dict[str, List[float]], ylabel="Loss", xlabel="Epoch", legloc="upper right"):
    plt.rcParams.update(plt.rcParamsDefault)
    plt.style.use("./res/style.mplstyle")

    img_width, img_height, DPI = 640, 400, 112
    plt.rcParams["figure.figsize"] = (img_width / DPI, img_height / DPI)
    plt.rcParams["figure.dpi"] = DPI
    plt.gca().set_title(title, pad=16)
    plt.gca().set_ylabel(ylabel, labelpad=8)
    plt.gca().set_xlabel(xlabel, labelpad=8)

    for label, loss in losses.items():
        epochs = len(loss)
        plt.plot(range(1, epochs + 1), loss, ".-", label=label)

    plt.gca().legend(loc=legloc)
    plt.gcf().tight_layout()

    plt.show()

def main():
    "Hardcoded values gathered from experiments with low variance"

    # Default nnet
    nnclass_train_avgloss = [0.089907, 0.089762, 0.089290, 0.086276, 0.074874, 0.062678, 0.052845, 0.045097, 0.037835, 0.031539, 0.026924, 0.023551, 0.021109, 0.019259, 0.017904, 0.016827]
    nnclass_train_hitrate = [0.10, 0.11, 0.14, 0.27, 0.43, 0.55, 0.64, 0.74, 0.80, 0.84, 0.87, 0.88, 0.89, 0.90, 0.90, 0.91]
    nnclass_test_avgloss = [0.089905, 0.089753, 0.089273, 0.086196, 0.074709, 0.062566, 0.052788, 0.044866, 0.037375, 0.030976, 0.026373, 0.023057, 0.020691, 0.018910, 0.017580, 0.016605]
    nnclass_test_hitrate = [0.10, 0.12, 0.14, 0.27, 0.43, 0.54, 0.64, 0.75, 0.81, 0.85, 0.87, 0.89, 0.89, 0.90, 0.90, 0.91]

    # Nnet with custom initialization
    niclass_train_avgloss = [0.088102, 0.076845, 0.059832, 0.047017, 0.038983, 0.032131, 0.027179, 0.023941, 0.021549, 0.019738, 0.018296, 0.017088, 0.016187, 0.015295, 0.014640, 0.014090]
    niclass_train_hitrate = [0.28, 0.40, 0.62, 0.70, 0.78, 0.82, 0.85, 0.87, 0.89, 0.89, 0.90, 0.90, 0.91, 0.91, 0.92, 0.92]
    niclass_test_avgloss = [0.088089, 0.076701, 0.059641, 0.046767, 0.038546, 0.031473, 0.026436, 0.023248, 0.020919, 0.019180, 0.017830, 0.016647, 0.015892, 0.015013, 0.014395, 0.013868]
    niclass_test_hitrate = [0.29, 0.41, 0.62, 0.71, 0.78, 0.83, 0.86, 0.88, 0.89, 0.89, 0.90, 0.91, 0.91, 0.91, 0.92, 0.92]

    # PyTorch net
    trclass_train_avgloss = [0.089601, 0.087517, 0.078065, 0.071867, 0.061582, 0.050261, 0.042568, 0.035939, 0.030280, 0.026084, 0.023157, 0.021038, 0.019487, 0.018340, 0.017419, 0.016705]
    trclass_train_hitrate = [0.18, 0.21, 0.28, 0.40, 0.55, 0.66, 0.75, 0.80, 0.84, 0.86, 0.88, 0.89, 0.89, 0.90, 0.90, 0.90]
    trclass_test_avgloss = [0.089582, 0.087438, 0.077742, 0.071516, 0.061256, 0.049896, 0.042071, 0.035399, 0.029686, 0.025589, 0.022704, 0.020699, 0.019200, 0.018128, 0.017283, 0.016586]
    trclass_test_hitrate = [0.18, 0.21, 0.28, 0.39, 0.55, 0.67, 0.76, 0.81, 0.85, 0.87, 0.88, 0.88, 0.89, 0.89, 0.90, 0.90]

    # Autoencoders
    nnauto_train_avgloss = [0.069238, 0.068180, 0.067844, 0.067637, 0.067393, 0.066926, 0.066147, 0.065078, 0.063716, 0.062233, 0.060811, 0.059478, 0.058218, 0.057032, 0.055918, 0.054869]
    nnauto_test_avgloss = [0.069434, 0.068377, 0.068040, 0.067833, 0.067589, 0.067126, 0.066329, 0.065221, 0.063839, 0.062292, 0.060839, 0.059452, 0.058180, 0.056956, 0.055807, 0.054738]

    niauto_train_avgloss = [0.069017, 0.067231, 0.065836, 0.064200, 0.062412, 0.060586, 0.058808, 0.057118, 0.055528, 0.054064, 0.052730, 0.051511, 0.050393, 0.049361, 0.048406, 0.047515]
    niauto_test_avgloss = [0.069205, 0.067411, 0.065998, 0.064316, 0.062483, 0.060596, 0.058760, 0.057026, 0.055396, 0.053901, 0.052546, 0.051312, 0.050173, 0.049128, 0.048161, 0.047265]

    trauto_train_avgloss = [0.069225, 0.068141, 0.067734, 0.067380, 0.066883, 0.066093, 0.065004, 0.063758, 0.062401, 0.061006, 0.059633, 0.058289, 0.056984, 0.055734, 0.054537, 0.053393]
    trauto_test_avgloss = [0.069410, 0.068329, 0.067918, 0.067572, 0.067064, 0.066261, 0.065132, 0.063862, 0.062454, 0.061028, 0.059607, 0.058220, 0.056889, 0.055607, 0.054400, 0.053235]

    # English graphs

    class_losses_en = { "Nnet Loss": nnclass_test_avgloss, "Nnet* Loss": niclass_test_avgloss, "PyTorch Loss": trclass_test_avgloss, }
    class_hitrates_en = { "Nnet Hitrate": [n * 100 for n in nnclass_test_hitrate], "Nnet* Hitrate": [n * 100 for n in niclass_test_hitrate], "PyTorch Hitrate": [n * 100 for n in trclass_test_hitrate], }
    auto_losses_en = { "Nnet Loss": nnauto_test_avgloss, "Nnet* Loss": niauto_test_avgloss, "PyTorch Loss": trauto_test_avgloss, }

    graph_loss_vs_epoch("MNIST Classifier - Error per Epoch", class_losses_en, ylabel="Error", xlabel="Epoch")
    graph_loss_vs_epoch("MNIST Classifier - Accuracy per Epoch", class_hitrates_en, ylabel="Hits (%)", xlabel="Epoch", legloc="lower right")
    graph_loss_vs_epoch("MNIST Autoencoder - Error per Epoch", auto_losses_en, ylabel="Error", xlabel="Epoch")

    # Spanish graphs

    class_losses_es = { "Error Nnet": nnclass_test_avgloss, "Error Nnet*": niclass_test_avgloss, "Error PyTorch": trclass_test_avgloss, }
    class_hitrates_es = { "Aciertos Nnet": [n * 100 for n in nnclass_test_hitrate], "Aciertos Nnet*": [n * 100 for n in niclass_test_hitrate], "Aciertos PyTorch": [n * 100 for n in trclass_test_hitrate], }
    auto_losses_es = { "Error Nnet": nnauto_test_avgloss, "Error Nnet*": niauto_test_avgloss, "Error PyTorch": trauto_test_avgloss, }

    graph_loss_vs_epoch("Clasificador MNIST - Error por Época", class_losses_es, ylabel="Error", xlabel="Época")
    graph_loss_vs_epoch("Clasificador MNIST - Precisión por Época", class_hitrates_es, ylabel="Aciertos (%)", xlabel="Época", legloc="lower right")
    graph_loss_vs_epoch("Autoencoder MNIST - Error por Época", auto_losses_es, ylabel="Error", xlabel="Época")



if __name__ == "__main__":
    main()
