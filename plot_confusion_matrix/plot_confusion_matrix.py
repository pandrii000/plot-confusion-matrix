import numpy as np
import matplotlib.pyplot as plt


def plot_confusion_matrix(
        confusion_matrix,
        row_labels,
        col_labels,
        row_title,
        col_title,
        title="Confusion matrix",
        normalize=False,
        cmap=plt.cm.Blues,
        tight_layout=False,
        ):

    if normalize:
        confusion_matrix = confusion_matrix.astype("float") / confusion_matrix.sum(axis=1)[:, None]

    plt.imshow(confusion_matrix, interpolation="nearest", cmap=cmap)

    plt.xlabel(col_title)
    plt.ylabel(row_title)
    plt.title(title)
    plt.colorbar()

    x_tick_marks = np.arange(len(col_labels))
    y_tick_marks = np.arange(len(row_labels))
    plt.xticks(x_tick_marks, col_labels, rotation=45)
    plt.yticks(y_tick_marks, row_labels)

    fmt = ".2f" if normalize else "d"
    thresh = 0.5 if normalize else confusion_matrix.max() / 2.0
    for x in np.arange(len(x_tick_marks)):
        for y in np.arange(len(y_tick_marks)):
            plt.text(y, x, format(confusion_matrix[x, y], fmt),
                horizontalalignment="center",
                color="white" if confusion_matrix[x, y] > thresh else "black",
            )

    if tight_layout:
        plt.tight_layout()
