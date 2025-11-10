# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
from collections import OrderedDict

data = {
    20: {
        "RANDOM":   [88.7, 83.9, 92.1, 93.2, 92.2, 92.5, 90.0, 88.7, 93.6, 93.4],
        "CONF":     [88.7, 85.0, 88.9, 90.4, 93.1, 93.1, 93.9, 93.4, 95.4, 94.5],
        "Margin":   [88.7, 89.0, 92.8, 90.7, 93.1, 94.0, 94.8, 94.6, 94.5, 96.4],
        "Entropy":  [88.7, 92.5, 87.3, 89.2, 88.0, 93.6, 90.3, 90.3, 93.4, 94.9],
        "CORESET":  [70.0, 75.2, 79.4, 82.6, 84.9, 86.2, 87.4, 88.3, 88.9, 89.4],
        "LL":       [84.9, 72.2, 89.8, 88.5, 86.7, 88.5, 87.9, 88.0, 94.9, 93.8],
        "BADGE":    [71.6, 91.7, 84.5, 87.1, 92.7, 93.1, 93.9, 92.6, 93.2, 90.2],
        "CCAL":     [91.2, 91.2, 81.2, 93.9, 93.3, 90.8, 93.9, 93.3, 92.2, 95.5],
        "MQNet":    [77.9, 85.7, 84.5, 88.1, 76.5, 90.0, 85.2, 90.8, 90.4, 83.6],
        "GOALDE":   [83.3, 77.3, 82.7, 87.5, 86.3, 87.4, 87.4, 92.2, 91.7, 93.1],
    },
    40: {
        "RANDOM":   [85.3, 88.4, 91.7, 93.6, 93.8, 93.6, 90.2, 93.2, 95.1, 93.4],
        "CONF":     [83.3, 89.3, 89.6, 90.8, 92.4, 94.6, 91.5, 92.2, 90.9, 89.6],
        "Margin":   [83.3, 91.3, 87.3, 91.1, 92.4, 90.6, 93.1, 92.3, 92.8, 94.7],
        "Entropy":  [87.4, 92.0, 90.5, 53.6, 88.7, 93.8, 82.4, 93.5, 94.6, 90.1],
        "CORESET":  [85.3, 93.1, 92.0, 94.1, 89.8, 95.1, 94.5, 93.6, 94.6, 93.9],
        "LL":       [66.7, 66.6, 82.5, 76.6, 76.4, 86.1, 88.6, 85.1, 92.2, 88.8],
        "BADGE":    [87.4, 88.3, 93.6, 87.6, 91.2, 92.3, 94.9, 93.6, 91.8, 93.0],
        "CCAL":     [85.0, 81.6, 90.5, 88.5, 90.8, 93.8, 91.0, 91.0, 90.2, 92.6],
        "MQNet":    [73.8, 81.0, 83.1, 91.3, 89.1, 75.0, 88.7, 90.3, 90.7, 91.7],
        "GOALDE":   [69.1, 76.0, 88.1, 82.6, 82.5, 87.8, 84.1, 88.2, 90.5, 91.3],
    },
    50: {
        "RANDOM":   [71.6, 90.4, 92.9, 88.3, 90.8, 87.6, 88.2, 94.6, 91.9, 93.0],
        "CONF":     [81.2, 87.0, 91.6, 79.8, 92.9, 91.5, 85.8, 91.5, 89.0, 89.8],
        "Margin":   [81.2, 92.8, 86.2, 90.9, 90.6, 91.4, 93.1, 90.4, 92.2, 88.3],
        "Entropy":  [68.7, 75.1, 89.4, 83.4, 21.5, 91.1, 86.5, 93.2, 90.7, 90.4],
        "CORESET":  [71.6, 90.9, 90.0, 75.1, 85.6, 93.3, 91.4, 91.8, 94.6, 94.2],
        "LL":       [66.4, 70.9, 73.9, 87.0, 83.0, 91.9, 80.1, 89.7, 88.5, 85.5],
        "BADGE":    [71.6, 91.7, 84.5, 87.1, 92.7, 93.1, 93.9, 92.6, 93.2, 90.2],
        "CCAL":     [71.9, 90.3, 78.1, 93.5, 87.5, 93.4, 94.0, 91.6, 90.5, 92.1],
        "MQNet":    [78.4, 86.1, 82.8, 80.8, 86.7, 89.3, 89.6, 87.3, 81.6, 80.5],
        "GOALDE":   [47.6, 76.1, 85.1, 85.0, 84.7, 87.2, 82.1, 84.8, 89.0, 86.9],
    },
}

rounds = None

methods_order = ["RANDOM","CONF","Margin","Entropy","CORESET","LL","BADGE","CCAL","MQNet","GOALDE"]

line_styles = {
    "RANDOM":  {"ls":"--", "marker":"*",  "lw":1.0, "ms":4},
    "CONF":    {"ls":"-",  "marker":"^",  "lw":1.0, "ms":4},
    "Margin":  {"ls":"-",  "marker":"s",  "lw":1.0, "ms":4},
    "Entropy": {"ls":"-",  "marker":"o",  "lw":1.0, "ms":4},
    "CORESET": {"ls":"-",  "marker":"P",  "lw":1.0, "ms":4},
    "LL":      {"ls":"--", "marker":"D",  "lw":1.0, "ms":4},
    "BADGE":   {"ls":"-.", "marker":"v",  "lw":1.0, "ms":4},
    "CCAL":    {"ls":"-",  "marker":"p",  "lw":1.0, "ms":4},
    "MQNet":   {"ls":"-",  "marker":"<",  "lw":1.2, "ms":4},
    "GOALDE":  {"ls":"-",  "marker":"X",  "lw":1.5, "ms":5},
}

def plot_accuracy_panels(data_dict, methods=methods_order, rounds=None,
                         noise_order=(20,40,50),
                         y_lims=None, figsize=(12, 2.6)):
    """
    y_lims: {noise_level: (ymin, ymax)} 형태로 노이즈별 y축 범위를 지정
    없으면 해당 noise의 실제 데이터로 자동 스케일.
    """
    # x축
    if rounds is None:
        # 첫 noise에서 최초로 존재하는 메서드의 길이로 라운드 수 결정
        first_noise = next(iter(data_dict))
        any_method = next(iter(data_dict[first_noise]))
        n = len(data_dict[first_noise][any_method])
        x = np.arange(1, n+1)
    else:
        x = np.array(rounds)

    fig, axes = plt.subplots(1, len(noise_order), figsize=figsize, dpi=160, constrained_layout=True)
    if len(noise_order) == 1:
        axes = [axes]

    for ax, nz in zip(axes, noise_order):
        # 실제 존재하는 메서드만 필터
        avail_methods = [m for m in methods if m in data_dict[nz]]

        # y축 범위
        if y_lims and nz in y_lims:
            y_low, y_high = y_lims[nz]
        else:
            all_vals = np.concatenate([np.array(data_dict[nz][m], dtype=float) for m in avail_methods])
            y_low  = np.floor(all_vals.min() - 1)
            y_high = np.ceil(all_vals.max() + 1)

        # plot
        for m in avail_methods:
            vals = np.array(data_dict[nz][m], dtype=float)
            style = line_styles.get(m, {"ls":"-","marker":None,"lw":1.0,"ms":4})
            ax.plot(x, vals,
                    label=m,
                    linestyle=style["ls"],
                    marker=style["marker"],
                    linewidth=style["lw"],
                    markersize=style["ms"])

        ax.set_title(f"{nz}% Noise")
        ax.set_xlabel("Round")
        ax.set_ylabel("Accuracy (%)")
        ax.grid(True, linestyle=":", linewidth=0.6)
        ax.set_xticks(x)
        ax.set_ylim(y_low, y_high)

    # 범례 (첫 축에서 수집 → 전체)
    handles, labels = axes[0].get_legend_handles_labels()
    uniq = list(OrderedDict((lab, h) for h, lab in zip(handles, labels)).items())
    fig.legend([h for _,h in uniq], [lab for lab,_ in uniq],
               loc="upper center", ncol=min(len(uniq), 10),
               frameon=False, bbox_to_anchor=(0.5, 1.15))

    plt.show()
    fig.savefig("accuracy_panels.png", bbox_inches="tight")

# 노이즈별 y축 범위 예시(원하면 수정)
y_lims = {20: (70, 100), 40: (70, 100), 50: (70, 100)}

plot_accuracy_panels(data, noise_order=(20,40,50), y_lims=y_lims)
