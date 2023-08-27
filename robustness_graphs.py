import csv
import numpy as np
from numpy.polynomial.polynomial import Polynomial
from matplotlib import pyplot as plt
from matplotlib.ticker import MultipleLocator
from matplotlib.lines import Line2D
from scipy import interpolate


def get_csv_data(path, min_param_val=None):
    with open(path, "r") as f:
        reader = csv.reader(f)
        reader.__next__()  # skip header
        params = []
        bit_accuracies = []
        id_pos = -1
        for i, (param, val) in enumerate(reader):
            if min_param_val is not None and param != "id" and float(param) < min_param_val:
                continue

            if param == "id":
                id_pos = i

            params.append(param)
            bit_accuracies.append(float(val))

        return params, bit_accuracies, id_pos


IDENTITY_COLOR = "#1f77b4"
COMBINED_COLOR = "#ff7f0e"
SPECIALIZED_COLOR = "#2ca02c"
RIVAGAN_COLOR = "#d62728"


ROBUSTNESS_DATA = [
    {
        "csv_files": ["robustness/crop/identity.csv", "robustness/crop/combined.csv"],
        "title": "Cropping (p)",
        "invert_x": True,
        "offsets": [0, -0.45],
        "x_interval": 0.2,
        "x_range": (0.1, 1),
        "id_param_val": 1,
        "x_label": "Crop ratio p",
        "colors": [IDENTITY_COLOR, COMBINED_COLOR],
        "labels": ["Identity", "Combined"],
    },
    {
        "csv_files": ["robustness/resize/identity.csv", "robustness/resize/specialized.csv", "robustness/resize/combined.csv"],
        "title": "Rescaling (s)",
        "invert_x": False,
        "offsets": [0, 0, 0],
        "x_interval": 0.5,
        "x_range": (0.5, 2),
        "id_param_val": 1,
        # "smoothing": 10,
        "x_label": "Rescale ratio s",
        "colors": [IDENTITY_COLOR, SPECIALIZED_COLOR, COMBINED_COLOR],
        "labels": ["Identity", "Specialized", "Combined"],
    },
    {
        "csv_files": ["robustness/translate/identity.csv", "robustness/translate/specialized.csv", "robustness/translate/combined.csv"],
        "title": "Translation (t)",
        "invert_x": False,
        "offsets": [0, 0, -0.35],
        "x_interval": 0.2,
        "x_range": (0, 0.75),
        "id_param_val": 0.0,
        "x_label": "Translation ratio t",
        "colors": [IDENTITY_COLOR, SPECIALIZED_COLOR, COMBINED_COLOR],
        "labels": ["Identity", "Specialized", "Combined"],
    },
    {
        "csv_files": ["robustness/rotate/identity.csv", "robustness/rotate/specialized.csv", "robustness/rotate/combined.csv",],
        "title": "Rotation (θ)",
        "invert_x": False,
        "offsets": [0, 0, 0],
        "x_interval": 15,
        "x_range": (0, 90),
        "id_param_val": 0,
        "format_int": True,
        "colors": [IDENTITY_COLOR, SPECIALIZED_COLOR, COMBINED_COLOR],
        "labels": ["Identity", "Specialized", "Combined"],
        # "smoothing": 27

    },
    {
        "csv_files": ["robustness/shear/identity.csv", "robustness/shear/specialized.csv", "robustness/shear/combined.csv"],
        "title": "Shearing (φ)",
        "invert_x": False,
        "offsets": [0, 0, 0],
        "x_interval": 15,
        "x_range": (0, 75),
        "id_param_val": 0,
        "format_int": True,
        "colors": [IDENTITY_COLOR, SPECIALIZED_COLOR, COMBINED_COLOR],
        "labels": ["Identity", "Specialized", "Combined"],
    },
    {
        "csv_files": ["robustness/blur/identity.csv", "robustness/blur/specialized.csv", "robustness/blur/combined.csv"],
        "title": "Gaussian blurring (σ)",
        "invert_x": False,
        "offsets": [0, 0, 0],
        "x_interval": 2,
        "x_range": (1, 9),
        "id_param_val": 0,
        "format_int": True,
        "colors": [IDENTITY_COLOR, SPECIALIZED_COLOR, COMBINED_COLOR],
        "labels": ["Identity", "Specialized", "Combined"],
    },
    {
        "csv_files": ["robustness/jpeg/identity.csv", "robustness/jpeg/specialized.csv", "robustness/jpeg/combined.csv"],
        "title": "JPEG (q)",
        "invert_x": True,
        "offsets": [0, 0, 0],
        "x_interval": 20,
        "x_range": (100, 10),
        "format_int": True,
        "colors": [IDENTITY_COLOR, SPECIALIZED_COLOR, COMBINED_COLOR],
        "labels": ["Identity", "Specialized", "Combined"],
    },
    {
        "data_files": ["robustness/mirror/identity.txt", "robustness/mirror/specialized.txt", "robustness/mirror/combined.txt", ],
        "title": "Mirroring",
        "categories": ["Identity", "Specialized", "Combined"],
        "colors": [IDENTITY_COLOR, SPECIALIZED_COLOR, COMBINED_COLOR],
        "labels": ["Identity", "Specialized", "Combined"],
    }
]

ROBUSTNESS_DATA_RIVAGAN = [
    {
        "csv_files": ["robustness/crop/identity.csv", "robustness/crop/combined.csv", "robustness/crop/rivagan.csv",],
        "title": "Cropping (p)",
        "invert_x": True,
        "offsets": [0, -.8, -0.75],
        "x_interval": 0.1,
        "x_range": (0.3, 1),
        "id_param_val": 1,
        "min_param_val": 0.3,
        "colors": [IDENTITY_COLOR, COMBINED_COLOR, RIVAGAN_COLOR],
        "labels": ["Identity", "Combined", "RivaGAN"],
    },
    {
        "csv_files": ["robustness/resize/identity.csv", "robustness/resize/specialized.csv", "robustness/resize/combined.csv", "robustness/resize/rivagan.csv", ],
        "title": "Rescaling (s)",
        "invert_x": False,
        "offsets": [0, 0, 0, 0],
        "x_interval": 0.5,
        "x_range": (0.5, 2),
        "id_param_val": 1,
        # "smoothing": 10,
        "colors": [IDENTITY_COLOR, SPECIALIZED_COLOR, COMBINED_COLOR, RIVAGAN_COLOR],
        "labels": ["Identity", "Specialized", "Combined", "RivaGAN"],
    },
    {
        "csv_files": ["robustness/translate/identity.csv", "robustness/translate/specialized.csv", "robustness/translate/combined.csv", "robustness/translate/rivagan.csv", ],
        "title": "Translation (t)",
        "invert_x": False,
        "offsets": [0, 0, -0.35, 0],
        "x_interval": 0.2,
        "x_range": (0, 0.75),
        "id_param_val": 0.0,
        "colors": [IDENTITY_COLOR, SPECIALIZED_COLOR, COMBINED_COLOR, RIVAGAN_COLOR],
        "labels": ["Identity", "Specialized", "Combined", "RivaGAN"],
    },
    {
        "csv_files": ["robustness/rotate/identity.csv", "robustness/rotate/specialized.csv", "robustness/rotate/combined.csv", "robustness/rotate/rivagan.csv", ],
        "title": "Rotation (θ)",
        "invert_x": False,
        "offsets": [0, 0, 0, 0],
        "x_interval": 15,
        "x_range": (0, 90),
        "id_param_val": 0,
        "format_int": True,
        # "smoothing": 27
        "colors": [IDENTITY_COLOR, SPECIALIZED_COLOR, COMBINED_COLOR, RIVAGAN_COLOR],
        "labels": ["Identity", "Specialized", "Combined", "RivaGAN"],
    },
    {
        "csv_files": ["robustness/shear/identity.csv", "robustness/shear/specialized.csv", "robustness/shear/combined.csv", "robustness/shear/rivagan.csv", ],
        "title": "Shearing (φ)",
        "invert_x": False,
        "offsets": [0, 0, 0, 0],
        "x_interval": 15,
        "x_range": (0, 75),
        "id_param_val": 0,
        "format_int": True,
        "colors": [IDENTITY_COLOR, SPECIALIZED_COLOR, COMBINED_COLOR, RIVAGAN_COLOR],
        "labels": ["Identity", "Specialized", "Combined", "RivaGAN"],
    },
    {
        "csv_files": ["robustness/blur/identity.csv", "robustness/blur/specialized.csv", "robustness/blur/combined.csv", "robustness/blur/rivagan.csv", ],
        "title": "Gaussian blurring (σ)",
        "invert_x": False,
        "offsets": [0, 0, 0, 0],
        "x_interval": 2,
        "x_range": (1, 9),
        "id_param_val": 0,
        "format_int": True,
        "colors": [IDENTITY_COLOR, SPECIALIZED_COLOR, COMBINED_COLOR, RIVAGAN_COLOR],
        "labels": ["Identity", "Specialized", "Combined", "RivaGAN"],
    },
    {
        "csv_files": ["robustness/jpeg/identity.csv", "robustness/jpeg/specialized.csv", "robustness/jpeg/combined.csv", "robustness/jpeg/rivagan.csv", ],
        "title": "JPEG (q)",
        "invert_x": True,
        "offsets": [0, 0, 0, 0],
        "x_interval": 20,
        "x_range": (100, 10),
        "format_int": True,
        "colors": [IDENTITY_COLOR, SPECIALIZED_COLOR, COMBINED_COLOR, RIVAGAN_COLOR],
        "labels": ["Identity", "Specialized", "Combined", "RivaGAN"],
    },
    {
        "data_files": ["robustness/mirror/identity.txt", "robustness/mirror/specialized.txt", "robustness/mirror/combined.txt", "robustness/mirror/rivagan.txt", ],
        "title": "Mirroring",
        "categories": ["Identity", "Specialized", "Combined", "RivaGAN"],
        "labels": ["Identity", "Specialized", "Combined", "RivaGAN"],
        "colors": [IDENTITY_COLOR, SPECIALIZED_COLOR, COMBINED_COLOR, RIVAGAN_COLOR],
    }
]


def robustness_graphs(robustness_data, legend, legend_colors, path):
    fig, axes = plt.subplots(3, 3, sharey='row', figsize=(12, 8))

    # Adjust the spacing as needed
    plt.subplots_adjust(hspace=0.33, wspace=0.07)

    all_colors = []
    all_labels = []

    for i, ax in enumerate(axes.flat):
        if i >= len(robustness_data):
            break

        attack_data = robustness_data[i]

        colors = attack_data["colors"]
        for color in colors:
            if color not in all_colors:
                all_colors.append(color)

        labels = attack_data["labels"]
        for label in labels:
            if label not in all_labels:
                all_labels.append(label)

        if "data_files" in attack_data:
            data_files = attack_data["data_files"]
            categories = attack_data["categories"]
            data = [float(np.loadtxt(data_file)) for data_file in data_files]
            ax.bar(categories, data, color=colors,
                   label=legend, width=0.33, align='center')
            ax.set_ylim(bottom=45, top=102)
            # y grid
            ax.yaxis.grid(True, linestyle='--', which='major',
                          color='grey', alpha=.33)
            ax.set_title(attack_data["title"])
            if i == len(robustness_data) - 1:
                # leg = ax.legend(legend,
                #                 loc='lower center', bbox_to_anchor=(4, 1))
                leg = ax.legend(legend,
                                loc='lower center', bbox_to_anchor=(4, 1))
                # legends.append(leg
            continue

        id_param_val = attack_data.get("id_param_val", None)

        csv_datas = [get_csv_data(csv_file, attack_data.get("min_param_val", None))
                     for csv_file in attack_data["csv_files"]]
        param_labels, _, id_pos = csv_datas[0]
        if id_pos == -1:
            params = [float(x) for x in param_labels]
        else:
            params = [
                id_param_val if i == id_pos and id_param_val is not None else float(
                    x)
                for i, x in enumerate(param_labels)
            ]

        acc_data = [csv_data[1] for csv_data in csv_datas]

        x_min = min(params)
        x_max = max(params)
        for j, acc in enumerate(acc_data):
            offset = attack_data["offsets"][j]
            acc_with_offset = [a + offset for a in acc]

            # Apply curve fitting if the "curve_fit_degree" parameter is present
            if "smoothing" in attack_data:
                smoothing = attack_data["smoothing"]
                x_new = np.linspace(x_min, x_max, smoothing)
                params_new = params[::-
                                    1] if attack_data["invert_x"] else params
                bspline = interpolate.make_interp_spline(
                    params_new, acc_with_offset)
                x_new = x_new[::-1] if attack_data["invert_x"] else x_new
                y_new = bspline(x_new)
                ax.plot(x_new, y_new,
                        color=colors[j], label=labels[j], marker='.')
            else:
                ax.plot(params, acc_with_offset,
                        color=colors[j], label=labels[j], marker='.')

        # Set the x-axis ticks using MultipleLocator
        x_interval = attack_data["x_interval"]
        x_major_locator = MultipleLocator(x_interval)
        ax.xaxis.set_major_locator(x_major_locator)

        x_ticks = np.arange(x_min, x_max + x_interval, x_interval)
        x_ticks = x_ticks[x_ticks <= x_max]
        if attack_data["invert_x"]:
            x_ticks = x_ticks[::-1]

        format_int = attack_data.get("format_int", False)
        x_tick_labels = [
            str(int(x)) if format_int else f"{x:.1f}" for x in x_ticks]

        if id_param_val is not None:
            if id_param_val in x_ticks:
                id_pos_new = np.where(x_ticks == id_param_val)[0][0]
                x_tick_labels[id_pos_new] = "id"
            else:
                # add id_param_val to x_ticks and x_tick_labels such that x_ticks remains sorted afterwards
                x_ticks = np.append(x_ticks, id_param_val)
                x_tick_labels.append("id")
                x_ticks, x_tick_labels = zip(
                    *sorted(zip(x_ticks, x_tick_labels)))

        # Always show the label for x_range[1]
        if x_max not in x_ticks:
            new_label = str(int(x_max) if format_int else f"{x_max:.1f}")
            if attack_data["invert_x"]:
                x_ticks = np.insert(x_ticks, 0, x_max)
                x_tick_labels = [new_label] + list(x_tick_labels)
            else:
                x_ticks = np.append(x_ticks, x_max)
                x_tick_labels = list(x_tick_labels) + [new_label]

        ax.set_xticks(x_ticks)
        ax.set_xticklabels(x_tick_labels)

        if attack_data["invert_x"]:
            ax.invert_xaxis()

        ax.set_ylim(bottom=45, top=102)

        ax.set_title(attack_data["title"])
        # y grid
        ax.yaxis.grid(True)
        # ax.yaxis.grid(True, linestyle='--', which='major', color='grey', alpha=.5)

        if i == len(robustness_data) - 1:
            leg = ax.legend(legend,
                            loc='center right', bbox_to_anchor=(4, 1))
            # legends.append(leg)

    # Add y-axis label to the leftmost subplots
    for i in range(0, len(axes)):
        axes[i, 0].set_ylabel("Bit accuracy (%)")

    # Remove unused subplots
    for i in range(len(robustness_data), len(axes.flat)):
        axes.flat[i].remove()

    # show legend outside of last plot
    # axes.flat[-1].legend(
    #     ["Identity", "Specialized", "Combined"],
    #     loc='upper center',)

    # Add shared legend
    # handles, labels = list(axes.flat)[1].get_legend_handles_labels()
    # for i, ax in enumerate(axes.flat):
    #     handles_temp, labels_temp = ax.get_legend_handles_labels()
    #     handles.extend(handles_temp)
    #     labels.extend(labels_temp)

    #     print(labels)
    #     print(handles)

    #     if i == len(robustness_data) - 1:
    #         break

    # plt.legend(legend,
    #            loc='lower right', bbox_to_anchor=(1.8, 0.3), prop={'size': 10.5})
    # fig.legend(handles, labels, loc='center right', bbox_to_anchor=(2.5, 0.3))
    print(legend)
    print(legend_colors)
    handles = [Line2D([0], [0], label=label, color=color, )
               for label, color in zip(legend, legend_colors)]

    fig.legend(handles,
               legend,
               loc='lower right',
               bbox_to_anchor=(0.85, 0.15),
               fancybox=True)

    plt.savefig(path)


if __name__ == "__main__":
    robustness_graphs(ROBUSTNESS_DATA,
                      ["Identity", "Specialized", "Combined"],
                      [IDENTITY_COLOR, SPECIALIZED_COLOR, COMBINED_COLOR],
                      "figures-output/robustness/robustness-own.pdf")
    robustness_graphs(ROBUSTNESS_DATA_RIVAGAN,
                      ["Identity", "Specialized", "Combined", "RivaGAN"],
                      [IDENTITY_COLOR, SPECIALIZED_COLOR,
                          COMBINED_COLOR, RIVAGAN_COLOR],
                      "figures-output/robustness/robustness-rivagan.pdf")
