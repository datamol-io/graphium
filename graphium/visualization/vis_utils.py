import inspect
from copy import deepcopy

from matplotlib.offsetbox import AnchoredText


def annotate_metric(ax, metrics, x, y, fontsize=10, loc="upper left", **kwargs):
    # Compute each metric from `metrics` on the `x` and `y` data.
    # Then Annotate the plot with with the the results of each metric
    # Compute the metrics and generate strings for each metric

    stat_text = ""

    for metric_name, metric in metrics.items():
        kwargs_copy = deepcopy(kwargs)
        stat = metric(x, y, **kwargs_copy)
        stat_text += "\n" + metric_name + " = {:0.3f}".format(stat)

    stat_text = stat_text[1:]

    # Display the metrics on the plot
    _annotate(ax=ax.ax_joint, text=stat_text, loc=loc)


def _annotate(ax, text, loc="upper left", bbox_to_anchor=(1.2, 1)):
    text = text.strip()

    text_loc_outside = dict()
    if loc == "outside":
        text_loc_outside["bbox_to_anchor"] = bbox_to_anchor
        text_loc_outside["bbox_transform"] = ax.transAxes
        loc = "upper left"

    anchored_text = AnchoredText(text, loc=loc, **text_loc_outside)
    anchored_text.patch._alpha = 0.25
    ax.add_artist(anchored_text)
