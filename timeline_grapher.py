import json
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.lines import Line2D
from matplotlib.legend_handler import HandlerBase, HandlerTuple

# Hardcoded JSON data
data_json = r'''
[
  {
    "model_name": "text-davinci-001",
    "release_date": "01/27/2022",
    "accuracy": 51.40,
    "accuracy_finetuned": null,
    "percent_error": 0,
    "has_seen_data": false,
    "knowledge_cutoff_date": "06/01/2021",
    "label_offset_x": 25,
    "label_offset_y": 70
  },
  {
    "model_name": "text-davinci-002",
    "release_date": "03/15/2022",
    "accuracy": 53.15,
    "accuracy_finetuned": null,
    "percent_error": 0,
    "has_seen_data": false,
    "knowledge_cutoff_date": "06/01/2021",
    "label_offset_x": 63,
    "label_offset_y": 17
  },
  {
    "model_name": "text-davinci-003",
    "release_date": "11/28/2022",
    "accuracy": 56.26,
    "accuracy_finetuned": null,
    "percent_error": 0,
    "has_seen_data": false,
    "knowledge_cutoff_date": "06/01/2021",
    "label_offset_x": 15,
    "label_offset_y": 30
  },
  {
    "model_name": "gpt-4-0613",
    "release_date": "06/13/2023",
    "accuracy": 64.05,
    "accuracy_finetuned": null,
    "percent_error": 0.92,
    "has_seen_data": true,
    "knowledge_cutoff_date": "03/01/2023",
    "label_offset_x": 40,
    "label_offset_y": 35
  },
  {
    "model_name": "gpt-4-1106-preview",
    "release_date": "11/06/2023",
    "accuracy": 62.03,
    "accuracy_finetuned": null,
    "percent_error": 0,
    "has_seen_data": true,
    "knowledge_cutoff_date": "04/30/2023",
    "label_offset_x": 20,
    "label_offset_y": -25
  },
  {
    "model_name": "gpt-3.5-turbo-1106",
    "release_date": "11/06/2023",
    "accuracy": 50.92,
    "accuracy_finetuned": null,
    "percent_error": 0.96,
    "has_seen_data": false,
    "knowledge_cutoff_date": "09/01/2021",
    "label_offset_x": -3,
    "label_offset_y": 65
  },
  {
    "model_name": "gpt-3.5-turbo-0125",
    "release_date": "01/25/2024",
    "accuracy": 50.89,
    "accuracy_finetuned": null,
    "percent_error": 0.96,
    "has_seen_data": false,
    "knowledge_cutoff_date": "09/01/2021",
    "label_offset_x": 37,
    "label_offset_y": 46
  },
  {
    "model_name": "gpt-4o-2024-05-13",
    "release_date": "05/13/2024",
    "accuracy": 63.29,
    "accuracy_finetuned": null,
    "percent_error": 0.92,
    "has_seen_data": true,
    "knowledge_cutoff_date": "10/01/2023",
    "label_offset_x": -37,
    "label_offset_y": 23
  },
  {
    "model_name": "qwen2.5-3b",
    "release_date": "09/19/2024",
    "accuracy": 53.5,
    "accuracy_finetuned": 76.2,
    "percent_error": 0,
    "has_seen_data": false,
    "knowledge_cutoff_date": "12/01/2023",
    "label_offset_x": 25,
    "label_offset_y": -20
  },
  {
    "model_name": "gpt-4o-2024-11-20",
    "release_date": "11/20/2024",
    "accuracy": 62.40,
    "accuracy_finetuned": null,
    "percent_error": 0.92,
    "has_seen_data": true,
    "knowledge_cutoff_date": "10/01/2023",
    "label_offset_x": 45,
    "label_offset_y": 25
  },
  {
    "model_name": "gpt-4.1-2025-04-14",
    "release_date": "04/14/2025",
    "accuracy": 68.71,
    "accuracy_finetuned": null,
    "percent_error": 0.98,
    "has_seen_data": true,
    "knowledge_cutoff_date": "06/01/2024",
    "label_offset_x": -20,
    "label_offset_y": 35
  },
  {
    "model_name": "olmo-3-7b",
    "release_date": "11/20/2025",
    "accuracy": 54.0,
    "accuracy_finetuned": 79.0,
    "percent_error": 0,
    "has_seen_data": false,
    "knowledge_cutoff_date": "12/01/2024",
    "label_offset_x": -20,
    "label_offset_y": -25
  },
  {
    "model_name": "gpt-5.2-2025-12-11",
    "release_date": "12/11/2025",
    "accuracy": 71.51,
    "accuracy_finetuned": null,
    "percent_error": 0.96,
    "has_seen_data": true,
    "knowledge_cutoff_date": "08/31/2025",
    "label_offset_x": -90,
    "label_offset_y": 40
  },
  {
    "model_name": "olmo-3.1-32b",
    "release_date": "12/12/2025",
    "accuracy": 57.4,
    "accuracy_finetuned": 81.9,
    "percent_error": 0,
    "has_seen_data": false,
    "knowledge_cutoff_date": "12/01/2024",
    "label_offset_x": -65,
    "label_offset_y": 25
  }
]
'''


#   {
#     "model_name": "llama-3.1-8b",
#     "release_date": "07/23/2024",
#     "accuracy": 57.2,
#     "accuracy_finetuned": 81.9,
#     "percent_error": 0,
#     "has_seen_data": false,
#     "knowledge_cutoff_date": "12/01/2023",
#     "label_offset_x": 10,
#     "label_offset_y": 10
#   },


#   {
#     "model_name": "gpt-oss-20b",
#     "release_date": "08/05/2025",
#     "accuracy": 75.9,
#     "accuracy_finetuned": null,
#     "percent_error": 0,
#     "has_seen_data": true,
#     "knowledge_cutoff_date": "06/01/2024",
#     "label_offset_x": 10,
#     "label_offset_y": 10
#   },



# Load JSON
models = json.loads(data_json)

# Parse dates
for model in models:
    model["release_dt"] = (
        datetime.strptime(model["release_date"], "%m/%d/%Y")
        if model["release_date"] is not None
        else None
    )
    model["knowledge_cutoff_dt"] = (
        datetime.strptime(model["knowledge_cutoff_date"], "%m/%d/%Y")
        if model["knowledge_cutoff_date"] is not None
        else None
    )

# Sort by release date (null dates go last)
models.sort(key=lambda m: (m["release_dt"] is None, m["release_dt"] or datetime.max))

# Helper: shift an x-position by a fixed number of pixels in display space
def shifted_x_by_pixels(ax, x_dt, y_val, pixel_shift):
    x_num = mdates.date2num(x_dt)
    x_disp, y_disp = ax.transData.transform((x_num, y_val))
    x_shifted_num, _ = ax.transData.inverted().transform((x_disp + pixel_shift, y_disp))
    return x_shifted_num

# Create a broken y-axis: bottom shows 50-85, top shows only 100
fig, (ax_top, ax) = plt.subplots(
    2, 1,
    sharex=True,
    figsize=(15, 8),
    gridspec_kw={"height_ratios": [1, 8], "hspace": 0.05}
)

# CLadder dataset published marker
cladder_date = datetime.strptime("05/25/2023", "%m/%d/%Y")

# X range: include release dates, CLadder line, and any knowledge cutoff dates
all_dates = [cladder_date]
all_dates += [m["release_dt"] for m in models if m["release_dt"] is not None]
all_dates += [m["knowledge_cutoff_dt"] for m in models if m["knowledge_cutoff_dt"] is not None]

min_date = min(all_dates) - timedelta(days=45)
max_date = max(all_dates) + timedelta(days=60)

ax.set_xlim(min_date, max_date)

# Broken y-axis limits
ax.set_ylim(50, 85)
ax_top.set_ylim(99, 100)

# Y ticks
ax.set_yticks([50, 55, 60, 65, 70, 75, 80, 85])
ax_top.set_yticks([100])

# Hide touching spines
ax_top.spines["bottom"].set_visible(False)
ax.spines["top"].set_visible(False)

# Hide top x-axis ticks/labels
ax_top.tick_params(axis="x", which="both", bottom=False, top=False, labelbottom=False)

# Draw y-axis break marks so the top one matches the bottom visually
break_kwargs = dict(color="k", clip_on=False, linewidth=1.2)

dx = 0.006
dy_bottom = 0.006

# Because the top panel is much shorter, scale its squiggle height up
fig.canvas.draw()
top_h = ax_top.get_position().height
bottom_h = ax.get_position().height
dy_top = dy_bottom * (bottom_h / top_h)

zig_x = (-dx, -dx/3, dx/3, dx)

# Bottom of top axis: same pronounced up-down-up-down pattern, scaled to match visually
ax_top.plot(
    zig_x,
    (-dy_top, +dy_top, -dy_top, +dy_top),
    transform=ax_top.transAxes,
    **break_kwargs
)

# Top of bottom axis
ax.plot(
    zig_x,
    (1 - dy_bottom, 1 + dy_bottom, 1 - dy_bottom, 1 + dy_bottom),
    transform=ax.transAxes,
    **break_kwargs
)

# Axes labels
ax.set_xlabel("Time", fontsize=15)
ax.set_ylabel("Model Accuracy (%)", fontsize=15)

# X-axis ticks every 6 months
ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 7]))
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))

# Tick formatting
ax.tick_params(axis="x", labelsize=14)
ax.tick_params(axis="y", labelsize=14)
ax_top.tick_params(axis="y", labelsize=14)

# Lighter solid gridlines
for target_ax in (ax, ax_top):
    target_ax.grid(True, linestyle="-", linewidth=0.6, alpha=0.18)
    target_ax.set_axisbelow(True)

# Force layout so transforms are accurate before pixel-based line shortening
fig.canvas.draw()

# Draw knowledge cutoff lines first (behind points)
for m in models:
    if (
        m["knowledge_cutoff_dt"] is not None
        and m["release_dt"] is not None
        and m["accuracy"] is not None
    ):
        line_color = "lightcoral" if m["has_seen_data"] else "lightskyblue"

        cutoff_x_num = mdates.date2num(m["knowledge_cutoff_dt"])
        release_x_num = mdates.date2num(m["release_dt"])

        # Start the line 5 pixels to the right of the circle so it does not poke inside
        line_start_x_num = shifted_x_by_pixels(ax, m["knowledge_cutoff_dt"], m["accuracy"], 8)

        # line from just-right-of-circle to release date
        ax.plot(
            [line_start_x_num, release_x_num],
            [m["accuracy"], m["accuracy"]],
            color=line_color,
            lw=3.0,
            alpha=0.50,
            zorder=1
        )

        # open circle at the cutoff end
        ax.plot(
            cutoff_x_num,
            m["accuracy"],
            marker="o",
            markersize=8,
            markerfacecolor="none",
            markeredgecolor=line_color,
            markeredgewidth=3.0,
            alpha=0.50,
            zorder=2
        )

# Draw fine-tuning vertical lines and blue diamond markers
for target_ax in (ax, ax_top):
    for m in models:
        if (
            m["release_dt"] is None
            or m["accuracy"] is None
            or m.get("accuracy_finetuned") is None
        ):
            continue

        # vertical gray line
        target_ax.plot(
            [m["release_dt"], m["release_dt"]],
            [m["accuracy"], m["accuracy_finetuned"]],
            color="gray",
            lw=2.5,
            alpha=0.60,
            zorder=2
        )

        # blue diamond at fine-tuned accuracy
        target_ax.plot(
            m["release_dt"],
            m["accuracy_finetuned"],
            marker="D",
            markersize=7,
            color="blue",
            markerfacecolor="blue",
            markeredgecolor="blue",
            zorder=4
        )

# Draw error bars and points on both axes (so anything near 100 shows correctly on the top axis)
for target_ax in (ax, ax_top):
    for m in models:
        if m["release_dt"] is None or m["accuracy"] is None:
            continue

        point_color = "red" if m["has_seen_data"] else "blue"
        percent_error = m["percent_error"]

        if percent_error is not None and percent_error > 0:
            target_ax.errorbar(
                m["release_dt"],
                m["accuracy"],
                yerr=percent_error,
                fmt="o",
                color=point_color,
                ecolor=point_color,
                elinewidth=1.2,
                capsize=4,
                markersize=7,
                zorder=3
            )
        else:
            target_ax.plot(
                m["release_dt"],
                m["accuracy"],
                "o",
                color=point_color,
                markersize=7,
                zorder=3
            )

cladder_dash = (0, (2, 2))  # 3pt dash, 3pt gap

for target_ax in (ax, ax_top):
    target_ax.axvline(
        cladder_date,
        color="gray",
        linestyle=cladder_dash,
        linewidth=1.2,
        zorder=1
    )



# Visually connect selected vertical lines across the broken-axis gap
def connect_vertical_gap(fig, ax_bottom, ax_top, x_dt, **line_kwargs):
    x_num = mdates.date2num(x_dt)

    # Convert the x-position from data coordinates into figure coordinates
    x_disp, _ = ax_bottom.transData.transform((x_num, ax_bottom.get_ylim()[1]))
    x_fig, _ = fig.transFigure.inverted().transform((x_disp, 0))

    # Gap boundaries in figure coordinates
    bottom_top = ax_bottom.get_position().y1
    top_bottom = ax_top.get_position().y0

    line = Line2D(
        [x_fig, x_fig],
        [bottom_top, top_bottom],
        transform=fig.transFigure,
        clip_on=False,
        **line_kwargs
    )

    fig.add_artist(line)


# Connect the CLadder dashed line across the gap
connect_vertical_gap(
    fig,
    ax,
    ax_top,
    cladder_date,
    color="gray",
    linestyle=cladder_dash,
    linewidth=1.2,
    zorder=1
)


# Connect the right border/spine across the gap
def connect_right_spine_gap(fig, ax_bottom, ax_top, **line_kwargs):
    right_x = ax_bottom.get_position().x1
    bottom_top = ax_bottom.get_position().y1
    top_bottom = ax_top.get_position().y0

    line = Line2D(
        [right_x, right_x],
        [bottom_top, top_bottom],
        transform=fig.transFigure,
        clip_on=False,
        **line_kwargs
    )

    fig.add_artist(line)


connect_right_spine_gap(
    fig,
    ax,
    ax_top,
    color="black",
    linewidth=0.8,
    zorder=5
)

# Connect vertical grid lines across the broken-axis gap
for tick_num in ax.get_xticks():
    tick_dt = mdates.num2date(tick_num).replace(tzinfo=None)

    if min_date <= tick_dt <= max_date:
        connect_vertical_gap(
            fig,
            ax,
            ax_top,
            tick_dt,
            color="gray",
            linestyle="-",
            linewidth=0.6,
            alpha=0.18,
            zorder=0
        )


# CLadder label on the top axis
ax_top.text(
    cladder_date + timedelta(days=18),
    99.85,
    "CLadder Dataset Published\nMay 25 2023",
    ha="left",
    va="top",
    fontsize=14,
    zorder=4
)

# Draw labels with connector lines, using per-model JSON offsets
for m in models:
    if m["release_dt"] is None or m["accuracy"] is None:
        continue

    dx = m["label_offset_x"]
    dy = m["label_offset_y"]

    ax.annotate(
        m["model_name"],
        xy=(m["release_dt"], m["accuracy"]),
        xytext=(dx, dy),
        textcoords="offset points",
        fontsize=14,
        fontfamily="Consolas",
        color="0.2",
        ha="center",
        va="center",
        arrowprops=dict(
            arrowstyle="-",
            color="0.5",
            lw=1.5,
            shrinkA=0,
            shrinkB=4
        ),
        bbox=dict(
            boxstyle="round,pad=0.15",
            facecolor="white",
            edgecolor="none",
            alpha=0.85
        ),
        zorder=4
    )

# Legend
knowledge_cutoff_handle = (
    Line2D(
        [0], [0],
        marker='o',
        linestyle='None',
        markerfacecolor='none',
        markeredgecolor='purple',
        markeredgewidth=3.0,
        markersize=10,
        alpha=0.55
    ),
    Line2D(
        [0, 1], [0, 0],
        color='purple',
        lw=3.2,
        alpha=0.55
    )
)

class HandlerVerticalLineWithMarker(HandlerBase):
    def create_artists(
        self, legend, orig_handle, xdescent, ydescent, width, height, fontsize, trans
    ):
        x = xdescent + width / 2

        extra = height * 0.1
        y0 = ydescent - extra
        y1 = ydescent + height + extra

        # vertical grey line
        line = Line2D(
            [x, x],
            [y0, y1],
            color=orig_handle.get_color(),
            lw=orig_handle.get_linewidth(),
            alpha=orig_handle.get_alpha(),
            transform=trans,
            clip_on=False
        )

        # blue marker at the top of the line
        marker = Line2D(
            [x],
            [y1],
            linestyle='None',
            marker='D',              # same as the graph
            markersize=7,
            markerfacecolor='blue',
            markeredgecolor='blue',
            transform=trans,
            clip_on=False
        )

        return [line, marker]

fine_tune_handle = Line2D(
    [], [],
    color='gray',
    lw=3.5,
    alpha=0.80
)

legend_handles = [
    Line2D(
        [0], [0],
        marker='o',
        linestyle='None',
        color='red',
        markerfacecolor='red',
        markersize=11
    ),
    Line2D(
        [0], [0],
        marker='o',
        linestyle='None',
        color='blue',
        markerfacecolor='blue',
        markersize=11
    ),
    knowledge_cutoff_handle,
    fine_tune_handle
]

legend_labels = [
    'Suspected Exposure to CLadder',
    'No Suspected Exposure to CLadder',
    'Knowledge cutoff date',
    'Accuracy change after fine-tuning'
]

# Make sure the top axes (and its legend) draw above the lower panel
ax_top.set_zorder(3)
ax.set_zorder(2)
ax_top.patch.set_alpha(0)

leg = ax_top.legend(
    legend_handles,
    legend_labels,
    loc="upper left",
    bbox_to_anchor=(0.01, 0.90),   # a little padding from the y-axis
    borderaxespad=0.0,
    fontsize=14,
    handlelength=2.0,
    handler_map={
        tuple: HandlerTuple(ndivide=None, pad=0.0),
        fine_tune_handle: HandlerVerticalLineWithMarker()
    }
)
leg.set_zorder(100)

plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig("Timeline Plot.png", dpi=400, bbox_inches="tight", pad_inches=20/300)
plt.show()