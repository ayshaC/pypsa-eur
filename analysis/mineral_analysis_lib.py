"""
Function library and shared configuration for mineral_analysis.ipynb.

This module holds the plotting/analysis functions and the static configuration
(colours, mineral lists, paths) that were previously defined inline in the
notebook. The data frames are loaded by `import_data()` and passed explicitly
into the functions that need them, so there is no hidden module-level state.

Usage in the notebook:
    from mineral_analysis_lib import import_data, compute_mineral_usage, ...

    (mineral_intensities, tech_scenarios, mineral_avail,
     non_energy_demand, grid_demand, mineral_alloc_factors) = import_data()
"""

import math

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from IPython.display import display

with open("../config/plotting.default.yaml") as f:
    pypsa_config = yaml.safe_load(f)

tech_colors = pypsa_config["plotting"]["tech_colors"]
nice_names = pypsa_config["plotting"]["nice_names"]

all_minerals = [
    "Aluminium",
    "Cobalt",
    "Copper",
    "Dysprosium",
    "Gallium",
    "Germanium",
    "Graphite",
    "Iridium",
    "Lithium",
    "Manganese",
    "Neodymium",
    "Nickel",
    "Praseodymium",
    "Platinum",
    "Vanadium",
]
rare_earths = ["Neodymium", "Praseodymium", "Dysprosium"]
iea_minerals = ["Copper", "Cobalt", "Lithium", "Nickel", "REE", "Graphite"]

# paths for EV data calculation
TRANSPORT_DATA_PATH = "../resources/all-countries/transport_data.csv"
POPULATION_DATA_PATH = pd.read_csv(
    "../resources/all-countries/pop_layout_base_s_40.csv"
)

minerals_color_map = {
    "Lithium": "#ff7f00",
    "Cobalt": "#a6cee3",
    "Copper": "#fb9a99",
    "Nickel": "black",
    "Graphite (natural and synthetic)": "#303234",
    "Graphite": "#303234",
    "REE": "#b15928",
    "Magnet rare earth elements": "#b15928",
    "Dysprosium": "#b15928",
    "Neodymium": "#b15928",
    "Gallium": "#ffff99",
    "Germanium": "#6a3d9a",
    "Iridium": "#33a02c",
    "Manganese": "#fdbf6f",
    "Vanadium": "#e31a1c",
    "Platinum": "blue",
    "Indium": "#cab2d6",
}

tech_colors["battery electric vehicles"] = tech_colors["BEV charger"]

# Defaults used by the plotting functions when no colour/label map is passed
DEFAULT_TECH_COLORS = tech_colors
DEFAULT_NICE_NAMES = nice_names
tech_colors["other"] = "#454545"
tech_colors["building heat demand"] = tech_colors["heat"]
tech_colors["ambient heat"] = tech_colors["heat pump"]
tech_colors["residential electricity demand"] = "#72709c"
tech_colors["industry electricity demand"] = tech_colors["electricity"]
tech_colors["hydrogen demand"] = tech_colors["land transport fuel cell"]
tech_colors["agriculture machinery"] = tech_colors["land transport fuel cell"]
# tech_colors["methane demand"] = tech_colors["helmeth"]
tech_colors["liquid hydrocarbon demand"] = tech_colors["kerosene for aviation"]
tech_colors["aviation fuels"] = tech_colors["kerosene for aviation"]
tech_colors["shipping fuels"] = tech_colors["shipping methanol"]
tech_colors["biomass demand"] = tech_colors["biogas"]
tech_colors["biogas upgrading"] = tech_colors["biogas"]
tech_colors["hydrogen for industry"] = tech_colors["H2 for industry"]
tech_colors["hydrogen-to-power/heat"] = tech_colors["gas-to-power/heat"]
tech_colors["hydrogen for land transport"] = "#8487e8"
# cost-group labels produced by rename_techs_costs
tech_colors["power-to-hydrogen"] = tech_colors["H2 Electrolysis"]
tech_colors["direct air capture"] = tech_colors["DAC"]
tech_colors["carbon capture"] = tech_colors["CO2 sequestration"]
tech_colors["fossil oil and gas"] = tech_colors["gas"]


def import_data():
    """
    Load and return the mineral data frames used throughout the analysis.

    Returned in a fixed order for explicit unpacking in the notebook:
        (mineral_intensities, tech_scenarios, mineral_avail,
         non_energy_demand, grid_demand, mineral_alloc_factors) = import_data()
    """
    mineral_alloc_factors = {
        "GDP share": 0.213,
        "Per capita share": 0.068,
        "Per capita share corrected for energy use": 0.046,
    }

    mineral_intensities = pd.read_csv("data/mineral_intensity_combined.csv")

    tech_scenarios = pd.read_csv("data/tech_market_shares.csv")

    # mineral_alloc = pd.read_csv("data/mineral_alloc.csv")

    mineral_avail = pd.read_csv("data/mineral_availability.csv")
    mineral_avail["Value (Mt)"] = mineral_avail["Value (kt)"] / 1000

    non_energy_demand = pd.read_csv("data/non_energy_demand.csv")
    # non_energy_demand["Value entso-e (Mt)"] = non_energy_demand["Value entso-e (kt)"]/1000
    # non_energy_demand = non_energy_demand.set_index("Mineral")
    non_energy_demand["GDP-adjusted value entso-e (Mt)"] = (
        non_energy_demand["Value global (kt)"]
        * mineral_alloc_factors["GDP share"]
        / 1000
    )

    grid_demand = pd.read_csv("data/grid_demand.csv")
    grid_demand["GDP-adjusted value entso-e (Mt)"] = (
        grid_demand["Value global (kt)"] * mineral_alloc_factors["GDP share"] / 1000
    )
    return (
        mineral_intensities,
        tech_scenarios,
        mineral_avail,
        non_energy_demand,
        grid_demand,
        mineral_alloc_factors,
    )


# EV share pathway from config (can be replaced by reading YAML)
land_transport_electric_share = {
    2020: 0.0,
    2025: 0.05,
    2030: 0.20,
    2035: 0.45,
    2040: 0.70,
    2045: 0.85,
    2050: 1.0,
}

# Assumptions
start_year = 2025  # set to your "now"
end_year = 2050
ev_lifetime_years = 15


# function that calculates stock of EVs by year, given the electric share, and returns the total EVs sold between start year and end year
def get_total_EVs(
    start_year, end_year, ev_lifetime_years, land_transport_electric_share
):

    # Build annual EV share series via linear interpolation
    share_points = pd.Series(land_transport_electric_share, dtype=float).sort_index()
    years = np.arange(start_year, end_year + 1)
    share_annual = share_points.reindex(
        np.arange(share_points.index.min(), end_year + 1)
    )
    share_annual = share_annual.interpolate("linear").ffill().bfill().loc[years]

    # Baseline fleet size from transport data (energy_totals_year=2023 in default config)
    # nodal_transport_data = build_nodal_transport_data(TRANSPORT_DATA_PATH, POPULATION_DATA_PATH, 2023)
    transport_data = pd.read_csv(TRANSPORT_DATA_PATH, index_col=[0, 1])
    td = transport_data.xs(2023, level="year")
    fleet_total_2023 = td["number cars"].sum()

    # Keep total fleet constant unless you have a separate fleet-growth series
    fleet = pd.Series(fleet_total_2023, index=years, dtype=float)

    # Target EV stock each year
    ev_stock = share_annual * fleet

    # Stock-turnover model with fixed retirement age
    # sales[t] = stock increase + retirements at t, where retirements[t] = sales[t-lifetime]
    sales = pd.Series(0.0, index=years)
    retirements = pd.Series(0.0, index=years)

    for i, y in enumerate(years):
        # get target EV stock from the previous year
        prev_stock = ev_stock.iloc[i - 1] if i > 0 else 0.0

        # get needed increase in stock to get to target share
        stock_increase = max(ev_stock.loc[y] - prev_stock, 0.0)

        # replacement sales needed
        repl = (
            sales.loc[y - ev_lifetime_years]
            if (y - ev_lifetime_years) in sales.index
            else 0.0
        )

        retirements.loc[y] = repl
        sales.loc[y] = stock_increase + repl

    results_ev_turnover = pd.DataFrame(
        {
            "ev_share": share_annual,
            "fleet_total": fleet,
            "ev_stock": ev_stock,
            "retirements": retirements,
            "ev_sales": sales,
            "ev_sales_cumulative": sales.cumsum(),
        }
    )

    total_ev_sold_2026_2050 = results_ev_turnover["ev_sales"].sum()
    # print(f"Estimated total EVs sold ({start_year}-{end_year}): {total_ev_sold_2026_2050:,.0f}")
    return results_ev_turnover, total_ev_sold_2026_2050


def compute_ev_mineral_usage(
    start_year,
    end_year,
    ev_lifetime_years,
    land_transport_electric_share,
    minerals,
    scenario,
    mineral_intensities=None,
    techs=None,
):
    # EV battery size in MWh, according to the config file
    BEV_battery_size = 0.05

    # get transport data from resources file
    results_ev_turnover, total_cars = get_total_EVs(
        start_year, end_year, ev_lifetime_years, land_transport_electric_share
    )

    evs_capacity_df = pd.DataFrame(
        {
            "component": ["Store", "Store"],
            "carrier": ["EV motor", "EV battery"],
            "capacity": [total_cars, total_cars * BEV_battery_size],
        }
    )

    # merge with LCI data
    # need to remove EV batteries from regular mineral computation so it's not double counted

    evs_merged = pd.merge(
        evs_capacity_df,
        techs,
        left_on="carrier",
        right_on="PyPSA technology",
        how="inner",
    )
    evs_merged = pd.merge(
        evs_merged,
        mineral_intensities,
        left_on="LCI name",
        right_on="Activity",
        how="inner",
    )

    evs_merged = evs_merged.rename(columns={"capacity": "carrier capacity"})
    evs_merged["Sub tech capacity"] = evs_merged["carrier capacity"].mul(
        evs_merged[scenario], axis=0
    )
    evs_merged[minerals] = evs_merged[minerals].mul(
        evs_merged["Sub tech capacity"], axis=0
    )

    return evs_merged

    # df = pd.concat([df, evs_merged], ignore_index=True)


def compute_mineral_usage(
    network_key,
    n,
    minerals,
    scenario,
    mineral_intensities=None,
    techs=None,
    capacity_col_name="capacity",
    verbose=False,
    evs=True,
):
    """
    Return (df_merged, totals_series) for a given network in dict n.

    - df_merged: expanded capacities merged with intensity data. Contains mineral columns
                 multiplied by capacity (absolute mineral amount per row).
    - totals_series: Series indexed by mineral names with total mineral demand (units: same as conversion below).

    Behavior / assumptions:
    - Uses `n[network_key].statistics.expanded_capacity()` and expects capacity in column 0 (renamed to 'capacity').
    - If techs_map is provided, maps carriers via techs_map[lci_map_key] into column 'lci_technology'.
    - Tries to merge intensity data:
         * primary: merge mineral_intensities on left_on='lci_technology' or provided name -> right_on='Activity'
         * fallback: merge mineral_intensities_carrara on left_on='Other source name' -> right_on='Technology'
      Coalesces mineral columns preferring the primary source.
    - Returns totals in Mt if you divide by 1e9 (the code below returns totals_in_Mt).
    """

    # only want ENERGY CAPACITY (ignore other carriers like co2, heat...)
    elec_carriers = [
        "AC",
        "DC",
        "Battery Storage",
        "home battery",
        "low voltage",
        "EV battery",
        "Battery Storage",
        "battery",
    ]

    # 1) expanded capacities
    df = pd.DataFrame(
        n[network_key]
        .statistics.expanded_capacity(bus_carrier=elec_carriers)
        .reset_index()
    ).rename(columns={0: capacity_col_name})

    # muptiply the H2 capacity by -1 (because it is electrical input not output)
    df.loc[df["carrier"] == "H2 Electrolysis", capacity_col_name] *= -1

    # rename different heat pump types to "heat pump"
    hp_carriers = [
        "residential rural air heat pump",
        "residential rural ground heat pump",
        "residential urban decentral air heat pump",
        "services rural air heat pump",
        "services rural ground heat pump",
        "services urban decentral air heat pump",
        "urban central air heat pump",
    ]

    df["carrier_renamed"] = np.where(
        df["carrier"].isin(hp_carriers), "heat pump", df["carrier"]
    )

    # use optimal battery capacity, not expanded
    # df_opt = pd.DataFrame(n[network_key].statistics.installed_capacity(bus_carrier=elec_carriers).reset_index()).rename(columns={0: capacity_col_name})
    # df.loc[df['carrier'] == 'EV battery', 'capacity'] = df_opt[df_opt['carrier']=='EV battery']['capacity'].values[0]

    df = pd.merge(
        df, techs, left_on="carrier_renamed", right_on="PyPSA technology", how="inner"
    )

    df = pd.merge(
        df, mineral_intensities, left_on="LCI name", right_on="Activity", how="inner"
    )

    # 6) compute absolute mineral use per row: intensity * capacity
    # Note: intensity units must match this multiplication. Adjust conversion factor as needed.
    df = df.rename(columns={"capacity": "carrier capacity"})
    df["Sub tech capacity"] = df["carrier capacity"].mul(df[scenario], axis=0)
    df[minerals] = df[minerals].mul(df["Sub tech capacity"], axis=0)

    # remove EV battery capacity because we calculate it manually
    df = df[df["carrier_renamed"] != "EV battery"]

    # separate process for EVs:
    if evs:
        evs_merged = compute_ev_mineral_usage(
            2025,
            2050,
            15,
            land_transport_electric_share,
            minerals,
            scenario,
            mineral_intensities,
            techs,
        )
        df = pd.concat([df, evs_merged], ignore_index=True)

    # combine offwind columns
    offwind_carriers = ["Offshore Wind (AC)", "Offshore Wind (DC)"]
    df["carrier_renamed"] = np.where(
        df["carrier_renamed"].isin(offwind_carriers),
        "Offshore wind",
        df["PyPSA technology"],
    )

    # 7) aggregate totals (convert to Mt: divide by 1e9)

    totals_Mt = df[minerals].sum() / 1e9

    if verbose:
        print(
            f"Network {network_key}: merged rows {len(df)}; minerals found from sources: {', '.join([m for m in minerals if not df[m].isna().all()])}"
        )

    return df, totals_Mt


def compare_networks(
    network_keys,
    n,
    minerals,
    techs,
    mineral_intensities=None,
    show_plot=True,
    figsize=(12, 6),
):
    """
    Compute mineral totals for multiple network keys and return a DataFrame of totals (minerals x networks).
    Also plots a side-by-side bar chart comparing totals (Mt).
    """
    results = {}
    details = {}

    for k in network_keys:
        dfk, tot = compute_mineral_usage(k, n, minerals, mineral_intensities, techs)
        results[k] = tot
        details[k] = dfk

    comparison = pd.DataFrame(results)

    if show_plot:
        # Optionally sort minerals by total across networks (largest minerals shown first in legend)
        comparison = comparison.loc[
            comparison.sum(axis=1).sort_values(ascending=False).index
        ]

        # Transpose so rows = networks, columns = minerals and plot stacked bars
        stack_df = comparison.T  # index: networks, columns: minerals

        ax = stack_df.plot.bar(stacked=True, figsize=figsize, colormap="tab20")
        ax.set_ylabel("Mineral use [Mt]")
        ax.set_xlabel("Network")
        ax.set_title("Mineral demand breakdown per network")
        plt.xticks(rotation=0)
        ax.legend(title="Mineral", bbox_to_anchor=(1.02, 1), loc="upper left")
        plt.tight_layout()
        plt.show()

    return comparison, details


def stacked_barh_signed(
    data,
    ax,
    *,
    tech_colors=None,
    nice_names=None,
    y_label="Total",
    group_labels=None,
    annotate_thresh=0.03,
    annotate_decimals=0,
    title=None,
    group_bracket_height=0.15,  # Height above bar for group brackets
    bar_height=0.6,
    group_label_font_size=9,
    show_legend=True,
):
    """
    Plot horizontal stacked bars robust to negative values.

    Parameters
    ----------
    - data: pandas Series OR dict of Series
      - If Series with MultiIndex: single stacked bar with grouped bracket annotations
      - If Series with single index: single stacked bar (original behavior)
      - If dict: single stacked bar with grouped bracket annotations above
    - ax: matplotlib axis
    - tech_colors: dict carrier -> color (fallback '#888888')
    - nice_names: dict carrier -> pretty label
    - y_label: str, text for the y-axis label
    - group_labels: dict {group_key: label}, nice labels for group annotations (for dict/MultiIndex data)
    - annotate_thresh: fraction threshold for showing labels
    - annotate_decimals: number of decimal places for annotations (default 0)
    - title: optional plot title
    - group_bracket_height: vertical offset for group bracket annotations
    - bar_height: height of the bar
    - show_legend: bool, whether to display the legend (default True)
    """
    if tech_colors is None:
        tech_colors = DEFAULT_TECH_COLORS
    if nice_names is None:
        nice_names = DEFAULT_NICE_NAMES
    if group_labels is None:
        group_labels = {}

    # Check if data is MultiIndex Series
    is_multiindex = isinstance(data, pd.Series) and isinstance(
        data.index, pd.MultiIndex
    )

    # Handle both Series and dict inputs
    # NEW: multiple bars if DataFrame (rows=scenarios, cols=carriers)
    if isinstance(data, pd.DataFrame):
        legend_handles = {}
        y_positions = np.arange(len(data.index))

        for y_pos, (_, row) in enumerate(data.iterrows()):
            _plot_single_bar(
                row.fillna(0.0),
                ax,
                y_pos,
                tech_colors,
                nice_names,
                annotate_thresh,
                bar_height,
                legend_handles,
                annotate_decimals,
            )

        ax.set_yticks(y_positions)
        ax.set_yticklabels([str(x) for x in data.index])
        ax.set_ylabel("Scenario")
        ax.invert_yaxis()  # optional: first scenario at top

        if title:
            ax.set_title(title, pad=20)

        if show_legend and legend_handles:
            ax.legend(
                handles=list(legend_handles.values()),
                title="Carrier",
                bbox_to_anchor=(1.05, 1),
                loc="upper left",
            )
        return

    if isinstance(data, dict) or is_multiindex:
        # Single stacked bar with group annotations
        legend_handles = {}

        # Track cumulative positions for positive and negative values separately
        left_pos_positive = 0.0  # Tracks rightward stacking from 0
        left_pos_negative = 0.0  # Tracks leftward stacking from 0

        # Store group boundaries separately for positive and negative portions
        # {group_key: {'pos': (start_x, end_x), 'neg': (start_x, end_x)}}
        group_boundaries = {}

        if is_multiindex:
            # Get unique group keys from first level
            group_keys = data.index.get_level_values(0).unique()
        else:
            # Dict case
            group_keys = data.keys()

        for group_key in group_keys:
            # Extract series for this group
            if is_multiindex:
                series = data.xs(group_key, level=0)
            else:
                series = data[group_key]

            series = series.dropna()

            # Split into positive and negative values (keep zeros with positives)
            pos_series = series[series >= 0].sort_values(ascending=False)
            neg_series = series[series < 0].sort_values(
                ascending=True
            )  # Sort ascending for leftward stacking

            # Initialize boundaries for this group
            group_boundaries[group_key] = {"pos": None, "neg": None}

            # --- Handle positive values (stack rightward) ---
            if len(pos_series) > 0:
                group_start_pos = left_pos_positive
                total_pos = pos_series.sum()

                for carrier, value in pos_series.items():
                    color = tech_colors.get(carrier, "#888888")
                    label = nice_names.get(carrier, carrier)

                    ax.barh(
                        0,
                        value,
                        left=left_pos_positive,
                        height=bar_height,
                        color=color,
                        edgecolor="white",
                    )

                    # Annotate if above threshold
                    if total_pos > 0 and (value / total_pos) >= annotate_thresh:
                        ax.text(
                            left_pos_positive + value / 2,
                            0,
                            f"{value:.{annotate_decimals}f}",
                            ha="center",
                            va="center",
                            fontsize=8,
                        )

                    left_pos_positive += value

                    # Add to legend (avoid duplicates)
                    if label not in legend_handles:
                        legend_handles[label] = mpatches.Patch(color=color, label=label)

                group_boundaries[group_key]["pos"] = (
                    group_start_pos,
                    left_pos_positive,
                )

            # --- Handle negative values (stack leftward) ---
            if len(neg_series) > 0:
                group_start_neg = left_pos_negative
                total_neg_abs = abs(neg_series.sum())

                for carrier, value in neg_series.items():
                    color = tech_colors.get(carrier, "#888888")
                    label = nice_names.get(carrier, carrier)

                    ax.barh(
                        0,
                        value,
                        left=left_pos_negative,
                        height=bar_height,
                        color=color,
                        edgecolor="white",
                    )

                    # Annotate if above threshold
                    if (
                        total_neg_abs > 0
                        and (abs(value) / total_neg_abs) >= annotate_thresh
                    ):
                        ax.text(
                            left_pos_negative + value / 2,
                            0,
                            f"{value:.{annotate_decimals}f}",
                            ha="center",
                            va="center",
                            fontsize=8,
                        )

                    left_pos_negative += value

                    # Add to legend (avoid duplicates)
                    if label not in legend_handles:
                        legend_handles[label] = mpatches.Patch(color=color, label=label)

                group_boundaries[group_key]["neg"] = (
                    left_pos_negative,
                    group_start_neg,
                )

        # Draw group brackets/annotations
        y_bracket_pos = bar_height / 2 + group_bracket_height  # Above bar
        y_bracket_neg = -bar_height / 2 - group_bracket_height  # Below bar

        for group_key, boundaries in group_boundaries.items():
            group_label = group_labels.get(group_key, group_key)

            # Draw positive bracket (above bar)
            if boundaries["pos"] is not None:
                x_start, x_end = boundaries["pos"]
                if x_end - x_start > 1e-6:  # Skip empty groups
                    x_mid = (x_start + x_end) / 2

                    # Draw horizontal line (bracket)
                    ax.plot(
                        [x_start, x_end],
                        [y_bracket_pos, y_bracket_pos],
                        color="black",
                        linewidth=1.5,
                        clip_on=False,
                    )

                    # Draw vertical ticks at ends
                    tick_height = 0.04
                    ax.plot(
                        [x_start, x_start],
                        [y_bracket_pos - tick_height, y_bracket_pos + tick_height],
                        color="black",
                        linewidth=1.5,
                        clip_on=False,
                    )
                    ax.plot(
                        [x_end, x_end],
                        [y_bracket_pos - tick_height, y_bracket_pos + tick_height],
                        color="black",
                        linewidth=1.5,
                        clip_on=False,
                    )

                    # Add group label above bracket
                    ax.text(
                        x_mid,
                        y_bracket_pos + 0.08,
                        group_label,
                        ha="center",
                        va="bottom",
                        fontsize=group_label_font_size,
                        fontweight="normal",
                        clip_on=False,
                    )

            # Draw negative bracket (below bar)
            if boundaries["neg"] is not None:
                x_start, x_end = boundaries["neg"]
                if x_end - x_start > 1e-6:  # Skip empty groups
                    x_mid = (x_start + x_end) / 2

                    # Draw horizontal line (bracket)
                    ax.plot(
                        [x_start, x_end],
                        [y_bracket_neg, y_bracket_neg],
                        color="black",
                        linewidth=1.5,
                        clip_on=False,
                    )

                    # Draw vertical ticks at ends
                    tick_height = 0.04
                    ax.plot(
                        [x_start, x_start],
                        [y_bracket_neg - tick_height, y_bracket_neg + tick_height],
                        color="black",
                        linewidth=1.5,
                        clip_on=False,
                    )
                    ax.plot(
                        [x_end, x_end],
                        [y_bracket_neg - tick_height, y_bracket_neg + tick_height],
                        color="black",
                        linewidth=1.5,
                        clip_on=False,
                    )

                    # Add group label below bracket
                    ax.text(
                        x_mid,
                        y_bracket_neg - 0.08,
                        group_label,
                        ha="center",
                        va="top",
                        fontsize=group_label_font_size,
                        fontweight="normal",
                        clip_on=False,
                    )

        # Set axis properties
        ax.set_yticks([0])
        ax.set_yticklabels([y_label])
        ax.axvline(0, color="k", lw=0.8, alpha=0.4)

        # Set x-limits with padding for both positive and negative
        xmax = left_pos_positive
        xmin = left_pos_negative
        pad = 0.05 * max(abs(xmax), abs(xmin), 1e-12)
        ax.set_xlim(xmin - pad, xmax + pad)

        # Adjust y-limits to show both upper and lower brackets
        ax.set_ylim(y_bracket_neg - 0.25, y_bracket_pos + 0.25)

    if title:
        ax.set_title(title, pad=20)

    # Create legend from collected handles
    if show_legend and legend_handles:
        ax.legend(
            handles=list(legend_handles.values()),
            title="Carrier",
            bbox_to_anchor=(1.05, 1),
            loc="upper left",
        )


def _plot_single_bar(
    s,
    ax,
    y_pos,
    tech_colors,
    nice_names,
    annotate_thresh,
    bar_height,
    legend_handles,
    annotate_decimals=0,
):
    """Helper function to plot a single stacked bar at given y position."""
    s = s.dropna()
    pos = s[s > 0].sort_values(ascending=False)
    neg = s[s < 0].sort_values()

    total_pos = pos.sum()
    total_neg_abs = (-neg).sum()

    left_pos = 0.0
    left_neg = 0.0

    # Plot positives (to the right)
    for key, v in pos.items():
        color = tech_colors.get(key, "#888888")
        label = nice_names.get(key, key)
        ax.barh(
            y_pos, v, left=left_pos, height=bar_height, color=color, edgecolor="white"
        )

        if total_pos > 0 and (v / total_pos) >= annotate_thresh:
            ax.text(
                left_pos + v / 2,
                y_pos,
                f"{v:.{annotate_decimals}f}",
                ha="center",
                va="center",
                fontsize=8,
            )
        left_pos += v

        # Add to legend (avoid duplicates)
        if label not in legend_handles:
            legend_handles[label] = mpatches.Patch(color=color, label=label)

    # Plot negatives (to the left)
    for key, v in neg.items():
        color = tech_colors.get(key, "#888888")
        label = nice_names.get(key, key)
        ax.barh(
            y_pos, v, left=left_neg, height=bar_height, color=color, edgecolor="white"
        )

        if total_neg_abs > 0 and ((-v) / total_neg_abs) >= annotate_thresh:
            ax.text(
                left_neg + v / 2,
                y_pos,
                f"{v:.{annotate_decimals}f}",
                ha="center",
                va="center",
                fontsize=8,
            )
        left_neg += v

        if label not in legend_handles:
            legend_handles[label] = mpatches.Patch(color=color, label=label)

    # Draw zero reference line
    ax.axvline(0, color="k", lw=0.8, alpha=0.4, zorder=1)

    # Set x-limits with padding
    xmax = max(0.0, left_pos)
    xmin = min(0.0, left_neg)

    # Update ax limits if needed (don't shrink)
    current_xlim = ax.get_xlim()
    pad = 0.05 * max(xmax, abs(xmin), 1e-12)
    new_xlim = (min(current_xlim[0], xmin - pad), max(current_xlim[1], xmax + pad))
    ax.set_xlim(new_xlim)


def plot_comparison_barh(
    data,
    ax,
    *,
    lower_col=None,
    upper_col=None,
    tech_colors=None,
    nice_names=None,
    group_labels=None,
    xlabel="Value",
    title=None,
    bar_height=0.6,
    group_label_font_size=9,
    marker_size=4,
    show_legend=True,
    annotate_decimals=1,
    annotate_font_size=8,
):
    """
    Plot horizontal bars comparing two columns with group brackets on y-axis.

    Parameters
    ----------
    - data: DataFrame with MultiIndex (group, item) and at least 2 columns for comparison
    - ax: matplotlib axis
    - lower_col: str or int, column name/index for lower bound (shown as 'x' marker). If None, uses first column
    - upper_col: str or int, column name/index for upper bound (shown as full bar). If None, uses second column
    - tech_colors: dict item -> color (fallback '#888888'). Uses second-level index for lookup
    - nice_names: dict item -> pretty label. Uses second-level index for lookup
    - group_labels: dict {group_key: label}, nice labels for group annotations
    - xlabel: str, label for x-axis
    - title: optional plot title
    - bar_height: height of the bars
    - group_label_font_size: font size for group labels
    - marker_size: size of the 'x' marker for lower bound values
    - show_legend: bool, whether to display the legend (default True)
    - annotate_decimals: number of decimal places for bar annotations (default 1)
    - annotate_font_size: font size for bar value annotations (default 8)
    """
    if tech_colors is None:
        tech_colors = DEFAULT_TECH_COLORS
    if nice_names is None:
        nice_names = DEFAULT_NICE_NAMES
    if group_labels is None:
        group_labels = {}

    # Determine which columns to use
    if lower_col is None:
        lower_col = data.columns[0]
    if upper_col is None:
        upper_col = data.columns[1] if len(data.columns) > 1 else data.columns[0]

    # Get unique groups in their original order
    groups = data.index.get_level_values(0).unique()

    # Build y-positions and group boundaries
    y_pos = 0
    group_boundaries = {}
    y_positions = []  # List of (item, y_position, label) tuples

    for group in groups:
        group_data = data.xs(group, level=0)
        group_start = y_pos

        for item in group_data.index:
            label = nice_names.get(item, item)
            y_positions.append((item, y_pos, label))
            y_pos += 1

        group_boundaries[group] = (group_start, y_pos - 1)

    # Calculate max and min values
    df = data[[lower_col, upper_col]]
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    max_val = df.max().max()
    min_val = df.min().min()

    if max_val == 0 or pd.isna(max_val):
        max_val = 1  # Avoid division by zero

    # Check if there are any negative values
    has_negative = min_val < 0 if pd.notna(min_val) else False

    # Plot bars
    for item, y, label in y_positions:
        # Find the item in the original dataframe
        # Need to search through all groups to find it
        for group in groups:
            try:
                group_data = data.xs(group, level=0)
                if item in group_data.index:
                    color = tech_colors.get(item, "#888888")

                    lower_val = group_data.loc[item, lower_col]
                    upper_val = group_data.loc[item, upper_col]

                    if np.isinf(upper_val):
                        ax.text(
                            0,
                            y,
                            " Infinity!",
                            ha="left",
                            va="center",
                            fontsize=annotate_font_size,
                        )

                    elif pd.isna(upper_val):
                        ax.text(
                            0,
                            y,
                            " NaN/None",
                            ha="left",
                            va="center",
                            fontsize=annotate_font_size,
                        )

                    elif upper_val == 0:
                        ax.text(
                            upper_val,
                            y,
                            " Not installed",
                            ha="left",
                            va="center",
                            fontsize=annotate_font_size,
                        )

                    else:
                        # Plot upper bound as full opaque bar
                        ax.barh(
                            y,
                            upper_val,
                            height=bar_height,
                            color=color,
                            edgecolor="white",
                            linewidth=0.5,
                        )

                        # Annotate bar with its value to the right
                        ax.text(
                            upper_val,
                            y,
                            f" {upper_val:.{annotate_decimals}f}",
                            ha="left",
                            va="center",
                            fontsize=annotate_font_size,
                        )

                    # Plot lower bound as 'x' marker
                    if pd.notna(lower_val) and np.isfinite(lower_val):
                        ax.plot(
                            lower_val,
                            y,
                            marker="x",
                            color="black",
                            markersize=marker_size,
                            markeredgewidth=1.5,
                            linestyle="",
                        )
                    break
            except KeyError:
                continue

    # Set axis properties first to get proper tick label positions
    y_ticks = [y for _, y, _ in y_positions]
    y_labels = [label for _, _, label in y_positions]

    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_labels)
    ax.set_ylim(-0.5, y_pos - 0.5)

    # Set x-axis limits
    if has_negative:
        # Has negative values - center around 0
        x_min = min_val - abs(min_val) * 0.05
        x_max = max_val * 1.05
    else:
        # No negative values - start at 0
        x_min = 0
        x_max = max_val * 1.05
    # if np.isinf(x_max):
    #     x_max = 1
    # if np.isinf(x_min):
    #     x_min = 0

    ax.set_xlim(x_min, x_max)

    # Force axis to draw so we can get label positions
    ax.figure.canvas.draw()

    # Get the position of the leftmost edge of the widest y-tick label
    # This will be our reference point for the brackets
    try:
        # Get bounding boxes of y-tick labels in display coordinates
        renderer = ax.figure.canvas.get_renderer()
        bboxes = [
            label.get_window_extent(renderer=renderer) for label in ax.get_yticklabels()
        ]

        if bboxes:
            # Find the leftmost edge among all labels (in display coordinates)
            leftmost_display = min([bbox.x0 for bbox in bboxes])

            # Convert from display to data coordinates
            # We need to transform a point at the left edge of the labels
            inv = ax.transData.inverted()
            # Use the y-coordinate from the middle of the plot
            y_middle_display = sum([bbox.y0 + bbox.y1 for bbox in bboxes]) / (
                2 * len(bboxes)
            )
            leftmost_x = inv.transform([[leftmost_display, y_middle_display]])[0][0]
        else:
            leftmost_x = x_min
    except Exception:
        # Fallback if we can't get label positions
        leftmost_x = x_min

    # Position brackets just to the left of the leftmost label edge
    bracket_offset = (x_max - x_min) * 0.06  # Small offset from labels
    x_bracket_pos = leftmost_x - bracket_offset

    # Draw group brackets on the left
    for group, (y_start, y_end) in group_boundaries.items():
        if y_end >= y_start:  # Skip empty groups
            y_mid = (y_start + y_end) / 2

            # Draw vertical line (bracket)
            ax.plot(
                [x_bracket_pos, x_bracket_pos],
                [y_start - 0.5, y_end + 0.5],
                color="black",
                linewidth=1.5,
                clip_on=False,
            )

            # Draw horizontal ticks at ends
            tick_width = (x_max - x_min) * 0.01
            ax.plot(
                [x_bracket_pos - tick_width, x_bracket_pos + tick_width],
                [y_start - 0.5, y_start - 0.5],
                color="black",
                linewidth=1.5,
                clip_on=False,
            )
            ax.plot(
                [x_bracket_pos - tick_width, x_bracket_pos + tick_width],
                [y_end + 0.5, y_end + 0.5],
                color="black",
                linewidth=1.5,
                clip_on=False,
            )

            # Add group label to the left of bracket
            group_label = group_labels.get(group, group)
            ax.text(
                x_bracket_pos - tick_width * 1.5,
                y_mid,
                group_label,
                ha="right",
                va="center",
                fontsize=group_label_font_size,
                fontweight="normal",
                clip_on=False,
            )

    ax.set_xlabel(xlabel)

    if title:
        ax.set_title(title)

    # Create custom legend
    if show_legend:
        from matplotlib.lines import Line2D

        legend_elements = [
            mpatches.Patch(facecolor="gray", label=upper_col),
            Line2D(
                [0],
                [0],
                marker="x",
                color="black",
                label=lower_col,
                markersize=marker_size,
                markeredgewidth=1.5,
                linestyle="",
            ),
        ]
        ax.legend(handles=legend_elements, loc="lower right")


def mineral_scenario_comparison_mixed_sources(
    n,
    base_scenario,
    network_scenarios,
    market_scenarios,
    mineral_specs,
    mineral_intensities,
    tech_scenarios,
    mineral_avail,
    non_energy_demand,
    grid_demand,
    mineral_alloc_factors,
    non_energy_col="GDP-adjusted value entso-e (Mt)",
    ncols=3,
    fig_title="",
):
    """
    Plot stacked mineral demand bars with mineral-specific non-energy and availability sources.
    Each entry in ``mineral_specs`` is a dict with required keys:
      - ``mineral``
      - ``DNEA_source``
      - ``DNEA_year``
      - ``avail_source``
      - ``avail_year``
      - ``avail_method``
    Optional keys per mineral:
      - ``energy_minerals``: list/str of minerals passed to ``compute_mineral_usage``
        (e.g. ["Graphite", "Graphite (synthetic)"] for the combined graphite bar).
      - ``non_energy_mineral``: mineral name used to fetch non-energy demand.
      - ``avail_mineral``: mineral name used to fetch availability.
      - ``color``: bar color override for energy demand segment.
    """

    bar_width = 0.6
    scenario_names = list(network_scenarios) + list(market_scenarios)
    n_scenarios = len(scenario_names)

    n_minerals = len(mineral_specs)
    nrows = math.ceil(n_minerals / ncols)
    fig, axs = plt.subplots(nrows, ncols, figsize=(12, nrows * 2.5 + 1))
    axes = np.array(axs, ndmin=1).flatten()

    def _energy_total_yearly(network_key, market_scenario, energy_minerals):
        _, totals_Mt = compute_mineral_usage(
            network_key,
            n,
            energy_minerals,
            market_scenario,
            mineral_intensities,
            tech_scenarios,
        )
        return float(totals_Mt.sum())

    for idx, spec in enumerate(mineral_specs):
        mineral = spec["mineral"]
        non_energy_mineral = spec.get("non_energy_mineral", mineral)
        avail_mineral = spec.get("avail_mineral", mineral)
        energy_minerals = spec.get("energy_minerals", mineral)

        dnea_mask = (
            (non_energy_demand["Mineral"] == non_energy_mineral)
            & (non_energy_demand["Year"] == spec["DNEA_year"])
            & (non_energy_demand["Source"] == spec["DNEA_source"])
        )
        dnea_series = non_energy_demand.loc[dnea_mask, non_energy_col]
        if dnea_series.empty:
            raise ValueError(
                f"No non-energy demand data for {mineral} with source={spec['DNEA_source']}, year={spec['DNEA_year']}"
            )
        dnea_value = float(dnea_series.iloc[0])

        grid_demand_mask = (grid_demand["Mineral"] == mineral) & (
            grid_demand["Year"] == spec["grid_demand_year"]
        )
        grid_demand_series = grid_demand.loc[
            grid_demand_mask, "GDP-adjusted value entso-e (Mt)"
        ]
        grid_demand_value = 0
        if not grid_demand_series.empty:
            grid_demand_value = float(grid_demand_series.iloc[0])

        avail_mask = (
            (mineral_avail["Mineral"] == avail_mineral)
            & (mineral_avail["Estimate method"] == spec["avail_method"])
            & (mineral_avail["Estimate year"] == spec["avail_year"])
            & (mineral_avail["Source"] == spec["avail_source"])
        )
        avail_series = mineral_avail.loc[avail_mask, "Value (Mt)"]

        energy_vals = []
        for network_key in network_scenarios:
            energy_vals.append(
                _energy_total_yearly(
                    network_key, "Market share current", energy_minerals
                )
            )
        for market_scenario in market_scenarios:
            energy_vals.append(
                _energy_total_yearly(base_scenario, market_scenario, energy_minerals)
            )

        energy_vals = np.array(energy_vals, dtype=float)
        dnea_vals = np.full(n_scenarios, dnea_value, dtype=float)
        grid_vals = np.full(n_scenarios, grid_demand_value, dtype=float)

        ax = axes[idx]
        energy_color = spec.get("color", minerals_color_map.get(mineral, "#3271a8"))

        ax.bar(
            scenario_names,
            dnea_vals,
            bar_width,
            label="Non-energy demand",
            color="#d9d9d9",
            zorder=2,
        )
        ax.bar(
            scenario_names,
            grid_vals,
            bar_width,
            bottom=dnea_vals,
            label="Grid demand",
            color="#696969",
            zorder=2,
        )
        ax.bar(
            scenario_names,
            energy_vals,
            bar_width,
            bottom=dnea_vals + grid_vals,
            color=energy_color,
            zorder=2,
        )

        if not avail_series.empty:
            avail_value = float(avail_series.iloc[0])
            min_prod_gdp = avail_value * mineral_alloc_factors["GDP share"]
            min_prod_pc = avail_value * mineral_alloc_factors["Per capita share"]
            min_prod_pce = (
                avail_value
                * mineral_alloc_factors["Per capita share corrected for energy use"]
            )

            ax.axhline(
                y=min_prod_gdp,
                linestyle="--",
                linewidth=1.5,
                zorder=3,
                color="blue",
                label="GDP-allocated supply",
            )
            ax.axhline(
                y=min_prod_pc,
                linestyle="--",
                linewidth=1.5,
                zorder=3,
                color="orange",
                label="Per-capita-allocated supply",
            )
            ax.axhline(
                y=min_prod_pce,
                linestyle="--",
                linewidth=1.5,
                zorder=3,
                color="green",
                label="Per-capita allocated supply\n(energy-corrected)",
            )

        ax.set_ylabel("Amount (Mt)")
        ax.set_xlabel("Scenario")
        ax.set_title(mineral)
        ax.grid(zorder=0)
        ax.tick_params(axis="x", labelrotation=90)

        # Distinguish network scenarios (left) from market scenarios (right)
        # with a light background tint and a vertical separator.
        # ax.axvspan(-0.5, n_network - 0.5, alpha=0.07, color="steelblue", zorder=0)
        # ax.axvspan(n_network - 0.5, n_scenarios - 0.5, alpha=0.07, color="darkorange", zorder=0)
        # ax.axvline(x=n_network - 0.5, color="gray", linestyle=":", linewidth=0.8, zorder=1)

    # Hide empty subplot axes when mineral count is not a multiple of ncols.
    for ax in axes[n_minerals:]:
        ax.set_visible(False)

    axs_2d = np.atleast_2d(axs)
    n_rows, n_cols = axs_2d.shape
    for i, ax in enumerate(axs_2d.flat):
        if not ax.get_visible():
            continue
        row = i // n_cols
        if row < n_rows - 1:
            ax.set_xlabel("")
            ax.tick_params(axis="x", labelbottom=False)

    handles, labels = axes[0].get_legend_handles_labels()

    # Create a dictionary to get unique labels and handles
    unique_labels = {}
    for handle, label in zip(handles, labels):
        if label not in unique_labels:
            unique_labels[label] = handle

    # network_patch = mpatches.Patch(facecolor="steelblue", alpha=0.4, linewidth=0, label="Technology scenarios")
    # market_patch = mpatches.Patch(facecolor="darkorange", alpha=0.4, linewidth=0, label="Sub-technology scenarios")

    fig.suptitle(fig_title, fontsize=14, y=0.99)
    fig.tight_layout(rect=[0, 0, 0.66, 0.99])

    legend_handles = list(unique_labels.values())
    legend_labels = list(unique_labels.keys())

    fig.legend(
        legend_handles,
        legend_labels,
        loc="upper left",
        bbox_to_anchor=(0.68, 0.95),
        frameon=False,
        borderaxespad=0.0,
        fontsize=10,
    )

    plt.show()


def mineral_scenario_comparison_mixed_sources_by_technology(
    n,
    base_scenario,
    network_scenarios,
    market_scenarios,
    mineral_specs,
    mineral_intensities,
    tech_scenarios,
    mineral_avail,
    non_energy_demand,
    grid_demand,
    mineral_alloc_factors,
    non_energy_col="GDP-adjusted value entso-e (Mt)",
    grid_demand_year="Cumulative",
    ncols=3,
    fig_title="",
    tech_group_col="carrier_renamed",
    min_tech_share=0.01,
    figwidth=13,
):
    """
    Plot mineral demand bars per scenario with energy demand split by technology.

    This function mirrors ``mineral_scenario_comparison_mixed_sources`` but breaks
    the energy-demand segment into technology contributions using the merged
    dataframe returned by ``compute_mineral_usage``.

    Parameters
    ----------
    - mineral_specs: list of dicts. Required keys per mineral:
      ``mineral``, ``DNEA_source``, ``DNEA_year``, ``avail_source``,
      ``avail_year``, ``avail_method``.
      Optional keys:
      ``energy_minerals``, ``non_energy_mineral``, ``avail_mineral``,
      ``grid_demand_year``.
    - tech_group_col: column used to aggregate technology contribution from
      ``compute_mineral_usage`` merged output.
    - min_tech_share: hide technologies with total contribution smaller than
      this share of total energy demand for each mineral subplot.
    """

    bar_width = 0.8
    scenario_names = list(network_scenarios) + list(market_scenarios)
    n_scenarios = len(scenario_names)

    n_minerals = len(mineral_specs)
    nrows = math.ceil(n_minerals / ncols)
    fig, axs = plt.subplots(nrows, ncols, figsize=(figwidth, nrows * 2.8 + 2))
    axes = np.array(axs, ndmin=1).flatten()

    def _to_list(value):
        if isinstance(value, (list, tuple, np.ndarray, pd.Index)):
            return list(value)
        return [value]

    def _energy_by_tech(network_key, market_scenario, energy_minerals):
        merged, _ = compute_mineral_usage(
            network_key,
            n,
            energy_minerals,
            market_scenario,
            mineral_intensities,
            tech_scenarios,
        )

        mineral_cols = [m for m in _to_list(energy_minerals) if m in merged.columns]
        if not mineral_cols:
            return pd.Series(dtype=float)

        if tech_group_col not in merged.columns:
            raise ValueError(
                f"Column '{tech_group_col}' not found in merged dataframe from compute_mineral_usage."
            )

        # Convert kg to Mt and sum selected mineral columns into one energy value per tech.
        tech_use_mt = (
            merged.groupby(tech_group_col)[mineral_cols].sum().sum(axis=1) / 1e9
        )
        tech_use_mt = tech_use_mt[tech_use_mt != 0].sort_values(ascending=False)
        return tech_use_mt

    inverse_nice_names = {v: k for k, v in nice_names.items()}
    tech_colors_lower = {str(k).strip().lower(): v for k, v in tech_colors.items()}

    def _resolve_tech_color(tech_name):
        if tech_name == "Other":
            return "#000000"

        pretty_name = nice_names.get(tech_name, tech_name)
        candidates = [tech_name, pretty_name, inverse_nice_names.get(tech_name)]

        for cand in candidates:
            if cand is None:
                continue
            if cand in tech_colors:
                return tech_colors[cand]
            cand_lower = str(cand).strip().lower()
            if cand_lower in tech_colors_lower:
                return tech_colors_lower[cand_lower]

        # Neutral fallback only for truly missing keys.
        return "#888888"

    for idx, spec in enumerate(mineral_specs):
        mineral = spec["mineral"]
        print(mineral)
        non_energy_mineral = spec.get("non_energy_mineral", mineral)
        avail_mineral = spec.get("avail_mineral", mineral)
        energy_minerals = spec.get("energy_minerals", mineral)
        local_grid_demand_year = spec.get("grid_demand_year", grid_demand_year)

        dnea_mask = (
            (non_energy_demand["Mineral"] == non_energy_mineral)
            & (non_energy_demand["Year"] == spec["DNEA_year"])
            & (non_energy_demand["Source"] == spec["DNEA_source"])
        )
        dnea_series = non_energy_demand.loc[dnea_mask, non_energy_col]
        if dnea_series.empty:
            raise ValueError(
                f"No non-energy demand data for {mineral} with source={spec['DNEA_source']}, year={spec['DNEA_year']}"
            )
        dnea_value = float(dnea_series.iloc[0])

        grid_demand_mask = (grid_demand["Mineral"] == mineral) & (
            grid_demand["Year"] == local_grid_demand_year
        )
        grid_demand_series = grid_demand.loc[
            grid_demand_mask, "GDP-adjusted value entso-e (Mt)"
        ]
        grid_demand_value = (
            float(grid_demand_series.iloc[0]) if not grid_demand_series.empty else 0.0
        )

        avail_mask = (
            (mineral_avail["Mineral"] == avail_mineral)
            & (mineral_avail["Estimate method"] == spec["avail_method"])
            & (mineral_avail["Estimate year"] == spec["avail_year"])
            & (mineral_avail["Source"] == spec["avail_source"])
        )
        avail_series = mineral_avail.loc[avail_mask, "Value (Mt)"]

        energy_by_scenario = []
        for network_key in network_scenarios:
            energy_by_scenario.append(
                _energy_by_tech(network_key, "Market share current", energy_minerals)
            )
        for market_scenario in market_scenarios:
            energy_by_scenario.append(
                _energy_by_tech(base_scenario, market_scenario, energy_minerals)
            )

        if energy_by_scenario:
            energy_df = pd.DataFrame(energy_by_scenario, index=scenario_names).fillna(
                0.0
            )
        else:
            energy_df = pd.DataFrame(index=scenario_names)

        if not energy_df.empty:
            totals_by_tech = energy_df.sum(axis=0)
            total_energy = float(totals_by_tech.sum())
            if total_energy > 0:
                threshold = min_tech_share * total_energy
                keep = totals_by_tech[totals_by_tech >= threshold].index
                other = totals_by_tech[totals_by_tech < threshold].index
                energy_df = energy_df.loc[:, keep]
                if len(other) > 0:
                    energy_df["Other"] = (
                        pd.DataFrame(energy_by_scenario, index=scenario_names)
                        .fillna(0.0)[other]
                        .sum(axis=1)
                    )

            energy_df = energy_df.reindex(
                columns=energy_df.sum(axis=0).sort_values(ascending=False).index
            )

        dnea_vals = np.full(n_scenarios, dnea_value, dtype=float)
        grid_vals = np.full(n_scenarios, grid_demand_value, dtype=float)

        ax = axes[idx]
        ax.bar(
            scenario_names,
            dnea_vals,
            bar_width,
            label="Non-energy demand",
            color="#d9d9d9",
            zorder=2,
        )
        ax.bar(
            scenario_names,
            grid_vals,
            bar_width,
            bottom=dnea_vals,
            label="Grid demand",
            color="#696969",
            zorder=2,
        )

        print("DNEA:", dnea_vals)
        print("Grid:", grid_vals)

        running_bottom = dnea_vals + grid_vals
        total_energy = 0
        print(energy_df)
        for tech in energy_df.columns:
            values = energy_df[tech].values
            if np.allclose(values, 0.0):
                continue

            tech_color = _resolve_tech_color(tech)
            label = nice_names.get(tech, tech)
            ax.bar(
                scenario_names,
                values,
                bar_width,
                bottom=running_bottom,
                color=tech_color,
                label=label,
                zorder=2,
            )
            total_energy += values
            running_bottom = running_bottom + values

        print("Energy demand:", total_energy)

        if not avail_series.empty:
            avail_value = float(avail_series.iloc[0])
            min_prod_gdp = avail_value * mineral_alloc_factors["GDP share"]
            min_prod_pc = avail_value * mineral_alloc_factors["Per capita share"]
            min_prod_pce = (
                avail_value
                * mineral_alloc_factors["Per capita share corrected for energy use"]
            )

            print(min_prod_gdp)

            ax.axhline(
                y=min_prod_gdp,
                linestyle="--",
                linewidth=1.5,
                zorder=3,
                color="blue",
                label="GDP-allocated supply",
            )
            ax.axhline(
                y=min_prod_pc,
                linestyle="--",
                linewidth=1.5,
                zorder=3,
                color="orange",
                label="Per-capita-allocated supply",
            )
            ax.axhline(
                y=min_prod_pce,
                linestyle="--",
                linewidth=1.5,
                zorder=3,
                color="green",
                label="Per-capita allocated supply\n(energy-corrected)",
            )

        ax.set_ylabel("Amount (Mt)")
        ax.set_xlabel("Scenario")
        ax.set_title(mineral)
        ax.grid(zorder=0)
        ax.tick_params(axis="x", labelrotation=90)

        # ax.axvspan(-0.5, n_network - 0.5, alpha=0.07, color="steelblue", zorder=0)
        # ax.axvspan(n_network - 0.5, n_scenarios - 0.5, alpha=0.07, color="darkorange", zorder=0)
        # ax.axvline(x=n_network - 0.5, color="gray", linestyle=":", linewidth=0.8, zorder=1)

    for ax in axes[n_minerals:]:
        ax.set_visible(False)

    axs_2d = np.atleast_2d(axs)
    n_rows, n_cols = axs_2d.shape
    for i, ax in enumerate(axs_2d.flat):
        print(i)
        if not ax.get_visible():
            continue
        row = i // n_cols
        if row < n_rows - 1:
            ax.set_xlabel("")
            ax.tick_params(axis="x", labelbottom=False)

    handles, labels = [], []
    for ax in axes[:n_minerals]:
        h, l = ax.get_legend_handles_labels()
        handles.extend(h)
        labels.extend(l)

    unique_labels = {}
    for handle, label in zip(handles, labels):
        if label not in unique_labels:
            unique_labels[label] = handle

    fig.suptitle(fig_title, fontsize=14, y=0.99, x=0.4)
    fig.tight_layout(rect=[0, 0, 0.66, 0.99])

    legend_handles = list(unique_labels.values())
    legend_labels = list(unique_labels.keys())

    fig.legend(
        legend_handles,
        legend_labels,
        loc="upper left",
        bbox_to_anchor=(0.68, 0.95),
        frameon=False,
        borderaxespad=0.0,
        fontsize=9,
    )

    plt.show()


def plot_mineral_to_tech_sankey(
    df,
    minerals,
    tech_col=None,
    network_name=None,
    mode="mass",
    value_divisor=1e9,
    criticality_df=None,
    criticality_min_col=None,
    criticality_score_col="GeoPolRisk Characterization Factor [eq. Kg-Cu/Kg]",
    figsize=(900, 600),
    fmt_decimals=3,
    min_tech_share=0.10,  # NEW: minimum share threshold (10%)
    title="",
):
    """
    # Updated Sankey: supports criticality column "GeoPolRisk Characterization Factor [eq. Kg-Cu/Kg]"
    Sankey of minerals -> technologies.

    Important assumptions:
      - `df[minerals]` are absolute masses IN KILOGRAMS (kg).
      - For mass mode, values plotted = mass in Mt (kg / 1e9).
      - For criticality mode, values plotted = mass_kg * criticality_score (uses criticality_score_col).
        The criticality score column you specified ("GeoPolRisk Characterization Factor [eq. Kg-Cu/Kg]")
        should have units like Kg-Cu/Kg; multiplying by kg yields units like Kg-Cu.

    Parameters are:
      - df, minerals, tech_col, network_name: as before
      - mode: 'mass' or 'criticality'
      - value_divisor: for mass mode convert kg -> Mt by dividing by 1e9
      - criticality_df: DataFrame mapping minerals -> score (required for mode='criticality')
      - criticality_min_col: override mineral name column in criticality_df (optional)
      - criticality_score_col: override score column name (defaults to the provided GeoPolRisk column)
      - min_tech_share: minimum share (0-1) for technology to appear individually; smaller ones grouped as "Other" (default 0.10)
    """
    # 1) detect tech_col
    candidates = [tech_col] if tech_col else []
    candidates += [
        "carrier",
        "PyPSA technology",
        "lci_technology",
        "LCI name",
        "LCI_name",
        "Technology",
    ]
    candidates = [c for c in candidates if c is not None]
    tech_col_found = None
    for c in candidates:
        if c in df.columns:
            tech_col_found = c
            break
    if tech_col_found is None:
        raise ValueError(
            f"Couldn't find a technology column in df. Tried: {candidates}. Provide tech_col explicitly."
        )
    tech_col = tech_col_found

    # 2) aggregate masses per tech (keep both kg and Mt)
    pivot_kg = (
        df.groupby(tech_col)[minerals].sum().T
    )  # index: minerals, columns: techs; values in kg

    # 3) long-format base links (use kg and Mt)
    links = (
        pivot_kg.reset_index()
        .melt(id_vars="index", var_name=tech_col, value_name="mass_kg")
        .rename(columns={"index": "mineral"})
    )
    links["mass_Mt"] = links["mass_kg"] / value_divisor
    links = links[links["mass_kg"] > 0].copy()
    if links.empty:
        raise ValueError(
            "No positive mineral->tech mass links found. Check data and minerals list."
        )

    # 4) mode handling
    if mode == "mass":
        links["value"] = links["mass_Mt"]
        node_unit_label = "Mt"
    elif mode == "criticality":
        if criticality_df is None:
            raise ValueError(
                "mode='criticality' requires `criticality_df` mapping minerals -> score."
            )

        # detect min & score columns (allow overrides)
        cd_cols = {c.lower(): c for c in criticality_df.columns}
        # default mineral name column candidates
        possible_min_cols = ["mineral", "name", "element", "metal"]
        min_col = criticality_min_col
        if min_col is None:
            for cand in possible_min_cols:
                if cand in cd_cols:
                    min_col = cd_cols[cand]
                    break
        if min_col is None:
            # fallback: try to find a column whose values match any mineral name (case-insensitive)
            for c in criticality_df.columns:
                if (
                    criticality_df[c]
                    .astype(str)
                    .str.contains(str(minerals[0]), case=False, na=False)
                    .any()
                ):
                    min_col = c
                    break
        # score column detection: prefer the exact provided name if present
        score_col = (
            criticality_score_col
            if criticality_score_col in criticality_df.columns
            else None
        )
        if score_col is None:
            possible_score_cols = [
                "geopolrisk characteri",
                "geopolrisk",
                "geopolrisk score",
                "score",
                "criticality",
                "geopolitical",
            ]
            for cand in possible_score_cols:
                for c in criticality_df.columns:
                    if cand in c.lower():
                        score_col = c
                        break
                if score_col:
                    break
        # final numeric fallback
        if score_col is None:
            numeric_cols = [
                c
                for c in criticality_df.columns
                if pd.api.types.is_numeric_dtype(criticality_df[c])
            ]
            score_col = numeric_cols[0] if numeric_cols else None

        if min_col is None or score_col is None:
            raise ValueError(
                f"Couldn't detect mineral name column or score column in criticality_df. "
                f"Columns available: {list(criticality_df.columns)}. Provide `criticality_min_col` and/or `criticality_score_col`."
            )

        # build mapping (case-insensitive)
        crit_map = {
            str(k).strip().lower(): v
            for k, v in zip(
                criticality_df[min_col].astype(str), criticality_df[score_col]
            )
        }
        # print(crit_map)
        links["score"] = links["mineral"].map(
            lambda x: crit_map.get(str(x).strip().lower(), float("nan"))
        )
        print(links)

        missing = links["score"].isna().sum()
        if missing > 0:
            print(f"Warning: {missing} links missing a score; filling with 0.0")
            links["score"] = links["score"].fillna(0.0)

        # criticality flow = mass_kg * score (units: score-units × kg, e.g., Kg-Cu)
        links["value"] = links["mass_kg"] * links["score"]
        node_unit_label = f"{score_col}×kg"
    else:
        raise ValueError("mode must be 'mass' or 'criticality'")

    # 5) GROUP SMALL TECHNOLOGIES INTO "OTHER" FOR EACH MINERAL
    # Calculate total value per mineral
    mineral_totals = links.groupby("mineral")["value"].sum()

    # For each mineral, identify techs below threshold
    links["tech_original"] = links[tech_col]  # Keep original tech name

    for mineral in links["mineral"].unique():
        mineral_links = links[links["mineral"] == mineral]
        mineral_total = mineral_totals[mineral]

        # Calculate share for each tech
        tech_shares = mineral_links.groupby(tech_col)["value"].sum() / mineral_total

        # Identify techs below threshold
        small_techs = tech_shares[tech_shares < min_tech_share].index.tolist()

        # Replace small tech names with "Other"
        links.loc[
            (links["mineral"] == mineral) & (links[tech_col].isin(small_techs)),
            tech_col,
        ] = "Other"

    # Reaggregate after grouping
    links = (
        links.groupby(["mineral", tech_col])
        .agg({"mass_kg": "sum", "mass_Mt": "sum", "value": "sum"})
        .reset_index()
    )

    # 6) nodes & indices
    minerals_list = list(pivot_kg.index)
    techs_list = list(links[tech_col].unique())  # Updated to use grouped techs
    nodes = minerals_list + techs_list
    mineral_to_idx = {m: i for i, m in enumerate(minerals_list)}
    tech_to_idx = {t: len(minerals_list) + i for i, t in enumerate(techs_list)}

    sources = links["mineral"].map(mineral_to_idx).tolist()
    targets = links[tech_col].map(tech_to_idx).tolist()
    values = links["value"].tolist()

    # 7) node totals (consistent with mode)
    if mode == "mass":
        mineral_totals = (
            links.groupby("mineral")["mass_Mt"].sum().reindex(minerals_list).fillna(0)
        )
        tech_totals = (
            links.groupby(tech_col)["mass_Mt"].sum().reindex(techs_list).fillna(0)
        )
        def node_label_fmt(x):
            return f"{x:,.{fmt_decimals}f} Mt"
    else:
        mineral_totals = (
            links.groupby("mineral")["value"].sum().reindex(minerals_list).fillna(0)
        )
        tech_totals = (
            links.groupby(tech_col)["value"].sum().reindex(techs_list).fillna(0)
        )
        def node_label_fmt(x):
            return f"{x:,.{fmt_decimals}f} {node_unit_label}"

    node_labels = [
        f"{m}\n{node_label_fmt(mineral_totals.loc[m])}" for m in minerals_list
    ] + [f"{t}\n{node_label_fmt(tech_totals.loc[t])}" for t in techs_list]
    node_custom = list(mineral_totals.values) + list(tech_totals.values)

    # link hover text: show mass & criticality info if available
    if mode == "mass":
        link_hover = [
            f"{links.iloc[i]['mineral']} → {links.iloc[i][tech_col]}: {links.iloc[i]['mass_Mt']:.{fmt_decimals}f} Mt"
            for i in range(len(links))
        ]
    else:
        # include mass_kg and score and final value
        link_hover = [
            (
                f"{links.iloc[i]['mineral']} → {links.iloc[i][tech_col]}: "
                f"{links.iloc[i]['mass_kg']:,} kg × {links.iloc[i].get('score', 'N/A'):.{fmt_decimals}f} = "
                f"{links.iloc[i]['value']:.{fmt_decimals}f} ({node_unit_label})"
            )
            for i in range(len(links))
        ]

    # 8) render with plotly
    try:
        import plotly.graph_objects as go

        node = dict(label=node_labels, customdata=node_custom, pad=15, thickness=18)
        link = dict(source=sources, target=targets, value=values, customdata=link_hover)
        fig = go.Figure(go.Sankey(node=node, link=link))
        fig.update_layout(
            title=title,
            font=dict(
                family="Georgia, serif, Arial, sans-serif",
                size=15,
            ),
            width=figsize[0],
            height=figsize[1],
        )
        display(fig)
    except Exception as e:
        print(
            "Plotly not available or sankey rendering failed. Install plotly (`pip install plotly`) to see interactive sankey."
        )
        print("Error:", e)
        display(links.sort_values("value", ascending=False).reset_index(drop=True))
        node_summary = pd.DataFrame({"node": nodes, "total": node_custom})
        display(node_summary)


def rename_techs_costs(tech):
    tech = rename_techs(tech)
    if "heat pump" in tech or "resistive heater" in tech:
        return "heat pumps"
    elif "solar" in tech:
        return "solar"
    elif tech in ["H2 Electrolysis"]:  # , "H2 liquefaction"]:
        return "power-to-hydrogen"
    elif tech == "H2":
        return "H2 storage"
    elif tech in ["OCGT", "CHP", "gas boiler", "H2 Fuel Cell"]:
        return "gas-to-power/heat"
    # elif "solar" in tech:
    #    return "solar"
    elif tech in ["Fischer-Tropsch", "methanolisation"]:
        return "power-to-liquid"
    elif "offshore wind" in tech:
        return "offshore wind"
    elif "SMR" in tech:
        return tech.replace("SMR", "steam methane reforming")
    elif "DAC" in tech:
        return "direct air capture"
    elif "CC" in tech or "sequestration" in tech:
        return "carbon capture"
    elif tech == "oil" or tech == "gas":
        return "fossil oil and gas"
    elif tech in [
        "HVC to air",
        "agriculture machinery oil",
        "naphtha for industry",
        "non-sequestered HVC",
        "co2",
        "process emissions",
        "solid biomass for industry",
        "gas for industry",
        "industry methanol",
        "solar thermal",
        "oil refining",
        "kerosene for aviation",
        "shipping methanol",
        "H2 Store",
        "H2 pipeline",
        "methanol",
        "BEV charger",
        "heat vent",
    ]:
        return "other"
    else:
        return tech


def plot_system_cost_compare(
    n,
    network_keys,
    cost_divisor=1e9,
    figsize=(8, 6),
    method="system cost",
    sort_components=True,
    annotate_thresh=0.03,
):
    """
    Compare system_cost() breakdown across networks as stacked bars.

    - n: dict of pypsa.Network objects (your `n` variable)
    - network_keys: list of keys in `n` to compare, e.g. ['sample_new_2h_2050', 'sample_low_wind']
    - cost_divisor: divide raw costs by this for display (default 1e6 => M€ if costs are in EUR)
    - annotate_thresh: fraction of bar height above which segment values are annotated (0..1)
    Returns (ax, df_display) where df_display is the DataFrame shown (components x networks).
    """
    # 1) collect series
    results = {}
    for k in network_keys:
        if method == "system cost":
            sc = n[k].statistics.system_cost(nice_names=False)
        # ensure pandas Series
        # if isinstance(sc, pd.DataFrame):
        #     # try to pick a numeric column if returned as single-row DF
        #     # sc = n[sample_base_2h].statistics.system_cost().droplevel(level=0)
        #     sc = sc.droplevel(level=0).groupby(rename_techs_tyndp).sum()
        #     print(sc)
        elif method == "expanded capex":
            sc = n[k].statistics.expanded_capex(nice_names=False)
        elif method == "opex":
            sc = n[k].statistics.opex(nice_names=False)
        elif method == "expanded":
            capex = n[k].statistics.expanded_capex(nice_names=False)
            opex = n[k].statistics.opex(nice_names=False)

            sc = capex.add(opex, fill_value=0)

        sc = sc.droplevel(level=0).groupby(rename_techs_costs).sum()
        results[k] = sc

    # 2) build DataFrame (components x networks), align missing components -> 0
    df = pd.DataFrame(results).fillna(0)
    if sort_components:
        df = df.loc[df.sum(axis=1).sort_values(ascending=False).index]

    # 3) convert units for display
    df_display = df / cost_divisor

    # 4) plot stacked bars (one bar per network)
    # Map each (renamed) component to a colour, falling back to grey for any
    # label missing from tech_colors (some cost groups such as
    # "power-to-hydrogen" / "carbon capture" have no entry).
    colors = [tech_colors.get(c, "#888888") for c in df_display.index]
    ax = df_display.T.plot.bar(stacked=True, figsize=figsize, color=colors, zorder=2)
    ax.set_ylabel("System cost (billion Euros)")
    ax.set_xlabel("")
    ax.set_title("System cost breakdown by technology class scenario, billion Euros")
    ax.legend(title="Component", bbox_to_anchor=(1.02, 1), loc="upper left")
    # Set y-axis top to 5% above the tallest stacked bar.
    max_bar_value = df_display.sum(axis=0).max()
    y_min, _ = ax.get_ylim()
    ax.set_ylim(y_min, max_bar_value * 1.05)

    # 5) annotate segment values (absolute) when they exceed annotate_thresh fraction of that network bar
    # data for annotation: df_display.T (index = networks, columns = components)
    data = df_display.T
    networks = list(data.index)
    components = list(data.columns)

    for i, net in enumerate(networks):
        cumulative = 0.0
        total = data.loc[net].sum()
        for comp in components:
            val = data.loc[net, comp]
            if val <= 0:
                cumulative += val
                continue
            # annotate if value/total >= threshold (or if total is 0, skip)
            if total > 0 and (val / total) >= annotate_thresh:
                ax.text(
                    i,
                    cumulative + val / 2,
                    f"{val:,.1f}",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="black",
                )
            cumulative += val

    # plt.tight_layout()
    # ax.grid(zorder=0)
    return ax, df_display


def rename_techs(label: str) -> str:
    """
    Rename technology labels for better readability.

    Removes some prefixes and renames if certain conditions defined in function body are met.

    Parameters
    ----------
    label: str
        Technology label to be renamed

    Returns
    -------
    str
        Renamed label
    """
    prefix_to_remove = [
        "residential ",
        "services ",
        "urban ",
        "rural ",
        "central ",
        "decentral ",
    ]

    rename_if_contains = [
        "CHP",
        "gas boiler",
        "biogas",
        "solar thermal",
        "air heat pump",
        "ground heat pump",
        "resistive heater",
        "Fischer-Tropsch",
    ]

    rename_if_contains_dict = {
        "water tanks": "hot water storage",
        "retrofitting": "building retrofitting",
        # "H2 Electrolysis": "hydrogen storage",
        # "H2 Fuel Cell": "hydrogen storage",
        # "H2 pipeline": "hydrogen storage",
        "battery": "battery storage",
        "H2 for industry": "H2 for industry",
        "land transport fuel cell": "land transport fuel cell",
        "land transport oil": "land transport oil",
        "oil shipping": "shipping oil",
        # "CC": "CC"
    }

    rename = {
        "solar": "solar PV",
        "Sabatier": "methanation",
        "offwind": "offshore wind",
        "offwind-ac": "offshore wind (AC)",
        "offwind-dc": "offshore wind (DC)",
        "offwind-float": "offshore wind (Float)",
        "onwind": "onshore wind",
        "ror": "hydroelectricity",
        "hydro": "hydroelectricity",
        "PHS": "hydroelectricity",
        "NH3": "ammonia",
        "co2 Store": "DAC",
        "co2 stored": "CO2 sequestration",
        "AC": "transmission lines",
        "DC": "transmission lines",
        "B2B": "transmission lines",
    }

    for ptr in prefix_to_remove:
        if label[: len(ptr)] == ptr:
            label = label[len(ptr) :]

    for rif in rename_if_contains:
        if rif in label:
            label = rif

    for old, new in rename_if_contains_dict.items():
        if old in label:
            label = new

    for old, new in rename.items():
        if old == label:
            label = new
    return label


# ---------------------------------------------------------------------------
# Functions moved from the notebook
# ---------------------------------------------------------------------------


def rename_techs_balances(tech):
    tech = rename_techs(tech)
    if "heat pump" in tech:
        return "ambient heat"
    elif tech in ["H2 Electrolysis"]:  # , "H2 liquefaction"]:
        return "power-to-hydrogen"
    elif "solar" in tech:
        return "solar"
    elif tech in ["Fischer-Tropsch", "methanolisation"]:
        return "power-to-liquid"
    elif tech == "DAC":
        return "direct air capture"
    elif "offshore wind" in tech:
        return "offshore wind"
    elif tech == "oil" or tech == "gas":
        return "fossil oil and gas"
    elif tech in ["BEV charger", "V2G", "Li ion", "land transport EV"]:
        return "battery electric vehicles"
    elif tech in ["biogas", "solid biomass"]:
        return "biomass"
    elif tech in ["electricity"]:
        return "residential electricity demand"
    elif tech in ["industry electricity", "agriculture electricity"]:
        return "industry electricity demand"
    elif tech in ["agriculture heat", "heat", "low-temperature heat for industry"]:
        return "heat demand"
    elif "solid biomass for industry" in tech:
        return "biomass demand"
    elif "gas for industry" in tech:
        return "methane demand"
    elif tech in ["H2 for industry", "land transport fuel cell"]:
        return "hydrogen demand"
    elif tech in [
        "kerosene for aviation",
        "naphtha for industry",
        "shipping methanol",
        "agriculture machinery oil",
    ]:
        return "liquid hydrocarbon demand"
    elif tech in [
        "transmission lines",
        "H2 pipeline",
        "H2 pipeline retrofitted",
        "H2",
        "electricity distribution grid",
        "SMR",
        "SMR CC",
        "OCGT",
        "CHP",
        "gas boiler",
        "H2 Fuel Cell",
        "resistive heater",
        "battery storage",
        "methanation",
    ]:
        return "other"
    else:
        return tech


def plot_stacked_capacity(
    data,
    ax,
    scenarios,
    tech_colors=None,
    nice_names=None,
    combine_heat_pumps=True,
    generation_only=False,
    storage_only=False,
    p_nom_col="p_nom",
    p_nom_min_col="p_nom_min",
    title="",
):
    """
    Plot one horizontal stacked bar per scenario and hatch the already-installed share.

    All capacities are plotted as positive magnitudes.
    If generation_only=True, only electricity generation carriers are shown
    (from Generators and generation Links such as CCGT/OCGT).
    """
    if tech_colors is None:
        tech_colors = DEFAULT_TECH_COLORS
    if nice_names is None:
        nice_names = DEFAULT_NICE_NAMES

    plot_df = data.copy()

    if generation_only:
        if isinstance(plot_df.index, pd.MultiIndex) and plot_df.index.nlevels >= 2:
            component = plot_df.index.get_level_values(0).astype(str).str.lower()
            carrier = plot_df.index.get_level_values(1).astype(str).str.lower()

            component_mask = component.isin(["generator", "link"])
            generation_carrier_mask = carrier.str.contains(
                r"onwind|offwind|solar|nuclear|run of river|ror|combined-cycle gas|ccgt|ocgt"
            )
            plot_df = plot_df[component_mask & generation_carrier_mask]
        else:
            carrier = pd.Index(plot_df.index).astype(str).str.lower()
            generation_carrier_mask = carrier.str.contains(
                r"onwind|offwind|solar|nuclear|run of river|ror|combined-cycle gas|ccgt|ocgt"
            )
            plot_df = plot_df[generation_carrier_mask]

    if storage_only:
        if isinstance(plot_df.index, pd.MultiIndex) and plot_df.index.nlevels >= 2:
            component = plot_df.index.get_level_values(0).astype(str).str.lower()
            print(component)
            carrier = plot_df.index.get_level_values(1).astype(str).str.lower()

            component_mask = component.isin(["storageunit", "store"])
            storage_carrier_mask = carrier.str.contains(
                r"^(?!.*EV battery).*?(battery|PHS|hydro|phs)", case=False, regex=True
            )
            plot_df = plot_df[component_mask & storage_carrier_mask]
        else:
            carrier = pd.Index(plot_df.index).astype(str).str.lower()
            storage_carrier_mask = carrier.str.contains(
                r"^(?!.*EV battery).*?(battery|PHS|hydro|phs)", case=False, regex=True
            )
            plot_df = plot_df[storage_carrier_mask]

    if combine_heat_pumps:
        hp_mask = plot_df.index.get_level_values(1).str.contains(
            "heat pump", case=False
        )
        if hp_mask.any():
            hp_data = plot_df.loc[hp_mask].sum()
            hp_data.name = ("Load", "heat pump")
            plot_df = pd.concat([plot_df.loc[~hp_mask], hp_data.to_frame().T])
            tech_colors.setdefault("heat pump", "#7a7a7a")
            nice_names.setdefault("heat pump", "heat pump")

    if plot_df.empty:
        raise ValueError("No technologies matched the selected filters for plotting.")

    technologies = plot_df.index.get_level_values(1).unique()

    p_nom = plot_df.loc[:, (scenarios, p_nom_col)].copy()
    p_nom.columns = p_nom.columns.droplevel(1)

    p_nom_min = plot_df.loc[:, (scenarios, p_nom_min_col)].copy()
    p_nom_min.columns = p_nom_min.columns.droplevel(1)

    y = np.arange(len(scenarios))
    bar_height = 0.42
    left = np.zeros(len(scenarios), dtype=float)
    scenario_totals = np.abs(p_nom.reindex(columns=scenarios).values).sum(axis=0)
    hatch_color = (0.3, 0.3, 0.3, 0.8)

    legend_handles = []
    seen_labels = set()

    for i, tech in enumerate(technologies):
        tech_mask = plot_df.index.get_level_values(1) == tech
        total_cap = p_nom.loc[tech_mask].sum().reindex(scenarios).fillna(0.0)
        existing_cap = p_nom_min.loc[tech_mask].sum().reindex(scenarios).fillna(0.0)

        # Force positive plotting so all capacities appear on the same side.
        total_vals = np.abs(total_cap.values.astype(float))
        existing_vals = np.abs(existing_cap.values.astype(float))
        existing_vals = np.clip(existing_vals, 0.0, total_vals)

        color = tech_colors.get(tech, "#888888")
        label = nice_names.get(tech, tech)

        # Full segment (total capacity)
        ax.barh(
            y,
            total_vals,
            bar_height,
            left=left,
            color=color,
            edgecolor="white",
            linewidth=0.5,
        )

        # Add labels only for segments above 10% of scenario total capacity
        for j, val in enumerate(total_vals):
            total_for_scenario = scenario_totals[j]
            share = val / total_for_scenario if total_for_scenario > 0 else 0.0
            if val > 0 and share > 0.10:
                ax.text(
                    left[j] + val / 2,  # x-position (center of segment)
                    y[j],  # y-position
                    f"{val:.1f} GW",
                    va="center",
                    ha="center",  # center text inside bar
                    fontsize=8,
                    fontweight="bold",
                    color="black",  # adjust for contrast if needed
                )

        # Overlay hatched installed share within each segment

        if i % 2 == 0:
            ax.barh(
                y,
                existing_vals,
                bar_height,
                left=left,
                facecolor="none",
                edgecolor=hatch_color,
                hatch="/////",
                linewidth=0.0,
            )
        else:
            ax.barh(
                y,
                existing_vals,
                bar_height,
                left=left,
                facecolor="none",
                edgecolor=hatch_color,
                hatch="////",
                linewidth=0.0,
            )

        left += total_vals

        # Add vertical lines between segments
        for i in range(len(scenarios)):
            if left[i] > 0:
                ax.vlines(
                    left[i],
                    y[i] - bar_height / 2,
                    y[i] + bar_height / 2,
                    color="black",
                    linewidth=0.5,
                )

        if label not in seen_labels:
            legend_handles.append(mpatches.Patch(facecolor=color, label=label))
            seen_labels.add(label)

    if generation_only:
        ax.set_xlabel("Installed capacity [GW]")
    if storage_only:
        ax.set_xlabel("Installed capacity [GWh]")
    ax.set_title(title)
    ax.set_yticks(y)
    ax.set_yticklabels(scenarios)

    tech_legend = ax.legend(
        handles=legend_handles,
        title="Technology",
        bbox_to_anchor=(1.02, 1.0),
        loc="upper left",
    )
    ax.add_artist(tech_legend)

    hatch_legend = [
        mpatches.Patch(facecolor="lightgray", label="total capacity"),
        mpatches.Patch(
            facecolor="white",
            edgecolor=hatch_color,
            hatch="/////",
            label="already installed",
        ),
    ]
    ax.legend(
        handles=hatch_legend,
        title="Segment meaning",
        bbox_to_anchor=(1.02, 0.4),
        loc="upper left",
    )


def build_capacity_bounds_df(
    stats, runs, *, storage=False, bus_carrier=None, scale=1e3
):
    """Build DataFrame with columns (scenario, {p_nom_min, p_nom}) from PyPSA stats."""
    frames = {}
    for run in runs:
        installed = stats[run].installed_capacity(
            drop_zero=False,
            nice_names=False,
            round=3,
            groupby="carrier",
            storage=storage,
            bus_carrier=bus_carrier,
        )
        optimal = stats[run].optimal_capacity(
            drop_zero=False,
            nice_names=False,
            round=3,
            groupby="carrier",
            storage=storage,
            bus_carrier=bus_carrier,
        )
        frames[run] = pd.concat({"p_nom_min": installed, "p_nom": optimal}, axis=1)

    out = pd.concat(frames, axis=1)
    return out / scale


def plot_mineral_reserve_shares(
    scenario_defs,
    n,
    minerals,
    mineral_intensities,
    tech_scenarios,
    non_energy_demand,
    grid_demand,
    mineral_avail,
    mineral_alloc_factors,
    reserve_method="Reserves",
    reserve_allocation_key="GDP share",
    non_energy_year="Cumulative",
    non_energy_col="GDP-adjusted value entso-e (Mt)",
    figsize=(15, 6),
    marker_map=None,
    color_map=None,
    title=None,
):
    """Plot non-energy + energy mineral demand as a share of reserves across scenarios."""

    # --- Base data ---
    reserves = mineral_avail[mineral_avail["Estimate method"] == reserve_method].copy()
    reserves = reserves[reserves["Mineral"].isin(minerals)].copy()

    non_energy = non_energy_demand[non_energy_demand["Year"] == non_energy_year][
        ["Mineral", non_energy_col]
    ].rename(columns={non_energy_col: "Non-energy (Mt)"})

    merged = pd.merge(reserves, non_energy, on="Mineral", how="inner")
    # merged["Allocated reserves (Mt)"] = (
    #     merged["Value (Mt)"] * mineral_alloc_factors[reserve_allocation_key]
    # )
    merged["Non-energy share"] = merged["Non-energy (Mt)"] / merged["Value (Mt)"]

    # Optional cumulative grid demand term (expected to be non-zero for Copper)
    grid_extra_Mt = pd.Series(0.0, index=merged["Mineral"])
    if (
        grid_demand is not None
        and not grid_demand.empty
        and {"Mineral", "Year", "GDP-adjusted value entso-e (Mt)"}.issubset(
            grid_demand.columns
        )
    ):
        grid_cumulative = grid_demand[grid_demand["Year"] == "Cumulative"].copy()
        grid_extra_Mt = (
            grid_cumulative.groupby("Mineral")["GDP-adjusted value entso-e (Mt)"]
            .sum()
            .reindex(merged["Mineral"])
            .fillna(0.0)
        )

    if merged["Value (Mt)"].isna().any():
        raise ValueError(
            "Some minerals have missing allocated reserves. Check reserve_method/alloc factor."
        )

    # Compute energy demand for each scenario and stack on top of non-energy
    results = []
    for label, network_key, scenario in scenario_defs:
        _, totals_Mt = compute_mineral_usage(
            network_key,
            n,
            all_minerals,
            scenario,
            mineral_intensities,
            tech_scenarios,
        )

        totals_Mt.loc["REE"] = totals_Mt.loc[
            ["Praseodymium", "Neodymium", "Dysprosium"]
        ].sum()

        totals_Mt = totals_Mt.reindex(merged["Mineral"]).fillna(0)
        merged[f"Energy (Mt) - {label}"] = totals_Mt.values

        # Add cumulative grid demand before normalizing by reserves
        merged[f"Energy share - {label}"] = (
            merged[f"Energy (Mt) - {label}"] + grid_extra_Mt.values
        ) / merged["Value (Mt)"]

        merged[f"Total share - {label}"] = (
            merged["Non-energy share"] + merged[f"Energy share - {label}"]
        )
        results.append(label)

    # --- Plot ---
    fig, ax = plt.subplots(figsize=figsize)
    minerals_order = [m for m in minerals if m in merged["Mineral"].values]
    x = np.arange(len(minerals_order))
    merged = merged.set_index("Mineral").reindex(minerals_order).reset_index()

    # # Plot non-energy baseline as a square marker
    ax.scatter(
        x,
        merged["Non-energy share"],
        marker="s",
        s=80,
        color="black",
        label="Non-energy share",
        zorder=3,
    )

    # Plot each scenario on top
    if marker_map is None:
        base_markers = ["o", "^", "D", "v", "P", "X", "*"]
        marker_map = {
            label: base_markers[i % len(base_markers)]
            for i, label in enumerate(results)
        }
    if color_map is None:
        cmap = plt.get_cmap("tab10")
        color_map = {label: cmap(i) for i, label in enumerate(results)}

    for label in results:
        ys = merged[f"Total share - {label}"]
        ax.scatter(
            x,
            ys,
            marker=marker_map[label],
            s=80,
            color=color_map[label],
            label=label,
            zorder=4,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(minerals_order, rotation=90)
    ax.set_ylabel("Fraction of total reserves")
    ax.set_xlabel("Metal")
    ax.set_title(
        title
        or "Mineral demand (non-energy + energy) as share of GDP-allocated reserves"
    )
    ax.grid(axis="y", linestyle="--", alpha=0.3)

    # add lines for gdp, per capita, and per capita adjusted for energy

    ax.axhline(
        y=mineral_alloc_factors["GDP share"],
        color="blue",
        linestyle="--",
        linewidth=2,
        label="GDP-allocated reserves",
        zorder=2,
    )
    ax.axhline(
        y=mineral_alloc_factors["Per capita share"],
        color="orange",
        linestyle="--",
        linewidth=2,
        label="Per-capita-allocated reserves",
        zorder=2,
    )
    ax.axhline(
        y=mineral_alloc_factors["Per capita share corrected for energy use"],
        color="green",
        linestyle="--",
        linewidth=2,
        label="Per-capita allocated reserves (energy corrected)",
        zorder=2,
    )

    ax.legend(loc="upper right", bbox_to_anchor=(1.8, 1))
    plt.tight_layout()

    return merged
