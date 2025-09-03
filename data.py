#-----------------data.py-----------------
import yaml
import numpy as np
import pandas as pd
from typing import Optional, Tuple  # add this


def preprocess_file(file_path, sequence_length=1000):
    """Preprocess a single gait file and return 1-channel segments shaped (1, L)."""
    try:
        # Load tab-separated file with no header
        data = pd.read_csv(file_path, sep='\t', header=None)

        # Select only Column 18 (0-based index 17). Keep it 2D (T, 1)
        col = data.iloc[:, [17]].to_numpy()

        T = col.shape[0]
        segments = []

        if T >= sequence_length:
            # Non-overlapping segments (step = sequence_length). Change to sequence_length//2 for 50% overlap.
            step_size = sequence_length
            for i in range(0, T - sequence_length + 1, step_size):
                seg = col[i:i + sequence_length, :]  # (L, 1)
                seg = seg.T                           # (1, L)  channels-first
                segments.append(seg)
        else:
            # Pad to required length
            pad = np.zeros((sequence_length - T, 1), dtype=col.dtype)
            seg = np.vstack([col, pad]).T           # (1, L)
            segments.append(seg)

        return segments

    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

# Load patients data from YAML
def load_data():
    with open("data.yaml") as f:
        return yaml.safe_load(f)


def get_units(param_type: str) -> str:
    """Return appropriate units for each parameter type."""
    units_dict = {
        "STRIDE TIME": "s",
        "STANCE %": "%",
        "SWING %": "%",
        # legacy labels (still supported if used elsewhere)
        "STANCE TIME": "s",
        "SWING TIME": "s",
    }
    return units_dict.get(param_type, "")


def render_gait_parameter(
    value: Optional[float],
    param_type: str,
    min_val: float,
    max_val: float,
    threshold: Optional[float] = None,
    is_higher_better: bool = True,
    band: Optional[Tuple[float, float]] = None
) -> str:
    """
    Create HTML visualization for gait parameters (gauge bar).

    If `band=(low, high)` is provided, we color green when value is inside the band
    and red otherwise (preferred for stance/swing %). If `band` is None, we
    fall back to single-threshold coloring using `threshold` + `is_higher_better`.
    If `value` is None, draw neutral bar with markers but no value.
    """
    units = get_units(param_type)

    def _pct(x: float) -> float:
        return min(max(((x - min_val) / (max_val - min_val) * 100), 0), 100)

    green_color = "#88c9bf"
    red_color   = "#d77c7c"
    neutral_bar = "#bbbbbb"

    if value is None:
        percent = 0
        bar_color = neutral_bar
        display_val = "--"
    else:
        percent = _pct(value)
        if band is not None:
            low, high = band
            inside = (value >= low) and (value <= high)
            bar_color = green_color if inside else red_color
        else:
            if threshold is None:
                bar_color = neutral_bar
            else:
                thr_pct = _pct(threshold)
                bar_color = (green_color if percent >= thr_pct else red_color) if is_higher_better \
                            else (red_color if percent >= thr_pct else green_color)
        display_val = f"{value:.3f}"

    if band is not None:
        low, high = band
        low_pct  = _pct(low)
        high_pct = _pct(high)
        thr_markup = f"""
        <div style="position:absolute; left:{(low_pct+high_pct)/2}%; top:-30px; transform:translateX(-50%); font-size:12px; color:#000;">
          {low}–{high} {units}
        </div>
        <div style="position:absolute; left:{low_pct}%; top:-6px; width:2px; height:18px; background:#000;"></div>
        <div style="position:absolute; left:{high_pct}%; top:-6px; width:2px; height:18px; background:#000;"></div>
        """
    else:
        thr_markup = "" if threshold is None else f"""
        <div style="position:absolute; left:{_pct(threshold)}%; top:-30px; transform:translateX(-50%); font-size:12px; color:#000;">
          {threshold} {units}
        </div>
        <div style="position:absolute; left:{_pct(threshold)}%; top:-6px; width:2px; height:18px; background:#000;"></div>
        """

    value_marker = "" if value is None else f'''
        <div style="position:absolute; left:{percent}%; top:-1px; width:2px; height:7px; background:#1976d2; border-radius:1px;"></div>
    '''

    html = f"""
    <div class="gait-param-box">
      <div style="font-size:15px; color:#1976d2; margin-left:5px; margin-bottom:8px; font-weight:700;">
        {param_type}
      </div>
      <div style="display:flex; justify-content:space-between; font-size:12px; color:#1976d2;">
        <span>{min_val} {units}</span>
        <span>{max_val} {units}</span>
      </div>
      <div style="position:relative; height:7px; margin:22px 0 14px 0; background:#f0f0f0; border:1px solid {bar_color}; border-radius:6px;">
        {thr_markup}
        <div style="position:absolute; left:0; width:{percent}%; height:100%; background:{bar_color}; border-radius:6px 0 0 6px;"></div>
        {value_marker}
      </div>
      <div style="font-size:15px; color:{bar_color}; text-align:center; font-weight:bold;">
        {display_val} {units}
      </div>
    </div>
    """
    return html
