#-------------------app.py-------------------

import gradio as gr
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import io, base64

# Import your existing modules
from config import PRETRAINED_MODEL_PATH, SEGMENT_LENGTH, CLASS_NAMES, css, js_func
from data import load_data, preprocess_file, render_gait_parameter
from model import ParkinsonsGaitCNN
from discrepancy import XAIComparativeAnalyzer
from gait_event import GaitMetricsCalculator
from chatbox import ParkinsonsGaitChatbot

import warnings

# Suppress all FutureWarnings
warnings.filterwarnings("ignore", category=FutureWarning)

# (optional) Suppress only the PyTorch hook warning instead of all:
# warnings.filterwarnings("ignore", category=FutureWarning, message=".*non-full backward hook.*")


class ParkinsonsGaitApp:
    """Main application class for Parkinson's Gait Analysis"""
    
    def __init__(self):
        """Initialize the application"""
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load model
        self.model = ParkinsonsGaitCNN(input_channels=1, sequence_length=SEGMENT_LENGTH)
        self.model.load_state_dict(torch.load(PRETRAINED_MODEL_PATH, map_location=self.device))
        self.model.to(self.device).eval()
        
        # Initialize XAI analyzer
        self.xai_analyzer = XAIComparativeAnalyzer(PRETRAINED_MODEL_PATH)
        
        # Initialize chatbot
        self.chatbot = ParkinsonsGaitChatbot()
        
        # Load patient data
        self.patient_data = load_data()
        self.patient_names = list(self.patient_data.keys())
        
        # Store current analysis results
        self.current_results = {}
        self.single_sensor_name = "Total Left Force"
        self.single_sensor_idx = 17  # Column index for single sensor analysis (0-based)
    
    def get_segment_count(self, patient_name: str) -> int:
        """Return how many 1000-sample segments are available for a patient."""
        if not patient_name:
            return 0
        try:
            patient_info = self.patient_data[patient_name]
            segments = preprocess_file(patient_info['gait_file'], SEGMENT_LENGTH)
            return len(segments) if segments else 0
        except Exception:
            return 0

    def update_segment_slider(self, patient_name: str):
        """Gradio helper to update the segment slider max/value when patient changes."""
        count = self.get_segment_count(patient_name)
        # If no segments, keep max at 0; else max index is count-1; reset value to 0
        max_idx = max(0, count - 1)
        return gr.update(minimum=0, maximum=max_idx, step=1, value=0, interactive=True)


    def plot_raw_gait_data(self, patient_name: str, segment_idx: int = 0):
        """Plot raw gait data for selected patient and sensor, and segment idx"""
        if not patient_name:
            return plt.figure()

        try:
            # Build segments using the same preprocessing the model uses
            patient_info = self.patient_data[patient_name]
            segments = preprocess_file(patient_info['gait_file'], SEGMENT_LENGTH)

            if not segments or len(segments) == 0:
                fig, ax = plt.subplots(figsize=(16, 4))
                ax.text(0.5, 0.5, 'No valid 1000-sample segments found.',
                        ha='center', va='center', transform=ax.transAxes)
                return fig

            # Clamp segment_idx to valid range
            segment_idx = max(0, min(segment_idx, len(segments) - 1))
            seg = segments[segment_idx]  # shape: (channels, 1000)


            # Plot the chosen channel from the chosen segment
            sensor_data = seg[0, :] 
            time_data = np.arange(sensor_data.shape[0])

            fig, ax = plt.subplots(figsize=(16, 4))
            ax.plot(time_data, sensor_data, linewidth=1)
            ax.set_xlabel('Time Steps', fontsize=12)
            ax.set_ylabel('Force (N)', fontsize=12)
            ax.set_title(f'{patient_name} • {self.single_sensor_name} • Segment #{segment_idx}', fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            return fig

        except Exception as e:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.text(0.5, 0.5, f'Error loading data: {str(e)}',
                    ha='center', va='center', transform=ax.transAxes)
            return fig


    def predict_severity(self, patient_name: str, segment_idx: int = 0):
        """Predict Parkinson's severity for selected patient using the chosen segment."""
        if not patient_name:
            return "Please select a patient"

        try:
            patient_info = self.patient_data[patient_name]
            segments = preprocess_file(patient_info['gait_file'], SEGMENT_LENGTH)

            if not segments or len(segments) == 0:
                return "Error: No valid 1000-sample segments found"

            # Clamp segment index and select that segment
            segment_idx = max(0, min(segment_idx, len(segments) - 1))
            segment = torch.FloatTensor(segments[segment_idx]).unsqueeze(0).to(self.device)

            with torch.no_grad():
                output = self.model(segment)
                probabilities = F.softmax(output, dim=1)
                predicted_class = output.argmax(dim=1).item()
                confidence = probabilities[0, predicted_class].item()

            result = f"{CLASS_NAMES[predicted_class]} (Confidence: {confidence:.3f}) • Segment #{segment_idx}"

            # Store for chatbot context
            self.current_results['prediction'] = {
                'class': CLASS_NAMES[predicted_class],
                'confidence': confidence,
                'patient': patient_name,
                'segment_idx': segment_idx
            }

            return result

        except Exception as e:
            return f"Error in prediction: {str(e)}"

    def compute_gait_metrics(self, patient_name: str, segment_idx: int = 0):
        """
        Compute gait event metrics for the selected patient and the currently
        selected 1000-sample segment (LEFT FOOT ONLY).

        Now also computes stance% and swing% (as % of stride) using the mean values,
        so it aligns with the gauge logic used in compute_gait_param_boxes().
        """
        if not patient_name:
            return "Please select a patient"

        try:
            patient_info = self.patient_data[patient_name]
            calculator = GaitMetricsCalculator(patient_info['gait_file'])

            # Slice to the selected 1000-sample window
            L = SEGMENT_LENGTH  # 1000
            N = len(calculator.data)
            start = int(segment_idx) * L
            end = start + L

            if start >= N:
                return (f"No data for segment #{segment_idx}: file has only {N} samples "
                        f"({N // L} full segments).")

            end = min(end, N)
            calculator.data = calculator.data.iloc[start:end].reset_index(drop=True)

            # --- LEFT FOOT ONLY ---
            calculator.detect_heel_strikes_toe_offs(foot='left', force_threshold=50)
            left_stride_times = calculator.calculate_stride_time('left')  # seconds
            left_stance_times = calculator.calculate_stance_time('left')  # seconds (in-contact)
            left_swing_times  = calculator.calculate_swing_time('left')   # seconds (in-air)

            # Helper for mean/std/count
            def mstats(arr):
                import numpy as np
                return (float(np.mean(arr)), float(np.std(arr)), int(len(arr))) if len(arr) else (None, None, 0)

            stride_mean, stride_std, stride_n = mstats(left_stride_times)
            stance_mean, stance_std, stance_n = mstats(left_stance_times)
            swing_mean,  swing_std,  swing_n  = mstats(left_swing_times)

            # Percent of stride (aligns with the gauge logic)
            if stride_mean and stride_mean > 0:
                stance_pct_mean = float(stance_mean / stride_mean * 100.0) if stance_mean is not None else None
                swing_pct_mean  = float(swing_mean  / stride_mean * 100.0) if swing_mean  is not None else None
            else:
                stance_pct_mean = None
                swing_pct_mean  = None

            # Store for chatbot/context (left only + window info + summary stats)
            results = {
                'segment_idx': int(segment_idx),
                'start_row': int(start),
                'end_row': int(end),
                'left_stride_times': left_stride_times,
                'left_stance_times': left_stance_times,
                'left_swing_times': left_swing_times,
                # summary stats for convenience
                'stride_mean_s': stride_mean, 'stride_std_s': stride_std, 'stride_n': stride_n,
                'stance_mean_s': stance_mean, 'stance_std_s': stance_std, 'stance_n': stance_n,
                'swing_mean_s':  swing_mean,  'swing_std_s':  swing_std,  'swing_n':  swing_n,
                'stance_pct_mean': stance_pct_mean,
                'swing_pct_mean':  swing_pct_mean,
            }
            self.current_results['gait_metrics'] = results

            # Human-readable text (now includes % of stride for stance/swing)
            def fmt(name, mean, std, n, pct=None):
                if mean is None:
                    return f"  {name}: No valid cycles detected\n"
                pct_str = f" (≈ {pct:.1f}% of stride)" if pct is not None else ""
                return f"  {name}: {mean:.3f} ± {std:.3f} sec{pct_str} ({n} cycles)\n"

            hdr = (f"=== GAIT METRICS (LEFT FOOT) for {patient_name} • Segment #{segment_idx} ===\n")
            text  = hdr
            text += fmt("Stride Time", stride_mean, stride_std, stride_n, None)
            text += fmt("Stance Time", stance_mean, stance_std, stance_n, stance_pct_mean)
            text += fmt("Swing Time",  swing_mean,  swing_std,  swing_n,  swing_pct_mean)

            return text

        except Exception as e:
            return f"Error computing gait metrics: {str(e)}"


    def _format_gait_metrics_block(self, gm: dict) -> str:
        """Compact, human-readable gait metrics for the current segment window
        including stance/swing as % of stride (to match the gauges)."""
        import numpy as np

        def stats(arr):
            if arr is None or len(arr) == 0:
                return None, None, 0
            return float(np.mean(arr)), float(np.std(arr)), int(len(arr))

        s_mean, s_std, s_n = stats(gm.get('left_stride_times', []))
        st_mean, st_std, st_n = stats(gm.get('left_stance_times', []))
        sw_mean, sw_std, sw_n = stats(gm.get('left_swing_times', []))

        stance_pct = float(st_mean / s_mean * 100.0) if s_mean and st_mean is not None else None
        swing_pct  = float(sw_mean  / s_mean * 100.0) if s_mean and sw_mean is not None else None

        lines = []
        if s_n > 0:
            lines.append(f"- Stride time (L): {s_mean:.3f} ± {s_std:.3f} s ({s_n} cycles)")
        else:
            lines.append("- Stride time (L): no valid cycles detected")

        if st_n > 0:
            if stance_pct is not None:
                lines.append(f"- Stance (L): {st_mean:.3f} ± {st_std:.3f} s (≈{stance_pct:.1f}% of stride, {st_n} cycles)")
            else:
                lines.append(f"- Stance (L): {st_mean:.3f} ± {st_std:.3f} s ({st_n} cycles)")
        else:
            lines.append("- Stance (L): no valid cycles detected")

        if sw_n > 0:
            if swing_pct is not None:
                lines.append(f"- Swing (L): {sw_mean:.3f} ± {sw_std:.3f} s (≈{swing_pct:.1f}% of stride, {sw_n} cycles)")
            else:
                lines.append(f"- Swing (L): {sw_mean:.3f} ± {sw_std:.3f} s ({sw_n} cycles)")
        else:
            lines.append("- Swing (L): no valid cycles detected")

        return "\n".join(lines)



    def compute_gait_metrics_to_state(self, patient_name: str, segment_idx: int = 0):
        """Compute metrics, store in self.current_results, and also return a dict for a Gradio State."""
        text = self.compute_gait_metrics(patient_name, segment_idx)  # this already sets self.current_results['gait_metrics']
        gm = self.current_results.get('gait_metrics', {})
        return text, gm

    def compute_gait_param_boxes(self, patient_name: str, segment_idx: int = 0):
        """
        Build three HTML gauge boxes for the current segment window (LEFT foot):
        - STRIDE TIME (s) mean  → show against a *target band* (e.g., 0.90–1.30 s as a default)
        - STANCE %   (% of stride) mean → target band ~55–62%
        - SWING %    (% of stride) mean → target band ~38–45%

        If not enough cycles are detected, we render the bars without a value.
        """
        from data import render_gait_parameter  # ensure latest version is used

        def mk_box(value, name, mn, mx, thr=None, higher_better=True, band=None):
            return render_gait_parameter(
                value=value,
                param_type=name,
                min_val=mn,
                max_val=mx,
                threshold=thr,
                is_higher_better=higher_better,
                band=band
            )

        # Defaults (empty/grey bars) with the new labels/units
        empty = (
            mk_box(None, "STRIDE TIME", 0.60, 1.80, band=(1.10, 1.30)),  # neutral band for habitual speed
            mk_box(None, "STANCE %",    0.00, 100.0, band=(58.0, 62.0)),
            mk_box(None, "SWING %",     0.00, 100.0, band=(38.0, 42.0)),
        )

        if not patient_name:
            return empty

        try:
            patient_info = self.patient_data[patient_name]
            calculator = GaitMetricsCalculator(patient_info['gait_file'])

            # Segment slicing (no padding for event detection)
            L = SEGMENT_LENGTH
            N = len(calculator.data)
            start = int(segment_idx) * L
            if start >= N:
                return empty
            end = min(start + L, N)
            calculator.data = calculator.data.iloc[start:end].reset_index(drop=True)

            # Detect events (left foot) and compute arrays
            calculator.detect_heel_strikes_toe_offs(foot='left', force_threshold=50)
            stride_times = calculator.calculate_stride_time('left')  # seconds
            stance_times = calculator.calculate_stance_time('left')  # seconds within cycle
            swing_times  = calculator.calculate_swing_time('left')   # seconds within cycle

            # Means (None if empty)
            stride_mean = float(np.mean(stride_times)) if len(stride_times) else None
            stance_mean = float(np.mean(stance_times)) if len(stance_times) else None
            swing_mean  = float(np.mean(swing_times))  if len(swing_times)  else None

            # Convert stance/swing to % of gait cycle using STRIDE mean (safer & clearer than absolute seconds)
            if stride_mean and stride_mean > 0:
                stance_pct = float(stance_mean / stride_mean * 100.0) if stance_mean is not None else None
                swing_pct  = float(swing_mean  / stride_mean * 100.0) if swing_mean  is not None else None
            else:
                stance_pct = None
                swing_pct  = None

            # Gauges:
            # - Stride time: keep seconds, show a neutral "comfortable" band (0.90–1.30 s).
            #   (You may adjust per population or cadence if available.)
            html_stride = mk_box(stride_mean, "STRIDE TIME", 0.60, 1.80, band=(1.10, 1.30))

            # - Stance % and Swing %: use 60/40 pattern bands with tolerance
            html_stance = mk_box(stance_pct, "STANCE %", 0.00, 100.0, band=(58.0, 62.0))
            html_swing  = mk_box(swing_pct,  "SWING %",  0.00, 100.0, band=(38.0, 42.0))

            return html_stride, html_stance, html_swing

        except Exception:
            return empty


    def _fig_to_data_uri(self, fig) -> str:
        """Convert a Matplotlib Figure to a PNG data URI."""
        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight", dpi=150)
        buf.seek(0)
        b64 = base64.b64encode(buf.read()).decode("ascii")
        return f"data:image/png;base64,{b64}"
    


    def generate_xai_analysis(self, patient_name: str, segment_idx: int = 0):
        """Generate XAI analysis plots for the selected segment (single-channel model)."""
        if not patient_name:
            return plt.figure(), plt.figure(), plt.figure(), [], {"text": "", "images": []}

        # -------- Replace _mask_to_ranges with gap-tolerant contiguous regions --------
        def _mask_to_regions(mask: np.ndarray, delta: int = 0):
            """
            Convert a boolean mask (length L) to contiguous discrepancy regions with
            a gap tolerance `delta`. Any zero-gaps <= delta between positives are bridged.
            Returns a list of (start_idx, end_idx) inclusive.
            """
            idx = np.where(mask)[0]
            if idx.size == 0:
                return []

            regions = []
            start = int(idx[0])
            prev  = int(idx[0])

            for i in map(int, idx[1:]):
                gap = i - prev - 1
                if gap <= delta:
                    # still in same region (bridge small gaps)
                    prev = i
                else:
                    # close previous region
                    regions.append((start, prev))
                    start = i
                    prev  = i
            regions.append((start, prev))
            return regions


        # -------- NEW: helper to draw vertical per-step relevance behind the waveform --------
        def _plot_vertical_relevance(ax, x, y, relevance, title, cmap, alpha=0.45):
            import numpy as np
            rel = np.asarray(relevance, dtype=float)
            # Robust normalize to [0,1]
            rmin, rmax = float(np.min(rel)), float(np.max(rel))
            rel = (rel - rmin) / (rmax - rmin + 1e-12)

            # y-limits with a small pad
            y_min, y_max = float(np.min(y)), float(np.max(y))
            pad = 0.05 * (y_max - y_min if y_max > y_min else 1.0)
            y0, y1 = y_min - pad, y_max + pad

            # Draw a 1xL image stretched vertically to create thin vertical bands
            # extent = [left, right, bottom, top]; use [0, L] so each column is one time step
            L = len(x)
            ax.imshow(rel[np.newaxis, :],
                      aspect='auto',
                      extent=[0, L, y0, y1],
                      origin='lower',
                      cmap=cmap,
                      interpolation='nearest',
                      alpha=alpha,
                      zorder=0)

            # Waveform on top
            ax.plot(x, y, linewidth=1.2, color='tab:blue', zorder=1)
            ax.set_xlim(0, L-1)
            ax.set_ylim(y0, y1)
            ax.set_title(title, fontsize=12, fontweight='bold')
            ax.set_xlabel('Time Steps')
            ax.set_ylabel('Signal Amplitude')
            ax.grid(True, alpha=0.25)

        try:
            # Build segments exactly as used by the model
            patient_info = self.patient_data[patient_name]
            gait_file = patient_info['gait_file']
            segments = preprocess_file(gait_file, SEGMENT_LENGTH)  # list of (1, L)

            if not segments or len(segments) == 0:
                fig, ax = plt.subplots(figsize=(16,4))
                ax.text(0.5, 0.5, 'No valid segments found.',
                        ha='center', va='center', transform=ax.transAxes)
                empty_context = {"text": "XAI context unavailable: no segments.", "images": []}
                return fig, fig, fig, [[None, "No segments available for XAI."]], empty_context

            # Clamp & select segment
            segment_idx = max(0, min(segment_idx, len(segments) - 1))
            full_segment = segments[segment_idx]      # (1, L)
            sensor_data = full_segment[0, :]          # (L,)
            time_points = np.arange(sensor_data.shape[0])


            _gm_text, gm_state = self.compute_gait_metrics_to_state(patient_name, segment_idx)

            # Compute relevance (length L, normalized to [0,1] by analyzers; we re-normalize anyway)
            gradcam_relevance = self.xai_analyzer.compute_gradcam_relevance(full_segment)
            lrp_relevance     = self.xai_analyzer.compute_lrp_relevance(full_segment)

            # ---------------------------- Discrepancy computation ----------------------------
            abs_diff = np.abs(gradcam_relevance - lrp_relevance)
            threshold = 0.5
            high_discrepancy_mask = abs_diff > threshold

            # --- NEW: merge mask into gap-tolerant regions ---
            gap_tolerance = 3  # ← tweak δ as you like (in time steps)
            merged_regions = _mask_to_regions(high_discrepancy_mask, delta=gap_tolerance)

            high_discrepancy_indices = np.where(high_discrepancy_mask)[0].astype(int).tolist()
            high_discrepancy_ranges = [
                {"start_idx": int(s), "end_idx": int(e), "length": int(e - s + 1)}
                for (s, e) in merged_regions
            ]


            # ---------------------------- Plot 1: Raw + *vertical* GradCAM relevance ----------------------------
            fig1, ax1 = plt.subplots(figsize=(16, 4))
            _plot_vertical_relevance(
                ax1,
                time_points,
                sensor_data,
                gradcam_relevance,
                f'GradCAM Explanation',
                cmap=self.xai_analyzer.colormap,
                alpha=0.45
            )

            # ---------------------------- Plot 2: Raw + *vertical* LRP relevance ----------------------------
            fig2, ax2 = plt.subplots(figsize=(16, 4))
            _plot_vertical_relevance(
                ax2,
                time_points,
                sensor_data,
                lrp_relevance,
                f'LRP Explanation',
                cmap=self.xai_analyzer.colormap,
                alpha=0.45
            )

            # ---------------------------- Plot 3: Raw + vertical red discrepancy bands ----------------------------
            # ---------------------------- Plot 3: Raw + merged high-discrepancy bands ----------------------------
            fig3, ax3 = plt.subplots(figsize=(16,4))
            for (s, e) in merged_regions:
                ax3.axvspan(s, e + 1, color='red', alpha=0.22, lw=0, zorder=0)
            ax3.plot(time_points, sensor_data, 'b-', linewidth=1.2, zorder=1)
            ax3.set_title(
                f'Raw Signal with High-Discrepancy Regions',
                fontsize=12, fontweight='bold'
            )
            ax3.set_ylabel('Signal Amplitude'); ax3.set_xlabel('Time Steps'); ax3.grid(True, alpha=0.3)
            ax3.set_xlim(0, len(time_points)-1)


            # ---------------------------- Store for chatbot/XAI consumers ----------------------------
            covered = sum((e - s + 1) for (s, e) in merged_regions)
            discrepancy_percentage = 100.0 * covered / len(time_points)

            self.current_results['xai_analysis'] = {
                'segment_idx': int(segment_idx),
                'threshold': float(threshold),
                'gradcam_relevance': gradcam_relevance,
                'lrp_relevance': lrp_relevance,
                'discrepancy': abs_diff,
                'high_discrepancy_points_count': int(np.sum(high_discrepancy_mask)),
                'high_discrepancy_indices': high_discrepancy_indices,
                'high_discrepancy_ranges': high_discrepancy_ranges,
                'discrepancy_percentage': discrepancy_percentage
            }


            ranges_str = ", ".join([f"[{r['start_idx']},{r['end_idx']}]" for r in high_discrepancy_ranges[:20]])
            if len(high_discrepancy_ranges) > 20:
                ranges_str += f" (+{len(high_discrepancy_ranges)-20} more)"


            # --------------------- Build minimal LLM context ---------------------
            if 'prediction' in self.current_results:
                pred = self.current_results['prediction']

                # Continuous high-discrepancy regions from merged_regions computed above
                regions = merged_regions  # list of (start, end)
                region_str = "; ".join(f"[{s},{e}]" for (s, e) in regions) if regions else "None"

                gait_block = self._format_gait_metrics_block(gm_state)
                text_context = (
                    f"Patient: {patient_name}\n"
                    f"Prediction: {pred['class']} (confidence {pred['confidence']:.3f})\n"
                    f"Discrepancy %: {discrepancy_percentage:.1f}%\n"
                    f"Continuous discrepancy region(s): {region_str}\n"
                    f"Gait event:\n{gait_block}"
                )
                chat_history = [[None, text_context]]
            else:
                chat_history = [[None, "Please run a prediction first!"]]


            # Convert plots to data URIs
            img1_uri = self._fig_to_data_uri(fig1)
            img2_uri = self._fig_to_data_uri(fig2)
            img3_uri = self._fig_to_data_uri(fig3)
            context_package = {"text": chat_history[0][1], "images": [img1_uri, img2_uri, img3_uri]}

            return fig1, fig2, fig3, chat_history, context_package

        except Exception as e:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.text(0.5, 0.5, f'Error in XAI analysis: {str(e)}',
                    ha='center', va='center', transform=ax.transAxes)
            error_history = [[None, f"Error generating XAI analysis: {str(e)}"]]
            empty_context = {"text": "XAI context unavailable due to an error.", "images": []}
            return fig, fig, fig, error_history, empty_context



#-----------------------------------------------------------
# ------------------- Gradio Interface ----------------------
#-----------------------------------------------------------
def create_gradio_interface():
    """Create the Gradio interface"""
    
    # Initialize the app
    app = ParkinsonsGaitApp()

    # Create Gradio interface
    with gr.Blocks(title="GUI", css=css, theme= gr.themes.Default()).queue() as demo: #'shivi/calm_seafoam' 'gr.themes.Default()'

        # gr.Markdown("""# Motion2Meaning""")

        
        chat_context = gr.State({"text": "", "images": []})
        histories_state = gr.State({})

        with gr.Tab("Parkinson's Gait Analysis"):
            # Patient Selection and Raw Data Visualization
            with gr.Group():
                gr.HTML("<h2>📊 Patient Data Selection</h2>")
                with gr.Row():
                    with gr.Column():
                        patient_selector = gr.Dropdown(
                            choices=app.patient_names,
                            label="Select Patient:",
                            value=app.patient_names[0] if app.patient_names else None
                        )
                    with gr.Column():
                        segment_selector = gr.Slider(
                            minimum=0, maximum=0, step=1, value=0,
                            label="Select Segment",
                            interactive=True
                        )
                with gr.Row():
                    raw_data_plot = gr.Plot(label="Raw Gait Data", show_label=False)

                with gr.Row():
                # ---------- NEW: Gait Events Overview gauges ----------
                    gr.HTML("""
                        <div style="text-align:center; font-size:20px; font-weight:600; margin:14px 0;">
                            Gait Events Overview (Left Foot)
                        </div>
                    """)
                with gr.Row():
                    with gr.Column(scale=1):
                        gauge_stride = gr.HTML(
                            render_gait_parameter(value=None, param_type="STRIDE TIME",
                                                min_val=0.60, max_val=1.80, threshold=None, band=(1.10, 1.30))
                        )
                    with gr.Column(scale=1):
                        gauge_stance = gr.HTML(
                            render_gait_parameter(value=None, param_type="STANCE %",
                                                min_val=0.00, max_val=100.0, threshold=None, band=(58.0, 62.0))
                        )
                    with gr.Column(scale=1):
                        gauge_swing = gr.HTML(
                            render_gait_parameter(value=None, param_type="SWING %",
                                                min_val=0.00, max_val=100.0, threshold=None, band=(38.0, 42.0))
                        )

                
                # Update plots AND gauges when selections change
                patient_selector.change(
                    fn=app.update_segment_slider,
                    inputs=patient_selector,
                    outputs=segment_selector
                ).then(
                    fn=app.plot_raw_gait_data,
                    inputs=[patient_selector, segment_selector],
                    outputs=raw_data_plot
                ).then(
                    fn=app.compute_gait_param_boxes,
                    inputs=[patient_selector, segment_selector],
                    outputs=[gauge_stride, gauge_stance, gauge_swing]
                )

                # When segment changes: refresh plot + gauges
                segment_selector.change(
                    fn=app.plot_raw_gait_data,
                    inputs=[patient_selector, segment_selector],
                    outputs=raw_data_plot
                ).then(
                    fn=app.compute_gait_param_boxes,
                    inputs=[patient_selector, segment_selector],
                    outputs=[gauge_stride, gauge_stance, gauge_swing]
                )

            
            # Prediction and Gait Metrics
            with gr.Group():
                gr.HTML("<h2>🔬 Analysis & Prediction</h2>")
                with gr.Row():
                    with gr.Column(scale=1):
                        predict_button = gr.Button("Run AI Prediction")
                        xai_button = gr.Button("Generate XAI Analysis")
                    with gr.Column(scale = 1):
                        prediction_output = gr.Text(label="Hoeh & Yahr Stage", lines=3)
                    with gr.Column(scale=1.2):
                        with gr.Row():
                            flag_btn = gr.Button("Contest & Justify")
                        with gr.Row():
                            flag_taxonomy = gr.CheckboxGroup(choices=["Factual Error", "Normative Conflict", "Reasoning Flaw"], label="Select Typoes of Issues", interactive=True)

            # XAI Analysis Plots
            with gr.Group():
                gr.HTML("<h2>🔍 Explainable AI Analysis</h2>")
                with gr.Row():
                    gradcam_plot = gr.Plot(label="GradCAM Explanation", show_label=True)
                with gr.Row():
                    lrp_plot = gr.Plot(label="LRP Explanation", show_label=True)
                with gr.Row():
                    discrepancy_plot = gr.Plot(label="Discrepancy Analysis", show_label=True)
        
        with gr.Tab("AI Clinical Assistant"):
            # Chatbot Interface
            with gr.Group():
                gr.HTML("<h2>🤖 AI Clinical Assistant</h2>")
                with gr.Row():
                    chat_interface = gr.Chatbot(
                        value=[[None, "Hello! I'm your AI clinical assistant for Parkinson's gait analysis. I can help you interpret model predictions, gait metrics, XAI explanations, and provide clinical insights. Please select a patient and run the analysis to get started."]],
                        show_copy_button=True,
                        show_share_button=True,
                        height=1000,
                    )
                with gr.Row():
                    with gr.Column(scale=1):
                        llm_selector = gr.Dropdown(
                            choices=[
                                "meta-llama/Llama-3.3-70B-Instruct:fireworks-ai",
                                "openai/gpt-oss-20b:fireworks-ai",
                                "meta-llama/Llama-4-Scout-17B-16E-Instruct:fireworks-ai",
                                "gpt-4o"
                            ],
                            value="meta-llama/Llama-3.3-70B-Instruct:fireworks-ai",
                            label="Select a LLM"
                        )
                    with gr.Column(scale=5):
                        msg = gr.Textbox(
                            placeholder="Ask about the analysis results, gait patterns, or clinical insights...",
                            show_label=False,
                            scale=9,
                            lines=2
                        )
                    clear_chat = gr.Button("🗑️ Clear Chat", scale=1)
        
        
        def on_contest_and_justify(ctx: dict, selected_types: list):
            """
            Append clinician challenge info into chat_context['text'] so the LLM
            can condition on it next time the user sends a message.
            """
            chosen = ", ".join(selected_types) if selected_types else "None"
            appendix = (
                f"Typos of challenges from clinicians: {chosen}\n"
                "Brief description of challenge types: "
                "Factual Error, to contest the integrity of the input gait signal; "
                "Normative Conflict, to flag a decision that contradicts their clinical expertise and established medical knowledge;" 
                "or Reasoning Flaw, to challenge the plausibility of the visual attribution provided by the explanation module.\n"
            )

            prev_text = (ctx or {}).get("text", "")
            images = (ctx or {}).get("images", [])

            new_text = (prev_text + ("\n\n" if prev_text else "") + appendix).strip()
            return {"text": new_text, "images": images}

        # Button click handlers
        predict_button.click(
            fn=app.predict_severity,
            inputs=[patient_selector, segment_selector],
            outputs=prediction_output
        )

        xai_button.click(
            fn=app.generate_xai_analysis,
            inputs=[patient_selector, segment_selector],
            outputs=[gradcam_plot, lrp_plot, discrepancy_plot, chat_interface, chat_context]
        )

        # Store clinician contest/justification signal into chat_context
        flag_btn.click(
            fn=on_contest_and_justify,
            inputs=[chat_context, flag_taxonomy],
            outputs=chat_context
        )

        def _default_greeting():
            return [[None, "Hello! I'm your AI clinical assistant for Parkinson's gait analysis. I can help you interpret model predictions, gait metrics, XAI explanations, and provide clinical insights. Please select a patient and run the analysis to get started."]]

        def on_model_change_load(histories: dict, model_name: str):
            # load existing history for this model or start fresh
            hist = histories.get(model_name) or _default_greeting()
            return hist, histories

        def save_current_history(histories: dict, model_name: str, chat_hist):
            # persist the latest chat for the active model
            histories[model_name] = chat_hist
            return histories

        def clear_current_history(histories: dict, model_name: str):
            hist = _default_greeting()
            histories[model_name] = hist
            return hist, histories
        
        llm_selector.change(
            fn=on_model_change_load,
            inputs=[histories_state, llm_selector],
            outputs=[chat_interface, histories_state]
        )

        # Chat functionality
        msg.submit(
            fn=app.chatbot.generate_response,
            inputs=[chat_interface, msg, chat_context, llm_selector],
            outputs=chat_interface,
            show_progress="hidden"
        ).then(
            fn=save_current_history,
            inputs=[histories_state, llm_selector, chat_interface],
            outputs=histories_state
        ).then(
            lambda: "",
            None,
            msg
        )
        
        clear_chat.click(
            fn=clear_current_history,
            inputs=[histories_state, llm_selector],
            outputs=[chat_interface, histories_state]
        )
        
        # Initialize 
        demo.load(
            fn=app.update_segment_slider,
            inputs=patient_selector,
            outputs=segment_selector
        ).then(
            fn=app.plot_raw_gait_data,
            inputs=[patient_selector, segment_selector],
            outputs=raw_data_plot
        ).then(
            fn=app.compute_gait_param_boxes,
            inputs=[patient_selector, segment_selector],
            outputs=[gauge_stride, gauge_stance, gauge_swing]
        )

    return demo


def main():    

    # Create and launch the Gradio interface
    demo = create_gradio_interface()

    demo.queue().launch(
        debug=True,
        # share=True, 
        server_port=7860,
        show_error=True
    )

if __name__ == "__main__":
    main()