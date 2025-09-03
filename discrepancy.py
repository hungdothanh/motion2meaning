# -------------------discrepancy.py-------------------
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from typing import Optional, Tuple
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.patches as mpatches
import pandas as pd

from config import PRETRAINED_MODEL_PATH, SEGMENT_LENGTH, CLASS_NAMES
from data import load_data
from model import ParkinsonsGaitCNN

# XAI modules
from gradCAM import GradCAM1D
from lrp import LRPModel

class XAIComparativeAnalyzer:
    """
    Comparative analyzer for GradCAM and LRP explanations on 1-channel gait data.
    """

    def __init__(self, model_path: str = PRETRAINED_MODEL_PATH):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # IMPORTANT: 1 input channel and your trained sequence length
        self.model = ParkinsonsGaitCNN(input_channels=1, sequence_length=SEGMENT_LENGTH)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device).eval()

        # XAI instances
        self.gradcam = GradCAM1D(self.model, target_layers=["conv4"])
        self.lrp_model = LRPModel(self.model).to(self.device).eval()

        # Simple blue→red colormap
        self.colormap = LinearSegmentedColormap.from_list('simple', ['blue', 'cyan', 'yellow', 'red'])


    def extract_sensor_data(self, 
                        patient_name: str,
                        start_time: int = 0,
                        sequence_length: int = 1000) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract the single input channel (Column 18: total left force) for a given time window.

        Args:
            gait_txt_path: Path to the raw gait .txt file (tab-separated)
            start_time: starting index (row)
            sequence_length: window length (L)

        Returns:
            (sensor_data_1d, full_segment) where:
              sensor_data_1d: shape (L,)
              full_segment: shape (1, L)  # channels-first for the model/XAI
        """

        # Load patient data
        data = load_data()[patient_name]
        
        # Load raw data file
        raw_data = pd.read_csv(data['gait_file'], sep='\t', header=None)

        # raw = pd.read_csv(gait_txt_path, sep='\t', header=None)

        # if raw.shape[1] < 18:
        #     raise ValueError(f"{gait_txt_path} has {raw.shape[1]} columns; need >= 18 to access column 18.")

        # Select only Column 18 (0-based index 17) -> (T, 1)
        col18 = raw_data.iloc[:, [17]].to_numpy()
        T = col18.shape[0]

        end = min(start_time + sequence_length, T)
        window = col18[start_time:end, :]  # (len, 1)

        if window.shape[0] < sequence_length:
            pad = np.zeros((sequence_length - window.shape[0], 1), dtype=window.dtype)
            window = np.vstack([window, pad])

        # Prepare outputs
        sensor_1d = window[:, 0]          # (L,)
        full_segment = window.T           # (1, L)

        return sensor_1d, full_segment

    def compute_gradcam_relevance(self, full_segment: np.ndarray,
                                target_class: Optional[int] = None,
                                layer_name: Optional[str] = None) -> np.ndarray:
        """
        Compute Grad-CAM relevance for the single channel.
        Args:
            full_segment: (1, L)
        Returns:
            1D CAM of length L (normalized to [0,1])
        """
        x = torch.as_tensor(full_segment, dtype=torch.float32, device=self.device).unsqueeze(0)  # (1,1,L)
        cams, pred_class = self.gradcam.generate_cam(x, target_class=target_class, layer_name=layer_name)


        cam = self.gradcam._upsample_cam(cams["_aggregated_"], target_size=full_segment.shape[-1])
        cam = cam.detach().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-12)  # final normalization

        return cam

    def compute_lrp_relevance(
        self,
        full_segment: np.ndarray,
        target_class: Optional[int] = None,
        rule: str = 'epsilon'
    ) -> np.ndarray:
        """
        Compute LRP relevance for the single channel.
        Args:
            full_segment: (1, L)
        Returns:
            1D relevance (|…| normalized to [0,1]) of length L
        """
        x = torch.as_tensor(full_segment, dtype=torch.float32, device=self.device).unsqueeze(0)  # (1,1,L)
        with torch.no_grad():
            R = self.lrp_model.explain(x, target_class=target_class, rule=rule)  # (1, C=1, L)
        r = R[0, 0].detach().cpu().numpy()  # (L,)
        r = np.abs(r)
        if r.max() > r.min():
            r = (r - r.min()) / (r.max() - r.min())
        return r

    def plot_comparative_analysis(self,
                                patient_name: str,
                                start_time: int = 0,
                                sequence_length: int = 1000,
                                save_path: Optional[str] = None,
                                figsize: Tuple[int, int] = (15, 18)
                            ) -> plt.Figure:
        """
        Create comparative visualization of Grad-CAM vs LRP on the single input channel.
        """
        # Extract data
        sensor_data, full_segment = self.extract_sensor_data(
            patient_name, start_time, sequence_length
        )

        # Model prediction
        x = torch.as_tensor(full_segment, dtype=torch.float32, device=self.device).unsqueeze(0)  # (1,1,L)
        with torch.no_grad():
            out = self.model(x)
            probs = F.softmax(out, dim=1)
            pred_class = int(out.argmax(dim=1).item())
            confidence = float(probs[0, pred_class].item())

        # Relevance
        gradcam_rel = self.compute_gradcam_relevance(full_segment)
        lrp_rel = self.compute_lrp_relevance(full_segment)

        # Plot
        fig, axes = plt.subplots(5, 1, figsize=figsize, facecolor='white')
        t = np.arange(sensor_data.shape[0])


        # ---------------------------- Fig 1: Raw + GradCAM overlay ----------------------------
        ax1 = axes[0]
        ax1.plot(t, sensor_data, 'k-', alpha=0.5, linewidth=1, label='Raw Data')
        for i in range(len(t) - 1):
            color = self.colormap(gradcam_rel[i])
            ax1.plot([t[i], t[i+1]], [sensor_data[i], sensor_data[i+1]], color=color, alpha=0.8, linewidth=2)
        ax1.set_title(f'Fig 1: Raw Data with GradCAM Overlay', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Amplitude'); ax1.grid(True, alpha=0.3); ax1.set_xlim(0, len(t)-1)
        sm1 = plt.cm.ScalarMappable(cmap=self.colormap, norm=plt.Normalize(0, 1))
        sm1.set_array([]); cbar1 = plt.colorbar(sm1, ax=ax1, fraction=0.03, pad=0.02)
        cbar1.set_label('GradCAM Relevance', fontsize=9)

        # ---------------------------- Fig 2: Raw + LRP overlay ----------------------------
        ax2 = axes[1]
        ax2.plot(t, sensor_data, 'k-', alpha=0.5, linewidth=1, label='Raw Data')
        for i in range(len(t) - 1):
            color = self.colormap(lrp_rel[i])
            ax2.plot([t[i], t[i+1]], [sensor_data[i], sensor_data[i+1]], color=color, alpha=0.8, linewidth=2)
        ax2.set_title(f'Fig 2: Raw Data with LRP Overlay', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Amplitude'); ax2.grid(True, alpha=0.3); ax2.set_xlim(0, len(t)-1)
        sm2 = plt.cm.ScalarMappable(cmap=self.colormap, norm=plt.Normalize(0, 1))
        sm2.set_array([]); cbar2 = plt.colorbar(sm2, ax=ax2, fraction=0.03, pad=0.02)
        cbar2.set_label('LRP Relevance', fontsize=9)

        # ---------------------------- Fig 3: Relevance comparison ----------------------------
        ax3 = axes[2]
        ax3.plot(t, gradcam_rel, linewidth=2, alpha=0.9, label='GradCAM')
        ax3.plot(t, lrp_rel, linewidth=2, alpha=0.9, label='LRP')
        ax3.set_title('Fig 3: Normalized Relevance Scores Comparison', fontsize=12, fontweight='bold')
        ax3.set_ylabel('Normalized Relevance'); ax3.set_ylim(-0.05, 1.05); ax3.grid(True, alpha=0.3)
        ax3.legend(loc='upper right'); ax3.set_xlim(0, len(t)-1)

        # ---------------------------- Fig 4: Absolute difference + threshold ----------------------------
        ax4 = axes[3]
        abs_diff = np.abs(gradcam_rel - lrp_rel)
        threshold = 0.5
        ax4.plot(t, abs_diff, linewidth=2, label='|GradCAM - LRP|')
        ax4.axhline(y=threshold, color='orange', linestyle='--', alpha=0.6, label=f'Threshold = {threshold}')
        mask = abs_diff > threshold
        ax4.fill_between(t, 0, abs_diff, where=mask, color='red', alpha=0.3, label='High Discrepancy')
        ax4.set_title('Fig 4: Absolute Difference with High Discrepancy Regions', fontsize=12, fontweight='bold')
        ax4.set_ylabel('Abs Diff'); ax4.set_ylim(-0.05, 1.05); ax4.grid(True, alpha=0.3)
        ax4.legend(loc='upper right'); ax4.set_xlim(0, len(t)-1)

        # ---------------------------- Fig 5: Raw with highlighted discrepancy regions ----------------------------
        ax5 = axes[4]
        ax5.plot(t, sensor_data, 'k-', linewidth=1.5, label='Raw Data')
        for i in range(len(t)):
            if mask[i]:
                ax5.axvspan(max(0, i-1), min(len(t)-1, i+1), color='red', alpha=0.2)
        if np.any(mask):
            red_patch = mpatches.Patch(color='red', alpha=0.2, label='High Discrepancy')
            ax5.legend(handles=[ax5.get_lines()[0], red_patch], loc='upper right')
        else:
            ax5.legend(loc='upper right')
        ax5.set_title('Fig 5: Raw Data with High Discrepancy Regions Highlighted', fontsize=12, fontweight='bold')
        ax5.set_ylabel('Amplitude'); ax5.set_xlabel('Time Steps'); ax5.grid(True, alpha=0.3)
        ax5.set_xlim(0, len(t)-1)

        fig.suptitle(
            f'XAI Comparative Analysis\nPredicted: {CLASS_NAMES[pred_class]} (Confidence: {confidence:.3f})\n'
            f'Colormap: Blue→Red (Low→High)',
            fontsize=14, fontweight='bold', y=1.02
        )
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300)
            print(f"Figure saved to: {save_path}")

        return fig
