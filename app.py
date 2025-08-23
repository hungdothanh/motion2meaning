#-------------------app.py-------------------

import gradio as gr
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import io, base64

# Import your existing modules
from config import PRETRAINED_MODEL_PATH, SEGMENT_LENGTH, CLASS_NAMES, SENSOR_NAMES
from data import load_data, preprocess_file
from model import ParkinsonsGaitCNN
from discrepancy import XAIComparativeAnalyzer
from gait_event import GaitMetricsCalculator
from chatbox import ParkinsonsGaitChatbot
from apikey import MY_OPENAI_API_KEY


class ParkinsonsGaitApp:
    """Main application class for Parkinson's Gait Analysis"""
    
    def __init__(self, openai_api_key: str):
        """Initialize the application"""
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load model
        self.model = ParkinsonsGaitCNN(input_channels=16, sequence_length=SEGMENT_LENGTH)
        self.model.load_state_dict(torch.load(PRETRAINED_MODEL_PATH, map_location=self.device))
        self.model.to(self.device).eval()
        
        # Initialize XAI analyzer
        self.xai_analyzer = XAIComparativeAnalyzer(PRETRAINED_MODEL_PATH)
        
        # Initialize chatbot
        self.chatbot = ParkinsonsGaitChatbot(openai_api_key)
        
        # Load patient data
        self.patient_data = load_data()
        self.patient_names = list(self.patient_data.keys())
        
        # Store current analysis results
        self.current_results = {}
    
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


    def plot_raw_gait_data(self, patient_name: str, sensor_idx: int = 0, segment_idx: int = 0):
        """Plot raw gait data for selected patient and sensor, and segment idx"""
        if not patient_name:
            return plt.figure()

        try:
            # Build segments using the same preprocessing the model uses
            patient_info = self.patient_data[patient_name]
            segments = preprocess_file(patient_info['gait_file'], SEGMENT_LENGTH)

            if not segments or len(segments) == 0:
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.text(0.5, 0.5, 'No valid 1000-sample segments found.',
                        ha='center', va='center', transform=ax.transAxes)
                return fig

            # Clamp segment_idx to valid range
            segment_idx = max(0, min(segment_idx, len(segments) - 1))
            seg = segments[segment_idx]  # shape: (channels, 1000)

            # Clamp sensor index
            sensor_idx = max(0, min(sensor_idx, seg.shape[0] - 1))

            # Plot the chosen channel from the chosen segment
            sensor_data = seg[sensor_idx, :]  # length 1000
            time_data = np.arange(sensor_data.shape[0])

            fig, ax = plt.subplots(figsize=(15, 6))
            ax.plot(time_data, sensor_data, linewidth=1)
            ax.set_xlabel('Time Steps', fontsize=12)
            ax.set_ylabel('Force (N)', fontsize=12)
            label = SENSOR_NAMES[sensor_idx] if sensor_idx < len(SENSOR_NAMES) else f"Sensor {sensor_idx}"
            ax.set_title(f'{patient_name} • {label} • Segment #{segment_idx}', fontsize=14, fontweight='bold')
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

    def compute_gait_metrics(self, patient_name: str):
        """Compute gait event metrics for selected patient"""
        if not patient_name:
            return "Please select a patient"
            
        try:
            patient_info = self.patient_data[patient_name]
            calculator = GaitMetricsCalculator(patient_info['gait_file'])
            
            # Analyze gait with appropriate threshold
            results = calculator.analyze_gait(force_threshold=20)
            
            # Format results
            metrics_text = f"=== GAIT METRICS for {patient_name} ===\n\n"
            
            # Left foot metrics
            if len(results['left_stride_times']) > 0:
                metrics_text += f"LEFT FOOT:\n"
                metrics_text += f"  Stride Time: {np.mean(results['left_stride_times']):.3f} ± {np.std(results['left_stride_times']):.3f} sec ({len(results['left_stride_times'])} cycles)\n"
                metrics_text += f"  Stance Time: {np.mean(results['left_stance_times']):.3f} ± {np.std(results['left_stance_times']):.3f} sec ({len(results['left_stance_times'])} cycles)\n"
                metrics_text += f"  Swing Time:  {np.mean(results['left_swing_times']):.3f} ± {np.std(results['left_swing_times']):.3f} sec ({len(results['left_swing_times'])} cycles)\n\n"
            
            # Right foot metrics
            if len(results['right_stride_times']) > 0:
                metrics_text += f"RIGHT FOOT:\n"
                metrics_text += f"  Stride Time: {np.mean(results['right_stride_times']):.3f} ± {np.std(results['right_stride_times']):.3f} sec ({len(results['right_stride_times'])} cycles)\n"
                metrics_text += f"  Stance Time: {np.mean(results['right_stance_times']):.3f} ± {np.std(results['right_stance_times']):.3f} sec ({len(results['right_stance_times'])} cycles)\n"
                metrics_text += f"  Swing Time:  {np.mean(results['right_swing_times']):.3f} ± {np.std(results['right_swing_times']):.3f} sec ({len(results['right_swing_times'])} cycles)\n\n"
            
            # Store for chatbot context
            self.current_results['gait_metrics'] = results
            
            return metrics_text
            
        except Exception as e:
            return f"Error computing gait metrics: {str(e)}"

    def _fig_to_data_uri(self, fig) -> str:
        """Convert a Matplotlib Figure to a PNG data URI."""
        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight", dpi=150)
        buf.seek(0)
        b64 = base64.b64encode(buf.read()).decode("ascii")
        return f"data:image/png;base64,{b64}"
    
    def generate_xai_analysis(self, patient_name: str, sensor_idx: int = 0):
        """Generate XAI analysis plots and update chat history"""
        if not patient_name:
            return plt.figure(), plt.figure(), plt.figure(), []
            
        try:
            # Generate XAI comparative analysis
            fig = self.xai_analyzer.plot_comparative_analysis(
                patient_name=patient_name,
                sensor_idx=sensor_idx,
                start_time=0,
                sequence_length=1000
            )
            
            # Plot 1: Raw data with GradCAM overlay
            fig1, ax1 = plt.subplots(figsize=(15, 6))
            # Get data for recreating plots
            sensor_data, full_segment = self.xai_analyzer.extract_sensor_data(
                patient_name, sensor_idx, 0, 1000
            )
            gradcam_relevance = self.xai_analyzer.compute_gradcam_relevance(full_segment, sensor_idx)
            
            time_points = np.arange(len(sensor_data))
            ax1.plot(time_points, sensor_data, 'k-', alpha=0.5, linewidth=1, label='Raw Data')
            
            # Create GradCAM overlay
            for i in range(len(time_points) - 1):
                color_intensity = gradcam_relevance[i]
                color = self.xai_analyzer.colormap(color_intensity)
                ax1.plot([time_points[i], time_points[i+1]], 
                        [sensor_data[i], sensor_data[i+1]], 
                        color=color, alpha=0.8, linewidth=2)
            
            ax1.set_title(f'Raw Data with GradCAM Overlay - {SENSOR_NAMES[sensor_idx]}', fontsize=12, fontweight='bold')
            ax1.set_ylabel('Signal Amplitude', fontsize=10)
            ax1.set_xlabel('Time Steps', fontsize=10)
            ax1.grid(True, alpha=0.3)
            plt.tight_layout()
            
            # Plot 2: Raw data with LRP overlay  
            fig2, ax2 = plt.subplots(figsize=(15, 6))
            lrp_relevance = self.xai_analyzer.compute_lrp_relevance(full_segment, sensor_idx)
            
            ax2.plot(time_points, sensor_data, 'k-', alpha=0.5, linewidth=1, label='Raw Data')
            
            # Create LRP overlay
            for i in range(len(time_points) - 1):
                color_intensity = lrp_relevance[i]
                color = self.xai_analyzer.colormap(color_intensity)
                ax2.plot([time_points[i], time_points[i+1]], 
                        [sensor_data[i], sensor_data[i+1]], 
                        color=color, alpha=0.8, linewidth=2)
            
            ax2.set_title(f'Raw Data with LRP Overlay - {SENSOR_NAMES[sensor_idx]}', fontsize=12, fontweight='bold')
            ax2.set_ylabel('Signal Amplitude', fontsize=10)
            ax2.set_xlabel('Time Steps', fontsize=10)
            ax2.grid(True, alpha=0.3)
            plt.tight_layout()
            
            # Plot 3: Discrepancy analysis
            fig3, ax3 = plt.subplots(figsize=(15, 6))
            abs_diff = np.abs(gradcam_relevance - lrp_relevance)
            threshold = 0.5
            high_discrepancy_mask = abs_diff > threshold
            
            ax3.plot(time_points, sensor_data, 'k-', linewidth=1.5, label='Raw Data')
            
            # Highlight regions with high discrepancy
            for i in range(len(time_points)):
                if high_discrepancy_mask[i]:
                    ax3.axvspan(max(0, i-1), min(len(time_points)-1, i+1), 
                              color='red', alpha=0.2)
            
            ax3.set_title('Raw Data with High Discrepancy Regions Highlighted', fontsize=12, fontweight='bold')
            ax3.set_ylabel('Signal Amplitude', fontsize=10)
            ax3.set_xlabel('Time Steps', fontsize=10)
            ax3.grid(True, alpha=0.3)
            ax3.legend(loc='upper right')
            plt.tight_layout()
            
            # Store results for chatbot
            self.current_results['xai_analysis'] = {
                'sensor_idx': sensor_idx,
                'gradcam_relevance': gradcam_relevance,
                'lrp_relevance': lrp_relevance,
                'discrepancy': abs_diff,
                'high_discrepancy_regions': np.sum(high_discrepancy_mask),
                'discrepancy_percentage': np.mean(high_discrepancy_mask) * 100
            }
            
            # Create initial chat history in Gradio format [[user, assistant], [user, assistant], ...]
            chat_history = []
            
            # Add analysis summary to chat
            if hasattr(self, 'current_results') and 'prediction' in self.current_results:
                prediction_info = self.current_results['prediction']
                analysis_summary = f"""Analysis Summary for {patient_name}:

                                    Model Prediction: {prediction_info['class']} (Confidence: {prediction_info['confidence']:.3f})

                                    XAI Analysis:
                                    - Sensor analyzed: {SENSOR_NAMES[sensor_idx]}
                                    - High discrepancy regions: {self.current_results['xai_analysis']['high_discrepancy_regions']} time points
                                    - Discrepancy percentage: {self.current_results['xai_analysis']['discrepancy_percentage']:.1f}%

                                    The red-highlighted regions show where GradCAM and LRP explanations significantly disagree, indicating areas of model uncertainty."""

                chat_history.append([None, analysis_summary])
            else:
                chat_history.append([None, "XAI Analysis completed. I can now help you interpret the results."])
            

            if 'prediction' in self.current_results:
                prediction_info = self.current_results['prediction']
                sensor_name = SENSOR_NAMES[sensor_idx]
                xai = self.current_results['xai_analysis']
                text_context = (
                    f"Model Prediction: {prediction_info['class']} "
                    f"(Confidence: {prediction_info['confidence']:.3f})\n\n"
                    f"XAI Analysis:\n"
                    f"- Sensor analyzed: {sensor_name}\n"
                    f"- High discrepancy regions: {xai['high_discrepancy_regions']} time points\n"
                    f"- Discrepancy percentage: {xai['discrepancy_percentage']:.1f}%\n\n"
                    "The red-highlighted regions show where GradCAM and LRP explanations "
                    "significantly disagree, indicating areas of model uncertainty."
                )
            else:
                text_context = (
                    "XAI Analysis context not available yet. Please run a prediction first."
                )

            # Convert the 3 plots (we just created) to data URIs
            img1_uri = self._fig_to_data_uri(fig1)
            img2_uri = self._fig_to_data_uri(fig2)
            img3_uri = self._fig_to_data_uri(fig3)

            context_package = {
                "text": text_context,
                "images": [img1_uri, img2_uri, img3_uri]
            }

            return fig1, fig2, fig3, chat_history, context_package
            
        except Exception as e:
            # Return empty figures and error message
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.text(0.5, 0.5, f'Error in XAI analysis: {str(e)}', 
                   ha='center', va='center', transform=ax.transAxes)
            
            error_history = [[None, f"Error generating XAI analysis: {str(e)}"]]
            empty_context = {"text": "XAI context unavailable due to an error.", "images": []}
            
            return fig, fig, fig, error_history, empty_context

    def _format_gait_metrics_summary(self) -> str:
        """Create a compact, readable gait metrics summary from self.current_results."""
        res = self.current_results.get('gait_metrics')
        if not res:
            return "Gait metrics not computed yet."

        def fmt_stats(arr, name):
            if len(arr) == 0:
                return f"{name}: n=0"
            return f"{name}: {float(np.mean(arr)):.3f} ± {float(np.std(arr)):.3f} sec (n={len(arr)})"

        lines = []
        left = [
            fmt_stats(res.get('left_stride_times', []), "Stride Time"),
            fmt_stats(res.get('left_stance_times', []), "Stance Time"),
            fmt_stats(res.get('left_swing_times', []),  "Swing Time"),
        ]
        right = [
            fmt_stats(res.get('right_stride_times', []), "Stride Time"),
            fmt_stats(res.get('right_stance_times', []), "Stance Time"),
            fmt_stats(res.get('right_swing_times', []),  "Swing Time"),
        ]
        lines.append("LEFT FOOT")
        lines.extend([f"  • {s}" for s in left])
        lines.append("RIGHT FOOT")
        lines.extend([f"  • {s}" for s in right])
        return "\n".join(lines)

    def _format_xai_summary(self, patient_name: str) -> str:
        """Summarize XAI analysis and discrepancy."""
        xai = self.current_results.get('xai_analysis')
        if not xai:
            return "XAI analysis not generated yet."
        sensor_idx = xai.get('sensor_idx', 0)
        disc_pct = xai.get('discrepancy_percentage', 0.0)
        high_pts = xai.get('high_discrepancy_regions', 0)
        sensor_label = SENSOR_NAMES[sensor_idx] if sensor_idx < len(SENSOR_NAMES) else f"Sensor {sensor_idx}"
        return (
            f"Sensor analyzed: {sensor_label}\n"
            f"High discrepancy time points (GradCAM vs LRP): {high_pts}\n"
            f"Discrepancy percentage: {disc_pct:.1f}%\n"
            f"Note: Red-highlighted regions in the plot indicate diagnostic uncertainty."
        )


#-----------------------------------------------------------
# ------------------- Gradio Interface ----------------------
#-----------------------------------------------------------
def create_gradio_interface(openai_api_key: str):
    """Create the Gradio interface"""
    
    # Initialize the app
    app = ParkinsonsGaitApp(openai_api_key)

    # Create Gradio interface
    with gr.Blocks(theme=gr.themes.Default(), title="Parkinson's Gait Analysis") as demo:
        
        gr.HTML(
            "<h1 class='title'><center>Parkinson's Disease Gait Analysis System</center></h1>"
        )
        
        chat_context = gr.State({"text": "", "images": []})
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
                    sensor_selector = gr.Dropdown(
                        choices=[(name, idx) for idx, name in enumerate(SENSOR_NAMES)],
                        label="Select Sensor for Analysis:",
                        value=0
                    )
                    segment_selector = gr.Slider(
                        minimum=0, maximum=0, step=1, value=0,
                        label="Select Segment Index (each is 1000 samples)",
                        interactive=True
                    )
            with gr.Row():
                raw_data_plot = gr.Plot(label="Raw Gait Data", show_label=False)
            
            # Update plots when selections change
            patient_selector.change(
                fn=app.update_segment_slider,
                inputs=patient_selector,
                outputs=segment_selector
            ).then(
                fn=app.plot_raw_gait_data,
                inputs=[patient_selector, sensor_selector, segment_selector],
                outputs=raw_data_plot
            )

            # When sensor changes: refresh plot (uses current segment)
            sensor_selector.change(
                fn=app.plot_raw_gait_data,
                inputs=[patient_selector, sensor_selector, segment_selector],
                outputs=raw_data_plot
            )

            # When segment changes: refresh plot
            segment_selector.change(
                fn=app.plot_raw_gait_data,
                inputs=[patient_selector, sensor_selector, segment_selector],
                outputs=raw_data_plot
            )
        
        # Prediction and Gait Metrics
        with gr.Group():
            gr.HTML("<h2>🔬 Analysis & Prediction</h2>")
            with gr.Row(equal_height=True):
                with gr.Column():
                    predict_button = gr.Button("Predict Severity Score", variant="primary")
                    gait_metrics_button = gr.Button("Compute Gait Metrics")
                    xai_button = gr.Button("Generate XAI Analysis")
                with gr.Column():
                    prediction_output = gr.Text(label="AI Prediction Result", lines=2)
                    gait_metrics_output = gr.Text(label="Gait Metrics", lines=10, max_lines=15)
        
        # XAI Analysis Plots
        with gr.Group():
            gr.HTML("<h2>🔍 Explainable AI Analysis</h2>")
            with gr.Row():
                gradcam_plot = gr.Plot(label="GradCAM Explanation", show_label=True)
            with gr.Row():
                lrp_plot = gr.Plot(label="LRP Explanation", show_label=True)
            with gr.Row():
                discrepancy_plot = gr.Plot(label="Discrepancy Analysis", show_label=True)
        
        # Chatbot Interface
        with gr.Group():
            gr.HTML("<h2>🤖 AI Clinical Assistant</h2>")
            with gr.Row():
                chat_interface = gr.Chatbot(
                    value=[[None, "Hello! I'm your AI clinical assistant for Parkinson's gait analysis. I can help you interpret model predictions, gait metrics, XAI explanations, and provide clinical insights. Please select a patient and run the analysis to get started."]],
                    show_copy_button=True,
                    show_share_button=True,
                    height=600,
                )
            with gr.Row():
                msg = gr.Textbox(
                    placeholder="Ask about the analysis results, gait patterns, or clinical insights...",
                    show_label=False,
                    scale=9
                )
                clear_chat = gr.Button("🗑️ Clear Chat", scale=1)
        
        # Button click handlers
        predict_button.click(
            fn=app.predict_severity,
            inputs=[patient_selector, segment_selector],
            outputs=prediction_output
        )
        
        gait_metrics_button.click(
            fn=app.compute_gait_metrics,
            inputs=patient_selector,
            outputs=gait_metrics_output
        )
        
        xai_button.click(
            fn=app.generate_xai_analysis,
            inputs=[patient_selector, sensor_selector],
            outputs=[gradcam_plot, lrp_plot, discrepancy_plot, chat_interface, chat_context]
        )
        
        # Chat functionality
        msg.submit(
            fn=app.chatbot.generate_response,
            inputs=[chat_interface, msg, chat_context],
            outputs=chat_interface,
            show_progress="hidden"
        ).then(
            lambda: "",  # Clear the textbox
            None,
            msg
        )
        
        clear_chat.click(
            lambda: [[None, "Chat cleared. How can I help you with the gait analysis?"]],
            outputs=chat_interface
        )
        
        # Initialize 
        demo.load(
            fn=app.update_segment_slider,
            inputs=patient_selector,
            outputs=segment_selector
        ).then(
            fn=app.plot_raw_gait_data,
            inputs=[patient_selector, sensor_selector, segment_selector],
            outputs=raw_data_plot
        )

    
    return demo

def main():    
    # OpenAI API Key - Replace with your actual key
    OPENAI_API_KEY = MY_OPENAI_API_KEY
    
    # Create and launch the Gradio interface
    demo = create_gradio_interface(OPENAI_API_KEY)
    demo.queue().launch(
        debug=True,
        # share=True, 
        server_port=7860,
        show_error=True
    )

if __name__ == "__main__":
    main()