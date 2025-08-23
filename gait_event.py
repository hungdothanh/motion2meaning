# -------------------gait_event.py-------------------
import numpy as np
import pandas as pd
from scipy.signal import find_peaks
import matplotlib.pyplot as plt

class GaitMetricsCalculator:
    def __init__(self, data_file_path):
        """
        Initialize the gait metrics calculator with force sensor data.
        
        Args:
            data_file_path: Path to the data file containing force sensor readings
        """
        self.data = self.load_data(data_file_path)
        self.heel_strikes = {'left': [], 'right': []}
        self.toe_offs = {'left': [], 'right': []}
        
    def load_data(self, file_path):
        """Load the force sensor data from file."""
        # Column names based on the data format specification
        columns = ['time'] + [f'L{i}' for i in range(1, 9)] + [f'R{i}' for i in range(1, 9)] + ['total_left', 'total_right']
        
        # Load data (assuming space or tab separated)
        data = pd.read_csv(file_path, sep=None, engine='python', names=columns, header=None)
        return data
    
    def detect_heel_strikes_toe_offs(self, foot='left', force_threshold=50, min_distance=50):
        """
        Detect heel strikes and toe-offs using total force under each foot.
        
        HEEL STRIKE DETECTION:
        - Uses TOTAL FORCE (column 18 for left, column 19 for right)
        - Heel strike occurs when total force rises above threshold (foot contacts ground)
        - We look for the rising edge of force signal
        
        TOE-OFF DETECTION:
        - Uses TOTAL FORCE (column 18 for left, column 19 for right)
        - Toe-off occurs when total force drops below threshold (foot leaves ground)
        - We look for the falling edge of force signal
        
        Args:
            foot: 'left' or 'right'
            force_threshold: Minimum force to consider foot contact (in Newtons)
            min_distance: Minimum samples between consecutive events
        """
        if foot == 'left':
            force_data = self.data['total_left'].values
        else:
            force_data = self.data['total_right'].values
        
        time_data = self.data['time'].values
        
        # Create binary contact signal (1 = contact, 0 = no contact)
        contact_signal = (force_data > force_threshold).astype(int)
        
        # Find transitions
        diff_signal = np.diff(contact_signal)
        
        # Heel strikes: transitions from 0 to 1 (foot contacts ground)
        # These should occur at the START of force increase
        heel_strike_indices = np.where(diff_signal == 1)[0]  # Don't add 1 - we want the transition point
        
        # Toe-offs: transitions from 1 to 0 (foot leaves ground)
        # These should occur at the END of force, when it drops to zero
        toe_off_indices = np.where(diff_signal == -1)[0]  # Don't add 1 - we want the transition point
        
        # Remove events that are too close together (noise filtering)
        heel_strike_indices = self._filter_close_events(heel_strike_indices, min_distance)
        toe_off_indices = self._filter_close_events(toe_off_indices, min_distance)
        
        # Convert indices to time stamps
        heel_strike_times = time_data[heel_strike_indices] if len(heel_strike_indices) > 0 else np.array([])
        toe_off_times = time_data[toe_off_indices] if len(toe_off_indices) > 0 else np.array([])
        
        self.heel_strikes[foot] = heel_strike_times
        self.toe_offs[foot] = toe_off_times
        
        return heel_strike_times, toe_off_times
    
    def _filter_close_events(self, indices, min_distance):
        """Remove events that are too close together to reduce noise."""
        if len(indices) == 0:
            return indices
        
        filtered_indices = [indices[0]]
        for i in range(1, len(indices)):
            if indices[i] - filtered_indices[-1] >= min_distance:
                filtered_indices.append(indices[i])
        
        return np.array(filtered_indices)
    
    def calculate_stride_time(self, foot='left'):
        """
        Calculate stride time: time between consecutive heel strikes of the same foot.
        Formula: Stride Time = t(HS_i+1) - t(HS_i)
        """
        heel_strikes = self.heel_strikes[foot]
        
        if len(heel_strikes) < 2:
            return []
        
        stride_times = []
        for i in range(len(heel_strikes) - 1):
            stride_time = heel_strikes[i + 1] - heel_strikes[i]
            stride_times.append(stride_time)
        
        return np.array(stride_times)
    
    def calculate_stance_time(self, foot='left'):
        """
        Calculate stance time: time when foot is in contact with ground.
        Formula: Stance Time = t(TO_i) - t(HS_i)
        """
        heel_strikes = self.heel_strikes[foot]
        toe_offs = self.toe_offs[foot]
        
        stance_times = []
        
        # Match each heel strike with its corresponding toe-off
        for hs_time in heel_strikes:
            # Find the next toe-off after this heel strike
            next_toe_offs = toe_offs[toe_offs > hs_time]
            if len(next_toe_offs) > 0:
                to_time = next_toe_offs[0]
                stance_time = to_time - hs_time
                stance_times.append(stance_time)
        
        return np.array(stance_times)
    
    def calculate_swing_time(self, foot='left'):
        """
        Calculate swing time: time when foot is in the air.
        Formula: Swing Time = t(HS_i+1) - t(TO_i)
        """
        heel_strikes = self.heel_strikes[foot]
        toe_offs = self.toe_offs[foot]
        
        swing_times = []
        
        # Match each toe-off with the next heel strike of the same foot
        for to_time in toe_offs:
            # Find the next heel strike after this toe-off
            next_heel_strikes = heel_strikes[heel_strikes > to_time]
            if len(next_heel_strikes) > 0:
                hs_time = next_heel_strikes[0]
                swing_time = hs_time - to_time
                swing_times.append(swing_time)
        
        return np.array(swing_times)

    
    def analyze_gait(self, force_threshold=50):
        """
        Complete gait analysis pipeline.
        Returns dictionary with all computed metrics.
        """
        print("Detecting gait events...")
        
        # Detect heel strikes and toe-offs for both feet
        self.detect_heel_strikes_toe_offs('left', force_threshold)
        self.detect_heel_strikes_toe_offs('right', force_threshold)
        
        print(f"Left foot - Heel strikes: {len(self.heel_strikes['left'])}, Toe-offs: {len(self.toe_offs['left'])}")
        print(f"Right foot - Heel strikes: {len(self.heel_strikes['right'])}, Toe-offs: {len(self.toe_offs['right'])}")
        
        # Calculate metrics
        results = {
            'left_stride_times': self.calculate_stride_time('left'),
            'right_stride_times': self.calculate_stride_time('right'),
            'left_stance_times': self.calculate_stance_time('left'),
            'right_stance_times': self.calculate_stance_time('right'),
            'left_swing_times': self.calculate_swing_time('left'),
            'right_swing_times': self.calculate_swing_time('right')
        }
        
        return results
    
    def print_summary(self, results):
        """Print summary statistics for all metrics."""
        print("\n" + "="*50)
        print("GAIT METRICS SUMMARY")
        print("="*50)
        
        metrics = [
            ('Left Stride Time', results['left_stride_times']),
            ('Right Stride Time', results['right_stride_times']),
            ('Left Stance Time', results['left_stance_times']),
            ('Right Stance Time', results['right_stance_times']),
            ('Left Swing Time', results['left_swing_times']),
            ('Right Swing Time', results['right_swing_times'])
        ]
        
        for name, values in metrics:
            if len(values) > 0:
                print(f"\n{name}:")
                print(f"  Mean: {np.mean(values):.3f} seconds")
                print(f"  Std:  {np.std(values):.3f} seconds")
                print(f"  Count: {len(values)} cycles")
            else:
                print(f"\n{name}: No valid cycles detected")
    


# # Example usage
# def main():
#     # Initialize calculator with your data file
#     calculator = GaitMetricsCalculator('dataset\SiCo01_01.txt')
    
#     # Run complete analysis
#     results = calculator.analyze_gait(force_threshold=20)
    
#     # Print summary
#     calculator.print_summary(results)

    
#     return results

# # Uncomment to run:
# results = main()