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
    
    def calculate_double_support_time(self):
        """
        Calculate double support time: time when both feet are on ground.
        Formula: Double Support Time = t(TO_opposite_foot) - t(HS_initial_foot)
        """
        left_hs = self.heel_strikes['left']
        right_hs = self.heel_strikes['right']
        left_to = self.toe_offs['left']
        right_to = self.toe_offs['right']
        
        double_support_times = []
        
        # Double support phase 1: Right heel strike to left toe-off
        for rhs_time in right_hs:
            # Find next left toe-off after right heel strike
            next_left_to = left_to[left_to > rhs_time]
            if len(next_left_to) > 0:
                lto_time = next_left_to[0]
                ds_time = lto_time - rhs_time
                if ds_time > 0:  # Valid double support phase
                    double_support_times.append(ds_time)
        
        # Double support phase 2: Left heel strike to right toe-off
        for lhs_time in left_hs:
            # Find next right toe-off after left heel strike
            next_right_to = right_to[right_to > lhs_time]
            if len(next_right_to) > 0:
                rto_time = next_right_to[0]
                ds_time = rto_time - lhs_time
                if ds_time > 0:  # Valid double support phase
                    double_support_times.append(ds_time)
        
        return np.array(double_support_times)
    
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
            'right_swing_times': self.calculate_swing_time('right'),
            'double_support_times': self.calculate_double_support_time()
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
            ('Right Swing Time', results['right_swing_times']),
            ('Double Support Time', results['double_support_times'])
        ]
        
        for name, values in metrics:
            if len(values) > 0:
                print(f"\n{name}:")
                print(f"  Mean: {np.mean(values):.3f} seconds")
                print(f"  Std:  {np.std(values):.3f} seconds")
                print(f"  Count: {len(values)} cycles")
            else:
                print(f"\n{name}: No valid cycles detected")
    
    def plot_force_and_events(self, duration=10):
        """
        Plot force data with detected heel strikes and toe-offs.
        Shows first 'duration' seconds of data.
        """
        # Limit to first 'duration' seconds
        mask = self.data['time'] <= duration
        time_subset = self.data['time'][mask]
        left_force = self.data['total_left'][mask]
        right_force = self.data['total_right'][mask]
        
        plt.figure(figsize=(14, 8))
        
        # Plot force data
        plt.plot(time_subset, left_force, 'b-', label='Left Foot Total Force', linewidth=1.5, alpha=0.8)
        plt.plot(time_subset, right_force, 'r-', label='Right Foot Total Force', linewidth=1.5, alpha=0.8)
        
        # Plot heel strikes and toe-offs within the time window at the actual force values
        for foot, color in [('left', 'blue'), ('right', 'red')]:
            hs_in_window = self.heel_strikes[foot][self.heel_strikes[foot] <= duration]
            to_in_window = self.toe_offs[foot][self.toe_offs[foot] <= duration]
            
            # For heel strikes, plot at the force value at that time
            if len(hs_in_window) > 0:
                hs_forces = []
                for hs_time in hs_in_window:
                    # Find closest time point
                    closest_idx = np.argmin(np.abs(time_subset - hs_time))
                    if foot == 'left':
                        hs_forces.append(left_force.iloc[closest_idx])
                    else:
                        hs_forces.append(right_force.iloc[closest_idx])
                
                plt.scatter(hs_in_window, hs_forces, 
                           c=color, marker='o', s=120, edgecolor='white', linewidth=2,
                           label=f'{foot.capitalize()} Heel Strike', zorder=5)
            
            # For toe-offs, plot at the force value at that time
            if len(to_in_window) > 0:
                to_forces = []
                for to_time in to_in_window:
                    # Find closest time point
                    closest_idx = np.argmin(np.abs(time_subset - to_time))
                    if foot == 'left':
                        to_forces.append(left_force.iloc[closest_idx])
                    else:
                        to_forces.append(right_force.iloc[closest_idx])
                
                plt.scatter(to_in_window, to_forces, 
                           c=color, marker='^', s=120, edgecolor='white', linewidth=2,
                           label=f'{foot.capitalize()} Toe-Off', zorder=5)
        
        # Add horizontal line at threshold
        plt.axhline(y=50, color='gray', linestyle='--', alpha=0.5, label='Force Threshold (50N)')
        
        plt.xlabel('Time (seconds)', fontsize=12)
        plt.ylabel('Force (Newtons)', fontsize=12)
        plt.title(f'Ground Reaction Forces with Gait Events (First {duration} seconds)', fontsize=14)
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()


# Example usage
def main():
    # Initialize calculator with your data file
    calculator = GaitMetricsCalculator('dataset\SiCo01_01.txt')
    
    # Run complete analysis
    results = calculator.analyze_gait(force_threshold=20)
    
    # Print summary
    calculator.print_summary(results)
    
    # Plot results (optional)
    calculator.plot_force_and_events(duration=10)
    
    return results

# Uncomment to run:
results = main()