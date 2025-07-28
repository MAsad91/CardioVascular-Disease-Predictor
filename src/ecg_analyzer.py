import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.ndimage import gaussian_filter1d

class ECGAnalyzer:
    """
    Class for analyzing ECG/EKG images and extracting key cardiac features
    such as QT intervals, ST elevation, heart rate, and rhythm analysis.
    """
    
    def __init__(self):
        """Initialize the ECG analyzer with default parameters"""
        # Common ECG parameters (can be adjusted based on image calibration)
        self.time_scale = 0.04  # 40ms per small square (standard)
        self.voltage_scale = 0.1  # 0.1mV per small square (standard)
    
    def preprocess_ecg_image(self, image):
        """
        Preprocess the ECG image to isolate the graph lines
        
        Parameters:
        - image: Path to the ECG image or OpenCV image array
        
        Returns:
        - Preprocessed image and grayscale original
        """
        # Handle input
        if isinstance(image, str):
            # Load image from path
            img = cv2.imread(image)
            if img is None:
                raise ValueError(f"Unable to load image from {image}")
        elif isinstance(image, np.ndarray):
            # Use provided image array
            img = image.copy()
        else:
            raise ValueError("Input must be either a file path or an OpenCV image array")
        
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Apply adaptive thresholding to handle varying brightness
        thresh = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY_INV, 11, 2
        )
        
        # Remove grid lines (which are usually lighter than ECG trace)
        kernel = np.ones((2, 2), np.uint8)
        morphed = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        
        # Find contours (the ECG signal lines)
        contours, _ = cv2.findContours(morphed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Create mask for the ECG lines
        mask = np.zeros_like(gray)
        for contour in contours:
            # Filter out small contours (noise)
            if cv2.contourArea(contour) > 50:
                cv2.drawContours(mask, [contour], -1, 255, 1)
        
        return mask, gray
    
    def extract_signal_from_image(self, image):
        """
        Extract ECG signal from image as time series data
        
        Parameters:
        - image: Path to the ECG image or OpenCV image array
        
        Returns:
        - Dictionary with extracted signals for each lead (if multiple leads are detected)
        """
        mask, gray = self.preprocess_ecg_image(image)
        
        # Get image dimensions
        height, width = mask.shape
        
        # Analyze the image to detect separate leads
        # This is a simplified approach that assumes horizontal separation of leads
        lead_regions = self._detect_lead_regions(mask)
        
        signals = {}
        for i, region in enumerate(lead_regions):
            y_start, y_end = region
            
            # Extract the region for this lead
            lead_image = mask[y_start:y_end, :]
            
            # For each column, find the y-coordinate of the signal line
            signal_data = []
            for x in range(width):
                column = lead_image[:, x]
                # Find the position of the signal line
                white_pixels = np.where(column > 0)[0]
                if len(white_pixels) > 0:
                    # Use the midpoint if multiple points are found
                    y_coord = np.mean(white_pixels)
                    # Invert y-coordinate (image has origin at top-left)
                    # and normalize to height of the region
                    normalized_y = 1 - (y_coord / (y_end - y_start))
                    signal_data.append(normalized_y)
                else:
                    # No signal point in this column
                    signal_data.append(np.nan)
            
            # Store the signal
            lead_name = f"Lead-{i+1}"
            signals[lead_name] = np.array(signal_data)
            
            # Interpolate missing values
            signals[lead_name] = self._interpolate_signal(signals[lead_name])
            
            # Apply smoothing to reduce noise
            signals[lead_name] = gaussian_filter1d(signals[lead_name], sigma=2)
        
        return signals
    
    def _detect_lead_regions(self, image):
        """
        Detect regions in the image corresponding to different ECG leads
        
        Parameters:
        - image: Preprocessed binary image
        
        Returns:
        - List of (y_start, y_end) tuples for each detected lead region
        """
        # Simple approach: divide the image into equal segments (for multiple leads)
        height, _ = image.shape
        
        # Project the image onto the y-axis to find signal-dense regions
        y_projection = np.sum(image, axis=1)
        
        # Threshold to find where ECG traces are present
        threshold = np.max(y_projection) * 0.1
        signal_rows = y_projection > threshold
        
        # Find contiguous regions of signal
        regions = []
        in_region = False
        start = 0
        
        for i, has_signal in enumerate(signal_rows):
            if has_signal and not in_region:
                # Start of a new region
                in_region = True
                start = i
            elif not has_signal and in_region:
                # End of a region
                in_region = False
                # Only add if the region is large enough
                if i - start > 20:  # Minimum height threshold
                    regions.append((start, i))
        
        # Check if last region extends to the end
        if in_region:
            regions.append((start, height))
        
        # If no regions found, use the entire image
        if not regions:
            regions = [(0, height)]
        
        return regions
    
    def _interpolate_signal(self, signal):
        """Interpolate missing values in the signal"""
        mask = np.isnan(signal)
        signal[mask] = np.interp(
            np.flatnonzero(mask), 
            np.flatnonzero(~mask), 
            signal[~mask]
        )
        return signal
    
    def detect_r_peaks(self, signal):
        """
        Detect R peaks in the ECG signal
        
        Parameters:
        - signal: 1D numpy array of the ECG signal
        
        Returns:
        - Array of indices where R peaks occur
        """
        # Use find_peaks to locate R peaks
        r_peaks, _ = signal.find_peaks(signal, height=0.5, distance=20)
        return r_peaks
    
    def calculate_heart_rate(self, r_peaks, signal_length, image_width):
        """
        Calculate heart rate from R peak intervals
        
        Parameters:
        - r_peaks: Array of R peak indices
        - signal_length: Length of the signal array
        - image_width: Width of the original image in pixels
        
        Returns:
        - Estimated heart rate in BPM
        """
        if len(r_peaks) < 2:
            return None
        
        # Calculate average R-R interval
        rr_intervals = np.diff(r_peaks)
        avg_rr_interval = np.mean(rr_intervals)
        
        # Convert to seconds using time scale
        pixels_per_second = image_width / (signal_length * self.time_scale)
        rr_seconds = avg_rr_interval / pixels_per_second
        
        # Calculate heart rate
        heart_rate = 60 / rr_seconds
        
        return heart_rate
    
    def detect_qt_interval(self, signal, r_peaks):
        """
        Detect QT intervals in the ECG signal
        
        Parameters:
        - signal: 1D numpy array of the ECG signal
        - r_peaks: Array of R peak indices
        
        Returns:
        - Average QT interval in signal samples
        """
        qt_intervals = []
        
        # For each R peak, try to find the Q point before it and T wave end after it
        for r_idx in r_peaks:
            # Look for Q point (minimum before R peak)
            # Search in a window before R peak
            q_search_start = max(0, r_idx - 20)
            q_search_end = r_idx
            
            if q_search_start < q_search_end:
                q_idx = q_search_start + np.argmin(signal[q_search_start:q_search_end])
            else:
                continue
            
            # Look for T wave end after the R peak
            # First find the T peak (local maximum after R)
            t_search_start = r_idx + 15  # Skip the S wave
            t_search_end = min(len(signal), r_idx + 100)  # Reasonable window for T wave
            
            if t_search_start < t_search_end:
                # Find local maximum after R peak (probable T wave peak)
                t_peak_idx = t_search_start + np.argmax(signal[t_search_start:t_search_end])
                
                # T wave end is where the slope changes after T peak
                t_end_search_start = t_peak_idx
                t_end_search_end = min(len(signal), t_peak_idx + 40)
                
                if t_end_search_start < t_end_search_end:
                    # Calculate first derivative
                    diff = np.diff(signal[t_end_search_start:t_end_search_end])
                    # Find where it crosses zero or reaches minimum slope
                    zero_crosses = np.where(np.diff(np.signbit(diff)))[0]
                    
                    if len(zero_crosses) > 0:
                        t_end_idx = t_end_search_start + zero_crosses[0] + 1
                        # Calculate QT interval
                        qt_intervals.append(t_end_idx - q_idx)
            
        # Return average QT interval if any were found
        if qt_intervals:
            return np.mean(qt_intervals)
        else:
            return None
    
    def detect_st_elevation(self, signal, r_peaks):
        """
        Detect ST segment elevation or depression
        
        Parameters:
        - signal: 1D numpy array of the ECG signal
        - r_peaks: Array of R peak indices
        
        Returns:
        - Average ST segment deviation (positive for elevation, negative for depression)
        """
        st_deviations = []
        
        # For each R peak, try to find the ST segment
        for r_idx in r_peaks:
            # Define J point (end of QRS complex, beginning of ST segment)
            # Typically about 80ms after R peak
            j_point_idx = min(len(signal) - 1, r_idx + 15)  # Approximation
            
            # Get baseline level (PR segment)
            pr_segment_idx = max(0, r_idx - 30)  # Approximation of PR segment
            baseline = signal[pr_segment_idx]
            
            # Measure ST segment 80ms after J point
            st_point_idx = min(len(signal) - 1, j_point_idx + 15)  # ~80ms after J point
            
            # Calculate deviation of ST segment from baseline
            st_deviation = signal[st_point_idx] - baseline
            
            # Convert to mV using voltage scale
            st_deviation_mv = st_deviation * self.voltage_scale
            
            st_deviations.append(st_deviation_mv)
        
        # Return average ST deviation if any were found
        if st_deviations:
            return np.mean(st_deviations)
        else:
            return None
    
    def analyze_ecg(self, image):
        """
        Perform comprehensive ECG analysis
        
        Parameters:
        - image: Path to the ECG image or OpenCV image array
        
        Returns:
        - Dictionary with analysis results for each detected lead
        """
        # Extract signals from the image
        signals = self.extract_signal_from_image(image)
        
        # Get image dimensions
        if isinstance(image, str):
            img = cv2.imread(image)
            if img is None:
                raise ValueError(f"Unable to load image from {image}")
        else:
            img = image
        
        image_width = img.shape[1]
        results = {}
        
        for lead_name, signal_data in signals.items():
            # Detect R peaks
            r_peaks = self.detect_r_peaks(signal_data)
            
            lead_results = {}
            
            # Calculate heart rate
            lead_results['heart_rate'] = self.calculate_heart_rate(
                r_peaks, len(signal_data), image_width
            )
            
            # Detect QT interval
            qt_samples = self.detect_qt_interval(signal_data, r_peaks)
            if qt_samples:
                # Convert to seconds
                qt_seconds = qt_samples * self.time_scale / (image_width / len(signal_data))
                lead_results['qt_interval'] = qt_seconds
                
                # Calculate corrected QT (QTc) using Bazett's formula
                if lead_results['heart_rate']:
                    rr_seconds = 60 / lead_results['heart_rate']
                    lead_results['qtc_interval'] = qt_seconds / np.sqrt(rr_seconds)
                else:
                    lead_results['qtc_interval'] = None
            else:
                lead_results['qt_interval'] = None
                lead_results['qtc_interval'] = None
            
            # Detect ST elevation
            lead_results['st_deviation'] = self.detect_st_elevation(signal_data, r_peaks)
            
            # Store results for this lead
            results[lead_name] = lead_results
        
        return results
    
    def visualize_analysis(self, image_path, save_path=None):
        """
        Visualize the ECG analysis with annotations
        
        Parameters:
        - image_path: Path to the ECG image
        - save_path: Optional path to save the visualization
        
        Returns:
        - Visualization figure
        """
        # Extract signals
        signals = self.extract_signal_from_image(image_path)
        
        # Create figure with subplots for each lead
        n_leads = len(signals)
        fig, axes = plt.subplots(n_leads, 1, figsize=(12, 3 * n_leads))
        
        # If only one lead, make axes iterable
        if n_leads == 1:
            axes = [axes]
        
        # Get image width for time calculations
        img = cv2.imread(image_path)
        image_width = img.shape[1]
        
        results = {}
        
        for i, (lead_name, signal_data) in enumerate(signals.items()):
            ax = axes[i]
            
            # Plot the signal
            ax.plot(signal_data, color='blue', linewidth=1)
            ax.set_title(f"{lead_name} Analysis")
            
            # Detect R peaks
            r_peaks = self.detect_r_peaks(signal_data)
            
            # Mark R peaks
            ax.scatter(r_peaks, signal_data[r_peaks], color='red', s=30, zorder=3,
                       label='R peaks')
            
            # Calculate heart rate
            heart_rate = self.calculate_heart_rate(r_peaks, len(signal_data), image_width)
            
            # Detect QT interval
            qt_samples = self.detect_qt_interval(signal_data, r_peaks)
            
            # Detect ST elevation
            st_deviation = self.detect_st_elevation(signal_data, r_peaks)
            
            # Annotate with metrics
            metrics_text = f"Heart Rate: {heart_rate:.1f} BPM\n" if heart_rate else "Heart Rate: N/A\n"
            
            if qt_samples:
                qt_seconds = qt_samples * self.time_scale / (image_width / len(signal_data))
                metrics_text += f"QT interval: {qt_seconds*1000:.1f} ms\n"
                
                if heart_rate:
                    rr_seconds = 60 / heart_rate
                    qtc = qt_seconds / np.sqrt(rr_seconds)
                    metrics_text += f"QTc interval: {qtc*1000:.1f} ms\n"
            else:
                metrics_text += "QT interval: N/A\n"
                metrics_text += "QTc interval: N/A\n"
            
            if st_deviation:
                metrics_text += f"ST deviation: {st_deviation*1000:.2f} mV"
            else:
                metrics_text += "ST deviation: N/A"
            
            ax.text(0.02, 0.05, metrics_text, transform=ax.transAxes, 
                   bbox=dict(facecolor='white', alpha=0.8))
            
            # Save results
            lead_results = {
                'heart_rate': heart_rate,
                'qt_interval': qt_samples * self.time_scale / (image_width / len(signal_data)) if qt_samples else None,
                'qt_interval_ms': qt_samples * self.time_scale * 1000 / (image_width / len(signal_data)) if qt_samples else None,
                'st_deviation': st_deviation,
                'st_deviation_mv': st_deviation * 1000 if st_deviation else None
            }
            
            if heart_rate and qt_samples:
                rr_seconds = 60 / heart_rate
                lead_results['qtc_interval'] = lead_results['qt_interval'] / np.sqrt(rr_seconds)
                lead_results['qtc_interval_ms'] = lead_results['qtc_interval'] * 1000
            
            results[lead_name] = lead_results
        
        # Adjust layout
        plt.tight_layout()
        
        # Save if requested
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        return fig, results 