import sys
import numpy as np
from scipy import stats
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QLabel, QDoubleSpinBox, QFormLayout,
                             QSpinBox, QPushButton, QTableWidget, QTableWidgetItem,
                             QCheckBox, QTabWidget)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap, QIcon, QPainter, QFont, QColor, QIcon, QPainter, QFont, QColor
import ctypes
from ctypes import wintypes
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from io import BytesIO

# Global font size for section titles
title_fontsize = 24
# Global font size for widget text (input fields, labels, etc.)
widget_text_size = 18
# Global font size for LaTeX rendered text
latex_fontsize = 15


def latex_to_pixmap(latex_str, fontsize=10):
    """Convert LaTeX/mathtext string to QPixmap"""
    # Use taller figure for expressions with sqrt to prevent cutoff
    has_sqrt = '\\sqrt' in latex_str
    fig_height = 0.8 if has_sqrt else 0.5
    fig = Figure(figsize=(4, fig_height))
    fig.patch.set_facecolor('none')
    ax = fig.add_subplot(111)
    ax.axis('off')
    # Adjust vertical position for sqrt expressions
    y_pos = 0.5 if not has_sqrt else 0.55
    ax.text(0.0, y_pos, latex_str, transform=ax.transAxes,
            fontsize=fontsize, ha='left', va='center')
    
    # Convert to pixmap
    canvas = FigureCanvas(fig)
    canvas.draw()
    
    # Get the image as bytes
    buf = BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', 
                facecolor='none', edgecolor='none', dpi=100, pad_inches=0.1)
    buf.seek(0)
    
    # Convert to QPixmap
    pixmap = QPixmap()
    pixmap.loadFromData(buf.read())
    buf.close()
    return pixmap


class NormalDistributionPlot(QWidget):
    def __init__(self):
        super().__init__()
        self.mean = 0.0
        self.std = 1.0
        # Store list of estimated distributions: [(mean1, std1), (mean2, std2), ...]
        self.estimated_distributions = []
        
        # Create matplotlib figure (no fixed size - will resize with widget)
        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)
        self.ax = self.figure.add_subplot(111)
        
        # Create layout with no margins to maximize space
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.canvas)
        self.setLayout(layout)
        
        # Initial plot
        self.update_plot()
    
    def update_plot(self):
        """Update the plot with current mean and std values"""
        self.ax.clear()
        
        # Calculate range: always use true distribution range (fixed axes)
        x_min = -5 #self.mean - 5 * self.std
        x_max = 5 # self.mean + 5 * self.std
        
        # Create linspace with 2000 steps
        x = np.linspace(x_min, x_max, 2000)
        
        # Calculate and plot true normal distribution (normalized PDF)
        y_true = (1 / (self.std * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - self.mean) / self.std) ** 2)
        self.ax.plot(x, y_true, 'b-', linewidth=2, label=f'True: μ={self.mean:.2f}, σ={self.std:.2f}')
        
        # Plot all estimated distributions
        colors = ['r', 'g', 'm', 'c', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive']
        for i, (est_mean, est_std) in enumerate(self.estimated_distributions):
            y_estimated = (1 / (est_std * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - est_mean) / est_std) ** 2)
            color = colors[i % len(colors)]
            self.ax.plot(x, y_estimated, '--', color=color, linewidth=1.5, 
                        label=f'N({est_mean:.2f}, {est_std:.2f})', alpha=0.8)
        
        # Set fixed x-axis limits based on true distribution
        self.ax.set_xlim(x_min, x_max)
        
        # Labels and title
        self.ax.set_xlabel('x', fontsize=12)
        self.ax.set_ylabel('Wahrscheinlichkeitsdichte', fontsize=12)
        self.ax.set_title('Verteilungen', fontsize=14, fontweight='bold')
        self.ax.grid(True, alpha=0.3)
        self.ax.legend()
        
        # Adjust layout to prevent text from being cut off
        self.figure.tight_layout(rect=[0, 0, 1, 0.96])
        self.canvas.draw()
    
    def set_parameters(self, mean, std):
        """Update mean and standard deviation"""
        self.mean = mean
        self.std = max(0.01, std)  # Prevent division by zero
        self.update_plot()
    
    def add_estimated_distribution(self, estimated_mean, estimated_std):
        """Add a new estimated distribution to the plot"""
        if estimated_mean is not None and estimated_std is not None:
            self.estimated_distributions.append((estimated_mean, max(0.01, estimated_std)))
            self.update_plot()
    
    def clear_estimated_distributions(self):
        """Clear all estimated distributions"""
        self.estimated_distributions = []
        self.update_plot()
    
    def get_x_limits(self):
        """Get the x-axis limits"""
        x_min = self.mean - 5 * self.std
        x_max = self.mean + 5 * self.std
        return x_min, x_max


class StandardNormalPlot(QWidget):
    """Plot widget for standard normal distribution PDF and CDF"""
    def __init__(self):
        super().__init__()
        
        # Create matplotlib figure with two subplots
        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)
        self.ax1, self.ax2 = self.figure.subplots(2, 1, sharex=True, 
                                                   gridspec_kw={'hspace': 0.3})
        
        # Store current z and alpha values
        self.current_z = None
        self.current_alpha = None
        
        # Create layout with no margins to maximize space
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.canvas)
        self.setLayout(layout)
        
        # Initial plot
        self.update_plot()
    
    def set_z_value(self, z):
        """Set the z value to display as vertical line"""
        self.current_z = z
        self.update_plot()
    
    def set_alpha_value(self, alpha):
        """Set the alpha value to display as filled area"""
        self.current_alpha = alpha
        self.update_plot()
    
    def update_plot(self):
        """Update the plot with standard normal PDF and CDF"""
        self.ax1.clear()
        self.ax2.clear()
        
        # Calculate range for standard normal
        x_min = -4
        x_max = 4
        x = np.linspace(x_min, x_max, 1000)
        
        # Calculate PDF: φ(z) = (1/√(2π)) * exp(-z²/2)
        pdf = (1 / np.sqrt(2 * np.pi)) * np.exp(-0.5 * x ** 2)
        
        # Calculate CDF: Φ(z) using scipy
        cdf = stats.norm.cdf(x)
        
        # Plot PDF on top
        self.ax1.plot(x, pdf, 'b-', linewidth=2, label=r'$\phi(z)$')
        
        # Fill area for quantile if alpha is set
        if self.current_alpha is not None:
            quantile_z = stats.norm.ppf(self.current_alpha)
            # Fill area from x_min to quantile_z
            x_fill = np.linspace(x_min, quantile_z, 500)
            pdf_fill = (1 / np.sqrt(2 * np.pi)) * np.exp(-0.5 * x_fill ** 2)
            self.ax1.fill_between(x_fill, 0, pdf_fill, alpha=0.3, color='green', 
                                 label=f'Quantil: α={self.current_alpha:.4f}')
        
        # Draw vertical line for z value if set
        if self.current_z is not None:
            self.ax1.axvline(self.current_z, color='red', linestyle='--', linewidth=2, 
                           label=f'z={self.current_z:.4f}')
        
        self.ax1.set_ylabel(r'$\phi(z)$', fontsize=12)
        self.ax1.set_title('Standardnormalverteilung', fontsize=14, fontweight='bold', pad=10)
        self.ax1.set_ylim(0, 0.42)
        self.ax1.grid(True, alpha=0.3)
        self.ax1.legend()
        
        # Plot CDF on bottom
        self.ax2.plot(x, cdf, 'r-', linewidth=2, label=r'$\Phi(z)$')
        
        # Fill area for quantile in CDF plot if alpha is set
        if self.current_alpha is not None:
            quantile_z = stats.norm.ppf(self.current_alpha)
            # Fill area from x_min to quantile_z
            x_fill = np.linspace(x_min, quantile_z, 500)
            cdf_fill = stats.norm.cdf(x_fill)
            self.ax2.fill_between(x_fill, 0, cdf_fill, alpha=0.3, color='green',
                                 label=f'Quantil: α={self.current_alpha:.4f}')
        
        # Draw vertical line for z value if set
        if self.current_z is not None:
            self.ax2.axvline(self.current_z, color='red', linestyle='--', linewidth=2,
                           label=f'z={self.current_z:.4f}')
        
        self.ax2.set_xlabel('z', fontsize=12)
        self.ax2.set_ylabel(r'$\Phi(z)$', fontsize=12)
        self.ax2.set_title('Verteilungsfunktion', fontsize=14, fontweight='bold', pad=10)
        self.ax2.grid(True, alpha=0.3)
        self.ax2.legend()
        self.ax2.set_ylim(0, 1)
        
        # Set shared x-axis limits
        self.ax1.set_xlim(x_min, x_max)
        
        # Adjust layout
        self.figure.tight_layout(rect=[0, 0, 1, 0.98])
        self.canvas.draw()


class StudentTPlot(QWidget):
    """Plot widget for Student-t distribution PDF and CDF"""
    def __init__(self):
        super().__init__()
        
        # Create matplotlib figure with two subplots
        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)
        self.ax1, self.ax2 = self.figure.subplots(2, 1, sharex=True, 
                                                   gridspec_kw={'hspace': 0.3})
        
        # Store current values
        self.df = 1  # degrees of freedom
        self.current_z = None
        self.current_alpha = None
        
        # Create layout with no margins to maximize space
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.canvas)
        self.setLayout(layout)
        
        # Initial plot
        self.update_plot()
    
    def set_degrees_of_freedom(self, df):
        """Set the degrees of freedom"""
        self.df = max(1, int(df))
        self.update_plot()
    
    def set_z_value(self, z):
        """Set the z value to display as vertical line"""
        self.current_z = z
        self.update_plot()
    
    def set_alpha_value(self, alpha):
        """Set the alpha value to display as filled area"""
        self.current_alpha = alpha
        self.update_plot()
    
    def update_plot(self):
        """Update the plot with Student-t PDF and CDF"""
        self.ax1.clear()
        self.ax2.clear()
        
        # Calculate range for Student-t (wider range than normal)
        x_min = -5
        x_max = 5
        x = np.linspace(x_min, x_max, 1000)
        
        # Calculate PDF and CDF using scipy
        pdf = stats.t.pdf(x, self.df)
        cdf = stats.t.cdf(x, self.df)
        
        # Find max PDF value for y-axis limit
        max_pdf = np.max(pdf)
        y_max = min(0.42, max_pdf * 1.1)
        
        # Plot PDF on top
        self.ax1.plot(x, pdf, 'b-', linewidth=2, label=f'Student-t (ν={self.df})')
        
        # Fill area for quantile if alpha is set
        if self.current_alpha is not None:
            quantile_z = stats.t.ppf(self.current_alpha, self.df)
            # Fill area from x_min to quantile_z
            x_fill = np.linspace(x_min, quantile_z, 500)
            pdf_fill = stats.t.pdf(x_fill, self.df)
            self.ax1.fill_between(x_fill, 0, pdf_fill, alpha=0.3, color='green', 
                                 label=f'Quantil: α={self.current_alpha:.4f}')
        
        # Draw vertical line for z value if set
        if self.current_z is not None:
            self.ax1.axvline(self.current_z, color='red', linestyle='--', linewidth=2, 
                           label=f't={self.current_z:.4f}')
        
        self.ax1.set_ylabel('PDF', fontsize=12)
        self.ax1.set_title(f'Student-t Verteilung (ν={self.df})', fontsize=14, fontweight='bold', pad=10)
        self.ax1.set_ylim(0, y_max)
        self.ax1.grid(True, alpha=0.3)
        self.ax1.legend()
        
        # Plot CDF on bottom
        self.ax2.plot(x, cdf, 'r-', linewidth=2, label=f'CDF (ν={self.df})')
        
        # Fill area for quantile in CDF plot if alpha is set
        if self.current_alpha is not None:
            quantile_z = stats.t.ppf(self.current_alpha, self.df)
            # Fill area from x_min to quantile_z
            x_fill = np.linspace(x_min, quantile_z, 500)
            cdf_fill = stats.t.cdf(x_fill, self.df)
            self.ax2.fill_between(x_fill, 0, cdf_fill, alpha=0.3, color='green',
                                 label=f'Quantil: α={self.current_alpha:.4f}')
        
        # Draw vertical line for z value if set
        if self.current_z is not None:
            self.ax2.axvline(self.current_z, color='red', linestyle='--', linewidth=2,
                           label=f't={self.current_z:.4f}')
        
        self.ax2.set_xlabel('t', fontsize=12)
        self.ax2.set_ylabel('CDF', fontsize=12)
        self.ax2.set_title('Verteilungsfunktion', fontsize=14, fontweight='bold', pad=10)
        self.ax2.grid(True, alpha=0.3)
        self.ax2.legend()
        self.ax2.set_ylim(0, 1)
        
        # Set shared x-axis limits
        self.ax1.set_xlim(x_min, x_max)
        
        # Adjust layout
        self.figure.tight_layout(rect=[0, 0, 1, 0.98])
        self.canvas.draw()


class ChiSquaredPlot(QWidget):
    """Plot widget for Chi-squared distribution PDF and CDF"""
    def __init__(self):
        super().__init__()
        
        # Create matplotlib figure with two subplots
        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)
        self.ax1, self.ax2 = self.figure.subplots(2, 1, sharex=True, 
                                                   gridspec_kw={'hspace': 0.3})
        
        # Store current values
        self.df = 1  # degrees of freedom
        self.current_z = None
        self.current_alpha = None
        
        # Create layout with no margins to maximize space
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.canvas)
        self.setLayout(layout)
        
        # Initial plot
        self.update_plot()
    
    def set_degrees_of_freedom(self, df):
        """Set the degrees of freedom"""
        self.df = max(1, int(df))
        self.update_plot()
    
    def set_z_value(self, z):
        """Set the z value to display as vertical line"""
        self.current_z = z
        self.update_plot()
    
    def set_alpha_value(self, alpha):
        """Set the alpha value to display as filled area"""
        self.current_alpha = alpha
        self.update_plot()
    
    def update_plot(self):
        """Update the plot with Chi-squared PDF and CDF"""
        self.ax1.clear()
        self.ax2.clear()
        
        # Calculate range for Chi-squared (starts at 0, extends based on df)
        x_min = 0
        x_max = max(20, self.df + 5 * np.sqrt(2 * self.df))
        x = np.linspace(x_min, x_max, 1000)
        
        # Calculate PDF and CDF using scipy
        pdf = stats.chi2.pdf(x, self.df)
        cdf = stats.chi2.cdf(x, self.df)
        
        # Find max PDF value for y-axis limit
        max_pdf = np.max(pdf)
        y_max = min(0.42, max_pdf * 1.1)
        
        # Plot PDF on top
        self.ax1.plot(x, pdf, 'b-', linewidth=2, label=f'χ² (ν={self.df})')
        
        # Fill area for quantile if alpha is set
        if self.current_alpha is not None:
            quantile_z = stats.chi2.ppf(self.current_alpha, self.df)
            # Fill area from x_min to quantile_z
            x_fill = np.linspace(x_min, quantile_z, 500)
            pdf_fill = stats.chi2.pdf(x_fill, self.df)
            self.ax1.fill_between(x_fill, 0, pdf_fill, alpha=0.3, color='green', 
                                 label=f'Quantil: α={self.current_alpha:.4f}')
        
        # Draw vertical line for z value if set
        if self.current_z is not None:
            self.ax1.axvline(self.current_z, color='red', linestyle='--', linewidth=2, 
                           label=f'χ²={self.current_z:.4f}')
        
        self.ax1.set_ylabel('PDF', fontsize=12)
        self.ax1.set_title(f'χ² Verteilung (ν={self.df})', fontsize=14, fontweight='bold', pad=10)
        self.ax1.set_ylim(0, y_max)
        self.ax1.grid(True, alpha=0.3)
        self.ax1.legend()
        
        # Plot CDF on bottom
        self.ax2.plot(x, cdf, 'r-', linewidth=2, label=f'CDF (ν={self.df})')
        
        # Fill area for quantile in CDF plot if alpha is set
        if self.current_alpha is not None:
            quantile_z = stats.chi2.ppf(self.current_alpha, self.df)
            # Fill area from x_min to quantile_z
            x_fill = np.linspace(x_min, quantile_z, 500)
            cdf_fill = stats.chi2.cdf(x_fill, self.df)
            self.ax2.fill_between(x_fill, 0, cdf_fill, alpha=0.3, color='green',
                                 label=f'Quantil: α={self.current_alpha:.4f}')
        
        # Draw vertical line for z value if set
        if self.current_z is not None:
            self.ax2.axvline(self.current_z, color='red', linestyle='--', linewidth=2,
                           label=f'χ²={self.current_z:.4f}')
        
        self.ax2.set_xlabel('χ²', fontsize=12)
        self.ax2.set_ylabel('CDF', fontsize=12)
        self.ax2.set_title('Verteilungsfunktion', fontsize=14, fontweight='bold', pad=10)
        self.ax2.grid(True, alpha=0.3)
        self.ax2.legend()
        self.ax2.set_ylim(0, 1)
        
        # Set shared x-axis limits
        self.ax1.set_xlim(x_min, x_max)
        
        # Adjust layout
        self.figure.tight_layout(rect=[0, 0, 1, 0.98])
        self.canvas.draw()


class CombinedPlot(QWidget):
    def __init__(self):
        super().__init__()
        self.mean = 0.0
        self.std = 1.0
        self.true_mean = 0.0
        self.x_min = -5.0
        self.x_max = 5.0
        self.confidence_level = 95.0
        # Store list of estimated distributions: [(mean1, std1), (mean2, std2), ...]
        self.estimated_distributions = []
        # Store list of confidence intervals: [(mean, ci_lower, ci_upper, n, confidence_level), ...]
        self.confidence_intervals = []
        
        # Create matplotlib figure with subplots sharing x-axis
        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)
        # Create three subplots: distribution, confidence intervals, and bar chart
        # First create all subplots, then share x-axis only between first two
        self.ax1, self.ax2, self.ax3 = self.figure.subplots(3, 1,
                                                             gridspec_kw={'hspace': 0.4, 'height_ratios': [2, 2, 1]})
        # Share x-axis only between distribution and confidence interval plots
        self.ax2.sharex(self.ax1)
        self.show_ci_plot = True  # Track visibility of confidence interval plot
        
        # Create layout with no margins to maximize space
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.canvas)
        self.setLayout(layout)
        
        # Initial plot
        self.update_plots()
    
    def update_plots(self):
        """Update all plots"""
        # Clear all axes
        self.ax1.clear()
        self.ax2.clear()
        self.ax3.clear()
        
        # Calculate range: always use true distribution range (fixed axes)
        x_min = -5 #self.mean - 5 * self.std
        x_max = 5# self.mean + 5 * self.std
        
        # Create linspace with 2000 steps
        x = np.linspace(x_min, x_max, 2000)
        
        # TOP PLOT: Distribution plot
        # Calculate and plot true normal distribution (normalized PDF)
        y_true = (1 / (self.std * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - self.mean) / self.std) ** 2)
        self.ax1.plot(x, y_true, 'b-', linewidth=2, label=f'N({self.mean:.2f}, {self.std:.2f})')
        
        # Plot all estimated distributions (limit legend to max 10 entries)
        colors = ['r', 'g', 'm', 'c', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive']
        max_legend_entries = 10
        for i, (est_mean, est_std) in enumerate(self.estimated_distributions):
            y_estimated = (1 / (est_std * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - est_mean) / est_std) ** 2)
            color = colors[i % len(colors)]
            # Only add label if we haven't exceeded max legend entries (1 for true dist + estimated)
            label = f'N({est_mean:.2f}, {est_std:.2f})' if i < max_legend_entries - 1 else None
            self.ax1.plot(x, y_estimated, '--', color=color, linewidth=1.5, 
                        label=label, alpha=0.8)
        
        # Set fixed x-axis limits based on true distribution
        self.ax1.set_xlim(x_min, x_max)
        self.ax1.set_ylabel('Wahrscheinlichkeitsdichte', fontsize=12)
        self.ax1.set_title('Verteilungen', fontsize=14, fontweight='bold', pad=10)
        self.ax1.grid(True, alpha=0.3)
        self.ax1.legend()
        
        # BOTTOM PLOT: Confidence interval plot (only if enabled)
        if self.show_ci_plot:
            # Set fixed x-axis limits (same as distribution plot - shared)
            self.ax2.set_xlim(x_min, x_max)
            
            # Plot vertical line for true mean
            self.ax2.axvline(self.true_mean, color='b', linestyle='-', linewidth=2, 
                           label=f'Wahrer Mittelwert: μ={self.true_mean:.2f}')
            
            # Plot confidence intervals as horizontal bars (limit legend to max 10 entries)
            max_legend_entries = 10
            for i, (mean, ci_lower, ci_upper, n, ci_level) in enumerate(self.confidence_intervals):
                color = colors[i % len(colors)]
                y_pos = len(self.confidence_intervals) - i - 1  # Position from top
                # Only add label if we haven't exceeded max legend entries (1 for true mean + CIs)
                label = f'KI #{i+1} ({ci_level:.1f}%): [{ci_lower:.3f}, {ci_upper:.3f}]' if i < max_legend_entries - 1 else None
                # Draw horizontal line for CI
                self.ax2.plot([ci_lower, ci_upper], [y_pos, y_pos], color=color, 
                            linewidth=3, alpha=0.7, label=label)
                # Mark the sample mean
                self.ax2.plot(mean, y_pos, 'o', color=color, markersize=8)
            
            # Set y-axis to show all intervals
            if self.confidence_intervals:
                self.ax2.set_ylim(-0.5, len(self.confidence_intervals) - 0.5)
            else:
                self.ax2.set_ylim(-0.5, 0.5)
            
            # Labels and title
            self.ax2.set_xlabel('x', fontsize=12)
            self.ax2.set_ylabel('Konfidenzintervall', fontsize=12)
            self.ax2.set_title(f'Konfidenzintervalle für μ (σ unbekannt)', fontsize=14, fontweight='bold', pad=10)
            self.ax2.grid(True, alpha=0.3, axis='x')
            if self.confidence_intervals:
                self.ax2.legend(loc='upper right', fontsize=8)
            self.ax2.set_visible(True)
        else:
            self.ax2.set_visible(False)
        
        # BOTTOM PLOT: Bar chart showing how many intervals contain μ
        if self.show_ci_plot and self.confidence_intervals:
            # Count intervals that contain true_mean
            contains_true_mean = 0
            does_not_contain = 0
            
            for mean, ci_lower, ci_upper, n, ci_level in self.confidence_intervals:
                if ci_lower <= self.true_mean <= ci_upper:
                    contains_true_mean += 1
                else:
                    does_not_contain += 1
            
            total = len(self.confidence_intervals)
            pct_contains = (contains_true_mean / total * 100) if total > 0 else 0
            pct_not_contains = (does_not_contain / total * 100) if total > 0 else 0
            
            # Create horizontal bar chart
            categories = ['Enthält μ', 'Enthält μ nicht']
            values = [pct_contains, pct_not_contains]
            colors_bar = ['green', 'red']
            
            bars = self.ax3.barh(categories, values, color=colors_bar, alpha=0.7)
            
            # Add percentage labels to the right of bars
            for i, (bar, val) in enumerate(zip(bars, values)):
                width = bar.get_width()
                self.ax3.text(width + 1, bar.get_y() + bar.get_height() / 2,
                            f'{val:.1f}% ({contains_true_mean if i == 0 else does_not_contain}/{total})',
                            ha='left', va='center', fontweight='bold', fontsize=11)
            
            self.ax3.set_xlim(0, 110)  # Extended to accommodate labels
            self.ax3.set_xlabel('Prozent', fontsize=12)
            self.ax3.set_title('Anteil der Konfidenzintervalle, die μ enthalten', fontsize=12, fontweight='bold', pad=10)
            self.ax3.grid(True, alpha=0.3, axis='x')
            self.ax3.set_visible(True)
        else:
            self.ax3.set_visible(False)
        
        # Adjust layout to prevent text from being cut off
        # Adjust subplot layout based on visibility
        if self.show_ci_plot:
            self.ax2.set_visible(True)
            self.figure.subplots_adjust(hspace=0.4)
        else:
            self.ax2.set_visible(False)
            self.ax3.set_visible(False)
            self.figure.subplots_adjust(hspace=0.0)
        self.figure.tight_layout(rect=[0, 0, 1, 0.98])
        self.canvas.draw()
    
    def set_parameters(self, mean, std):
        """Update mean and standard deviation"""
        self.mean = mean
        self.std = max(0.01, std)  # Prevent division by zero
        self.true_mean = mean
        self.update_plots()
    
    def get_x_limits(self):
        """Get the x-axis limits"""
        x_min = self.mean - 5 * self.std
        x_max = self.mean + 5 * self.std
        return x_min, x_max
    
    def add_estimated_distribution(self, estimated_mean, estimated_std):
        """Add a new estimated distribution to the plot"""
        if estimated_mean is not None and estimated_std is not None:
            self.estimated_distributions.append((estimated_mean, max(0.01, estimated_std)))
            self.update_plots()
    
    def clear_estimated_distributions(self):
        """Clear all estimated distributions"""
        self.estimated_distributions = []
        self.update_plots()
    
    def set_confidence_level(self, confidence_level):
        """Set the confidence level percentage"""
        self.confidence_level = confidence_level
        self.update_plots()
    
    def add_confidence_interval(self, sample_mean, sample_std, n, confidence_level=None):
        """Add a confidence interval for the mean"""
        if confidence_level is None:
            confidence_level = self.confidence_level
        
        if n > 1 and sample_std is not None:
            # Calculate z-score for the given confidence level
            alpha = confidence_level / 100.0
            z_score = stats.norm.ppf((1 + alpha) / 2)
            
            # Calculate confidence interval: mean ± z * std_error
            std_error = sample_std / np.sqrt(n)
            margin = z_score * std_error
            ci_lower = sample_mean - margin
            ci_upper = sample_mean + margin
            self.confidence_intervals.append((sample_mean, ci_lower, ci_upper, n, confidence_level))
            self.update_plots()
    
    def clear_confidence_intervals(self):
        """Clear all confidence intervals"""
        self.confidence_intervals = []
        self.update_plots()
    
    def set_ci_plot_visible(self, visible):
        """Show or hide the confidence interval plot"""
        self.show_ci_plot = visible
        self.update_plots()


class ConfidenceIntervalPlot(QWidget):
    def __init__(self, true_mean, x_min, x_max, confidence_level=95.0):
        super().__init__()
        self.true_mean = true_mean
        self.x_min = x_min
        self.x_max = x_max
        self.confidence_level = confidence_level
        # Store list of confidence intervals: [(mean, ci_lower, ci_upper, n, confidence_level), ...]
        self.confidence_intervals = []
        
        # Create matplotlib figure (no fixed size - will resize with widget)
        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)
        self.ax = self.figure.add_subplot(111)
        
        # Create layout with no margins to maximize space
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.canvas)
        self.setLayout(layout)
        
        # Initial plot
        self.update_plot()
    
    def update_plot(self):
        """Update the confidence interval plot"""
        self.ax.clear()
        
        # Set fixed x-axis limits (same as distribution plot)
        self.ax.set_xlim(self.x_min, self.x_max)
        
        # Plot vertical line for true mean
        self.ax.axvline(self.true_mean, color='b', linestyle='-', linewidth=2, 
                       label=f'True mean: μ={self.true_mean:.2f}')
        
        # Plot confidence intervals as horizontal bars
        colors = ['r', 'g', 'm', 'c', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive']
        for i, (mean, ci_lower, ci_upper, n, ci_level) in enumerate(self.confidence_intervals):
            color = colors[i % len(colors)]
            y_pos = len(self.confidence_intervals) - i - 1  # Position from top
            # Draw horizontal line for CI
            self.ax.plot([ci_lower, ci_upper], [y_pos, y_pos], color=color, 
                        linewidth=3, alpha=0.7, 
                        label=f'CI #{i+1} ({ci_level:.1f}%): [{ci_lower:.3f}, {ci_upper:.3f}]')
            # Mark the sample mean
            self.ax.plot(mean, y_pos, 'o', color=color, markersize=8)
        
        # Set y-axis to show all intervals
        if self.confidence_intervals:
            self.ax.set_ylim(-0.5, len(self.confidence_intervals) - 0.5)
        else:
            self.ax.set_ylim(-0.5, 0.5)
        
        # Labels and title - use current confidence level
        self.ax.set_xlabel('x', fontsize=12)
        self.ax.set_ylabel('Konfidenzintervall', fontsize=12)
        self.ax.set_title(f'Konfidenzintervalle ({self.confidence_level:.1f}%)', fontsize=14, fontweight='bold')
        self.ax.grid(True, alpha=0.3, axis='x')
        if self.confidence_intervals:
            self.ax.legend(loc='upper right', fontsize=8)
        
        # Adjust layout to prevent text from being cut off
        self.figure.tight_layout(rect=[0, 0, 1, 0.96])
        self.canvas.draw()
    
    def set_true_mean(self, true_mean):
        """Update true mean"""
        self.true_mean = true_mean
        self.update_plot()
    
    def set_x_limits(self, x_min, x_max):
        """Update x-axis limits"""
        self.x_min = x_min
        self.x_max = x_max
        self.update_plot()
    
    def set_confidence_level(self, confidence_level):
        """Set the confidence level percentage"""
        self.confidence_level = confidence_level
        self.update_plot()
    
    def add_confidence_interval(self, sample_mean, sample_std, n, confidence_level=None):
        """Add a confidence interval for the mean"""
        if confidence_level is None:
            confidence_level = self.confidence_level
        
        if n > 1 and sample_std is not None:
            # Calculate z-score for the given confidence level
            # For confidence level α%, z = norm.ppf((1 + α/100)/2)
            alpha = confidence_level / 100.0
            z_score = stats.norm.ppf((1 + alpha) / 2)
            
            # Calculate confidence interval: mean ± z * std_error
            std_error = sample_std / np.sqrt(n)
            margin = z_score * std_error
            ci_lower = sample_mean - margin
            ci_upper = sample_mean + margin
            self.confidence_intervals.append((sample_mean, ci_lower, ci_upper, n, confidence_level))
            self.update_plot()
    
    def clear_confidence_intervals(self):
        """Clear all confidence intervals"""
        self.confidence_intervals = []
        self.update_plot()


def create_sigma_icon():
    """Create an icon with Sigma symbol for taskbar"""
    # Create larger icon for better visibility (64x64 for high DPI)
    pixmap = QPixmap(64, 64)
    pixmap.fill(QColor(52, 152, 219))  # Modern blue background (#3498db)
    
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.Antialiasing)
    
    # Draw rounded rectangle background
    painter.setBrush(QColor(52, 152, 219))
    painter.setPen(QColor(41, 128, 185))  # Slightly darker blue border
    painter.drawRoundedRect(2, 2, 60, 60, 8, 8)
    
    # Set font for Sigma symbol - larger and white
    font = QFont("Arial", 36, QFont.Bold)
    painter.setFont(font)
    painter.setPen(QColor(255, 255, 255))  # White color
    
    # Draw Sigma symbol (Σ) centered
    painter.drawText(0, 0, 64, 64, Qt.AlignCenter, "Σ")
    painter.end()
    
    return QIcon(pixmap)


def save_icon_to_file(icon_path="app_icon.ico"):
    """Save the icon as .ico file for use with PyInstaller"""
    icon = create_sigma_icon()
    # Create multiple sizes for .ico file (Windows requires multiple sizes)
    sizes = [16, 32, 48, 64, 128, 256]
    pixmaps = []
    
    for size in sizes:
        pixmap = icon.pixmap(size, size)
        pixmaps.append(pixmap)
    
    # Save as PNG (PyInstaller can use PNG as icon)
    pixmaps[-1].save(icon_path.replace('.ico', '.png'), 'PNG')
    print(f"Icon saved to {icon_path.replace('.ico', '.png')}")
    print(f"To use with PyInstaller, run: pyinstaller --icon={icon_path.replace('.ico', '.png')} --onefile --windowed main.py")


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Statistik Rechner")
        self.setGeometry(100, 100, 1400, 800)
        # Set window icon with Sigma symbol
        icon = create_sigma_icon()
        self.setWindowIcon(icon)
        
        # Try to set taskbar icon on Windows (works better when compiled, but we try anyway)
        if sys.platform == 'win32':
            try:
                # Set AppUserModelID to help Windows identify the app
                ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("StatistikRechner.App")
            except:
                pass  # If it fails, just continue
        
        # Create tab widget as central widget
        self.tab_widget = QTabWidget()
        self.setCentralWidget(self.tab_widget)
        
        # Create the Konfidenzintervall tab (existing content)
        konfidenzintervall_tab = self.create_normal_distribution_tab()
        self.tab_widget.addTab(konfidenzintervall_tab, "Konfidenz")
        
        # Create the Normal distribution tab
        normal_dist_tab = self.create_normal_tab()
        self.tab_widget.addTab(normal_dist_tab, "Normal")
        
        # Create the Student-t tab
        student_t_tab = self.create_student_t_tab()
        self.tab_widget.addTab(student_t_tab, "Student-t")
        
        # Create the Chi-squared tab
        chi_squared_tab = self.create_chi_squared_tab()
        self.tab_widget.addTab(chi_squared_tab, "χ²")
        
        # Set initial values
        self.on_parameters_changed()
    
    def create_normal_distribution_tab(self):
        """Create the normal distribution tab with all existing functionality"""
        # Create main widget for this tab
        tab_widget = QWidget()
        main_layout = QHBoxLayout()
        tab_widget.setLayout(main_layout)
        
        # LEFT COLUMN: Controls
        left_column = QWidget()
        left_layout = QVBoxLayout()
        left_column.setLayout(left_layout)
        
        # Section 1: Parameters of True distribution
        params_section = QWidget()
        params_layout = QVBoxLayout()
        params_section.setLayout(params_layout)
        
        params_label = QLabel("Parameter der wahren Verteilung")
        params_label.setStyleSheet(f"font-weight: bold; font-size: {title_fontsize}px; padding: 5px;")
        params_layout.addWidget(params_label)
        
        # Create input section
        input_widget = QWidget()
        input_layout = QFormLayout()
        input_widget.setLayout(input_layout)
        
        # Mean input
        mean_label = QLabel("Erwartungswert (μ):")
        mean_label.setStyleSheet(f"font-size: {widget_text_size}px;")
        self.mean_spinbox = QDoubleSpinBox()
        self.mean_spinbox.setRange(-1000.0, 1000.0)
        self.mean_spinbox.setValue(0.0)
        self.mean_spinbox.setDecimals(2)
        self.mean_spinbox.setSingleStep(0.1)
        self.mean_spinbox.setFixedWidth(120)
        self.mean_spinbox.setStyleSheet(f"font-size: {widget_text_size}px;")
        self.mean_spinbox.valueChanged.connect(self.on_parameters_changed)
        input_layout.addRow(mean_label, self.mean_spinbox)
        
        # Standard deviation input
        std_label = QLabel("Standardabweichung (σ):")
        std_label.setStyleSheet(f"font-size: {widget_text_size}px;")
        self.std_spinbox = QDoubleSpinBox()
        self.std_spinbox.setRange(0.01, 100.0)
        self.std_spinbox.setValue(1.0)
        self.std_spinbox.setDecimals(2)
        self.std_spinbox.setSingleStep(0.1)
        self.std_spinbox.setFixedWidth(120)
        self.std_spinbox.setStyleSheet(f"font-size: {widget_text_size}px;")
        self.std_spinbox.valueChanged.connect(self.on_parameters_changed)
        input_layout.addRow(std_label, self.std_spinbox)
        
        params_layout.addWidget(input_widget)
        left_layout.addWidget(params_section)
        
        # Section 2: Draw samples
        draw_samples_section = QWidget()
        draw_samples_layout = QVBoxLayout()
        draw_samples_section.setLayout(draw_samples_layout)
        
        draw_samples_label = QLabel("Stichproben")
        draw_samples_label.setStyleSheet(f"font-weight: bold; font-size: {title_fontsize}px; padding: 5px;")
        draw_samples_layout.addWidget(draw_samples_label)
        
        # Number of samples input
        samples_input_widget = QWidget()
        samples_input_layout = QHBoxLayout()
        samples_input_widget.setLayout(samples_input_layout)
        
        samples_label = QLabel("Größe der Stichprobe (n):")
        samples_label.setStyleSheet(f"font-size: {widget_text_size}px;")
        self.n_samples_spinbox = QSpinBox()
        self.n_samples_spinbox.setRange(1, 10000)
        self.n_samples_spinbox.setValue(10)
        self.n_samples_spinbox.setStyleSheet(f"font-size: {widget_text_size}px;")
        
        # Draw samples button
        self.draw_samples_button = QPushButton("Ziehen")
        self.draw_samples_button.setStyleSheet(f"font-size: {widget_text_size}px;")
        self.draw_samples_button.clicked.connect(self.draw_samples)
        
        samples_input_layout.addWidget(samples_label)
        samples_input_layout.addWidget(self.n_samples_spinbox)
        samples_input_layout.addWidget(self.draw_samples_button)
        samples_input_layout.addStretch()
        
        draw_samples_layout.addWidget(samples_input_widget)
        
        # Store current sample statistics
        self.current_sample_mean = None
        self.current_sample_std = None
        self.current_n_samples = None
        
        # Labels for sample mean and std dev estimates (using LaTeX)
        stats_widget = QWidget()
        stats_layout = QVBoxLayout()
        stats_widget.setLayout(stats_layout)
        
        self.sample_mean_label = QLabel()
        self.sample_mean_label.setStyleSheet("padding: 5px;")
        self.sample_std_label = QLabel()
        self.sample_std_label.setStyleSheet("padding: 5px;")
        # Initialize with placeholder
        self.update_statistics_labels(0.0, 0.0)
        
        stats_layout.addWidget(self.sample_mean_label)
        stats_layout.addWidget(self.sample_std_label)
        draw_samples_layout.addWidget(stats_widget)
        
        # Create table
        self.samples_table = QTableWidget()
        self.samples_table.setRowCount(2)
        self.samples_table.setVerticalHeaderLabels(["Stichprobe", "Wert"])
        self.samples_table.setAlternatingRowColors(True)
        self.samples_table.horizontalHeader().setVisible(False)
        self.samples_table.verticalHeader().setDefaultSectionSize(30)
        self.samples_table.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        self.samples_table.setFixedHeight(100)
        # Set fixed width to prevent table from expanding horizontally
        # You can adjust this value (in pixels) to change the table width
        self.samples_table.setFixedWidth(400)
        draw_samples_layout.addWidget(self.samples_table)
        
        left_layout.addWidget(draw_samples_section)
        
        # Section 3: Confidence intervals
        ci_section = QWidget()
        ci_layout = QVBoxLayout()
        ci_section.setLayout(ci_layout)
        
        ci_label = QLabel("Konfidenzintervalle")
        ci_label.setStyleSheet(f"font-weight: bold; font-size: {title_fontsize}px; padding: 5px;")
        ci_layout.addWidget(ci_label)
        
        # Confidence level input
        ci_input_widget = QWidget()
        ci_input_layout = QHBoxLayout()
        ci_input_widget.setLayout(ci_input_layout)
        
        ci_level_label = QLabel("Konfidenzniveau (%):")
        ci_level_label.setStyleSheet(f"font-size: {widget_text_size}px;")
        self.confidence_level_spinbox = QDoubleSpinBox()
        self.confidence_level_spinbox.setRange(0.1, 99.9)
        self.confidence_level_spinbox.setValue(95.0)
        self.confidence_level_spinbox.setDecimals(1)
        self.confidence_level_spinbox.setSingleStep(0.1)
        self.confidence_level_spinbox.setFixedWidth(80)
        self.confidence_level_spinbox.setStyleSheet(f"font-size: {widget_text_size}px;")
        self.confidence_level_spinbox.valueChanged.connect(self.on_confidence_level_changed)
        
        ci_input_layout.addWidget(ci_level_label)
        ci_input_layout.addWidget(self.confidence_level_spinbox)
        ci_input_layout.addStretch()
        
        ci_layout.addWidget(ci_input_widget)
        left_layout.addWidget(ci_section)
        
        # Section 4: Plot
        plot_section = QWidget()
        plot_layout = QVBoxLayout()
        plot_section.setLayout(plot_layout)
        
        plot_label = QLabel("Abbildunge")
        plot_label.setStyleSheet(f"font-weight: bold; font-size: {title_fontsize}px; padding: 5px;")
        plot_layout.addWidget(plot_label)
        
        # Checkbox for automatic drawing
        self.auto_draw_checkbox = QCheckBox("Automatisch ziehen")
        self.auto_draw_checkbox.setChecked(False)
        self.auto_draw_checkbox.setStyleSheet(f"font-size: {widget_text_size}px;")
        plot_layout.addWidget(self.auto_draw_checkbox)
        
        # Plot buttons
        plot_buttons_widget = QWidget()
        plot_buttons_layout = QHBoxLayout()
        plot_buttons_widget.setLayout(plot_buttons_layout)
        
        # Button to plot estimated distribution
        self.plot_estimated_button = QPushButton("Verteilung")
        self.plot_estimated_button.setStyleSheet(f"font-size: {widget_text_size}px;")
        self.plot_estimated_button.clicked.connect(self.plot_estimated_distribution)
        
        # Button to plot confidence interval
        self.plot_ci_button = QPushButton("Konf. Intervall")
        self.plot_ci_button.setStyleSheet(f"font-size: {widget_text_size}px;")
        self.plot_ci_button.clicked.connect(self.plot_confidence_interval)
        
        # Button to clear estimated distributions
        self.clear_estimated_button = QPushButton("Löschen")
        self.clear_estimated_button.setStyleSheet(f"font-size: {widget_text_size}px;")
        self.clear_estimated_button.clicked.connect(self.clear_estimated_distributions)
        
        plot_buttons_layout.addWidget(self.plot_estimated_button)
        plot_buttons_layout.addWidget(self.plot_ci_button)
        plot_buttons_layout.addWidget(self.clear_estimated_button)
        plot_buttons_layout.addStretch()
        
        plot_layout.addWidget(plot_buttons_widget)
        left_layout.addWidget(plot_section)
        
        left_layout.addStretch()
        # Set maximum width for left column to prevent it from expanding too much
        # The table width (400px) + padding should fit comfortably
        left_column.setMaximumWidth(500)
        main_layout.addWidget(left_column)
        
        # RIGHT COLUMN: Plots
        right_column = QWidget()
        right_layout = QVBoxLayout()
        right_column.setLayout(right_layout)
        
        # Create combined plot widget with subplots sharing x-axis
        self.plot_widget = CombinedPlot()
        right_layout.addWidget(self.plot_widget)
        
        main_layout.addWidget(right_column)
        
        return tab_widget
    
    def create_normal_tab(self):
        """Create the Normal distribution tab"""
        # Create main widget for this tab
        tab_widget = QWidget()
        main_layout = QHBoxLayout()
        tab_widget.setLayout(main_layout)
        
        # LEFT COLUMN: Controls
        left_column = QWidget()
        left_layout = QVBoxLayout()
        left_column.setLayout(left_layout)
        
        # Section 1: CDF calculation (z → Φ(z))
        cdf_section = QWidget()
        cdf_layout = QVBoxLayout()
        cdf_section.setLayout(cdf_layout)
        
        cdf_label = QLabel("Verteilungsfunktion Φ(z)")
        cdf_label.setStyleSheet(f"font-weight: bold; font-size: {title_fontsize}px; padding: 5px;")
        cdf_layout.addWidget(cdf_label)
        
        cdf_input_widget = QWidget()
        cdf_input_layout = QFormLayout()
        cdf_input_widget.setLayout(cdf_input_layout)
        
        z_label = QLabel("z:")
        z_label.setStyleSheet(f"font-size: {widget_text_size}px;")
        self.z_input = QDoubleSpinBox()
        self.z_input.setRange(-10.0, 10.0)
        self.z_input.setValue(0.0)
        self.z_input.setDecimals(4)
        self.z_input.setSingleStep(0.1)
        self.z_input.setFixedWidth(120)
        self.z_input.setStyleSheet(f"font-size: {widget_text_size}px;")
        self.z_input.valueChanged.connect(self.on_z_changed)
        cdf_input_layout.addRow(z_label, self.z_input)
        
        phi_label = QLabel("Φ(z) =")
        phi_label.setStyleSheet(f"font-size: {widget_text_size}px;")
        self.phi_output = QLabel("0.5000")
        self.phi_output.setStyleSheet(f"font-size: {widget_text_size}px; padding: 5px; font-weight: bold;")
        cdf_input_layout.addRow(phi_label, self.phi_output)
        
        cdf_layout.addWidget(cdf_input_widget)
        left_layout.addWidget(cdf_section)
        
        # Section 2: Quantile calculation (α → z)
        quantile_section = QWidget()
        quantile_layout = QVBoxLayout()
        quantile_section.setLayout(quantile_layout)
        
        quantile_label = QLabel("Quantil (Inverse von Φ)")
        quantile_label.setStyleSheet(f"font-weight: bold; font-size: {title_fontsize}px; padding: 5px;")
        quantile_layout.addWidget(quantile_label)
        
        quantile_input_widget = QWidget()
        quantile_input_layout = QFormLayout()
        quantile_input_widget.setLayout(quantile_input_layout)
        
        alpha_label = QLabel("α:")
        alpha_label.setStyleSheet(f"font-size: {widget_text_size}px;")
        self.alpha_input = QDoubleSpinBox()
        self.alpha_input.setRange(0.0001, 0.9999)
        self.alpha_input.setValue(0.5)
        self.alpha_input.setDecimals(4)
        self.alpha_input.setSingleStep(0.01)
        self.alpha_input.setFixedWidth(120)
        self.alpha_input.setStyleSheet(f"font-size: {widget_text_size}px;")
        self.alpha_input.valueChanged.connect(self.on_alpha_changed)
        quantile_input_layout.addRow(alpha_label, self.alpha_input)
        
        z_quantile_label = QLabel("z =")
        z_quantile_label.setStyleSheet(f"font-size: {widget_text_size}px;")
        self.z_quantile_output = QLabel("0.0000")
        self.z_quantile_output.setStyleSheet(f"font-size: {widget_text_size}px; padding: 5px; font-weight: bold;")
        quantile_input_layout.addRow(z_quantile_label, self.z_quantile_output)
        
        quantile_layout.addWidget(quantile_input_widget)
        left_layout.addWidget(quantile_section)
        
        left_layout.addStretch()
        # Set maximum width for left column to prevent it from expanding too much
        left_column.setMaximumWidth(500)
        main_layout.addWidget(left_column)
        
        # RIGHT COLUMN: Plots
        right_column = QWidget()
        right_layout = QVBoxLayout()
        right_column.setLayout(right_layout)
        
        # Create standard normal plot widget
        self.normal_plot = StandardNormalPlot()
        right_layout.addWidget(self.normal_plot)
        
        main_layout.addWidget(right_column)
        
        # Initialize values
        self.on_z_changed()
        self.on_alpha_changed()
        
        return tab_widget
    
    def create_student_t_tab(self):
        """Create the Student-t distribution tab"""
        # Create main widget for this tab
        tab_widget = QWidget()
        main_layout = QHBoxLayout()
        tab_widget.setLayout(main_layout)
        
        # LEFT COLUMN: Controls
        left_column = QWidget()
        left_layout = QVBoxLayout()
        left_column.setLayout(left_layout)
        
        # Section 0: Degrees of freedom
        df_section = QWidget()
        df_layout = QVBoxLayout()
        df_section.setLayout(df_layout)
        
        df_label = QLabel("Freiheitsgrade")
        df_label.setStyleSheet(f"font-weight: bold; font-size: {title_fontsize}px; padding: 5px;")
        df_layout.addWidget(df_label)
        
        df_input_widget = QWidget()
        df_input_layout = QFormLayout()
        df_input_widget.setLayout(df_input_layout)
        
        df_input_label = QLabel("ν:")
        df_input_label.setStyleSheet(f"font-size: {widget_text_size}px;")
        self.student_df_input = QSpinBox()
        self.student_df_input.setRange(1, 1000)
        self.student_df_input.setValue(1)
        self.student_df_input.setStyleSheet(f"font-size: {widget_text_size}px;")
        self.student_df_input.valueChanged.connect(self.on_student_df_changed)
        df_input_layout.addRow(df_input_label, self.student_df_input)
        
        df_layout.addWidget(df_input_widget)
        left_layout.addWidget(df_section)
        
        # Section 1: CDF calculation (t → CDF(t))
        cdf_section = QWidget()
        cdf_layout = QVBoxLayout()
        cdf_section.setLayout(cdf_layout)
        
        cdf_label = QLabel("Verteilungsfunktion")
        cdf_label.setStyleSheet(f"font-weight: bold; font-size: {title_fontsize}px; padding: 5px;")
        cdf_layout.addWidget(cdf_label)
        
        cdf_input_widget = QWidget()
        cdf_input_layout = QFormLayout()
        cdf_input_widget.setLayout(cdf_input_layout)
        
        t_label = QLabel("t:")
        t_label.setStyleSheet(f"font-size: {widget_text_size}px;")
        self.student_t_input = QDoubleSpinBox()
        self.student_t_input.setRange(-10.0, 10.0)
        self.student_t_input.setValue(0.0)
        self.student_t_input.setDecimals(4)
        self.student_t_input.setSingleStep(0.1)
        self.student_t_input.setFixedWidth(120)
        self.student_t_input.setStyleSheet(f"font-size: {widget_text_size}px;")
        self.student_t_input.valueChanged.connect(self.on_student_t_changed)
        cdf_input_layout.addRow(t_label, self.student_t_input)
        
        student_cdf_label = QLabel("CDF(t) =")
        student_cdf_label.setStyleSheet(f"font-size: {widget_text_size}px;")
        self.student_cdf_output = QLabel("0.5000")
        self.student_cdf_output.setStyleSheet(f"font-size: {widget_text_size}px; padding: 5px; font-weight: bold;")
        cdf_input_layout.addRow(student_cdf_label, self.student_cdf_output)
        
        cdf_layout.addWidget(cdf_input_widget)
        left_layout.addWidget(cdf_section)
        
        # Section 2: Quantile calculation (α → t)
        quantile_section = QWidget()
        quantile_layout = QVBoxLayout()
        quantile_section.setLayout(quantile_layout)
        
        quantile_label = QLabel("Quantil (Inverse)")
        quantile_label.setStyleSheet(f"font-weight: bold; font-size: {title_fontsize}px; padding: 5px;")
        quantile_layout.addWidget(quantile_label)
        
        quantile_input_widget = QWidget()
        quantile_input_layout = QFormLayout()
        quantile_input_widget.setLayout(quantile_input_layout)
        
        student_alpha_label = QLabel("α:")
        student_alpha_label.setStyleSheet(f"font-size: {widget_text_size}px;")
        self.student_alpha_input = QDoubleSpinBox()
        self.student_alpha_input.setRange(0.0001, 0.9999)
        self.student_alpha_input.setValue(0.5)
        self.student_alpha_input.setDecimals(4)
        self.student_alpha_input.setSingleStep(0.01)
        self.student_alpha_input.setFixedWidth(120)
        self.student_alpha_input.setStyleSheet(f"font-size: {widget_text_size}px;")
        self.student_alpha_input.valueChanged.connect(self.on_student_alpha_changed)
        quantile_input_layout.addRow(student_alpha_label, self.student_alpha_input)
        
        student_t_quantile_label = QLabel("t =")
        student_t_quantile_label.setStyleSheet(f"font-size: {widget_text_size}px;")
        self.student_t_quantile_output = QLabel("0.0000")
        self.student_t_quantile_output.setStyleSheet(f"font-size: {widget_text_size}px; padding: 5px; font-weight: bold;")
        quantile_input_layout.addRow(student_t_quantile_label, self.student_t_quantile_output)
        
        quantile_layout.addWidget(quantile_input_widget)
        left_layout.addWidget(quantile_section)
        
        left_layout.addStretch()
        # Set maximum width for left column to prevent it from expanding too much
        left_column.setMaximumWidth(500)
        main_layout.addWidget(left_column)
        
        # RIGHT COLUMN: Plots
        right_column = QWidget()
        right_layout = QVBoxLayout()
        right_column.setLayout(right_layout)
        
        # Create Student-t plot widget
        self.student_plot = StudentTPlot()
        right_layout.addWidget(self.student_plot)
        
        main_layout.addWidget(right_column)
        
        # Initialize values
        self.on_student_df_changed()
        self.on_student_t_changed()
        self.on_student_alpha_changed()
        
        return tab_widget
    
    def create_chi_squared_tab(self):
        """Create the Chi-squared distribution tab"""
        # Create main widget for this tab
        tab_widget = QWidget()
        main_layout = QHBoxLayout()
        tab_widget.setLayout(main_layout)
        
        # LEFT COLUMN: Controls
        left_column = QWidget()
        left_layout = QVBoxLayout()
        left_column.setLayout(left_layout)
        
        # Section 0: Degrees of freedom
        df_section = QWidget()
        df_layout = QVBoxLayout()
        df_section.setLayout(df_layout)
        
        df_label = QLabel("Freiheitsgrade")
        df_label.setStyleSheet(f"font-weight: bold; font-size: {title_fontsize}px; padding: 5px;")
        df_layout.addWidget(df_label)
        
        df_input_widget = QWidget()
        df_input_layout = QFormLayout()
        df_input_widget.setLayout(df_input_layout)
        
        df_input_label = QLabel("ν:")
        df_input_label.setStyleSheet(f"font-size: {widget_text_size}px;")
        self.chi2_df_input = QSpinBox()
        self.chi2_df_input.setRange(1, 1000)
        self.chi2_df_input.setValue(1)
        self.chi2_df_input.setStyleSheet(f"font-size: {widget_text_size}px;")
        self.chi2_df_input.valueChanged.connect(self.on_chi2_df_changed)
        df_input_layout.addRow(df_input_label, self.chi2_df_input)
        
        df_layout.addWidget(df_input_widget)
        left_layout.addWidget(df_section)
        
        # Section 1: CDF calculation (χ² → CDF(χ²))
        cdf_section = QWidget()
        cdf_layout = QVBoxLayout()
        cdf_section.setLayout(cdf_layout)
        
        cdf_label = QLabel("Verteilungsfunktion")
        cdf_label.setStyleSheet(f"font-weight: bold; font-size: {title_fontsize}px; padding: 5px;")
        cdf_layout.addWidget(cdf_label)
        
        cdf_input_widget = QWidget()
        cdf_input_layout = QFormLayout()
        cdf_input_widget.setLayout(cdf_input_layout)
        
        chi2_label = QLabel("χ²:")
        chi2_label.setStyleSheet(f"font-size: {widget_text_size}px;")
        self.chi2_input = QDoubleSpinBox()
        self.chi2_input.setRange(0.0, 1000.0)
        self.chi2_input.setValue(1.0)
        self.chi2_input.setDecimals(4)
        self.chi2_input.setSingleStep(0.1)
        self.chi2_input.setFixedWidth(120)
        self.chi2_input.setStyleSheet(f"font-size: {widget_text_size}px;")
        self.chi2_input.valueChanged.connect(self.on_chi2_changed)
        cdf_input_layout.addRow(chi2_label, self.chi2_input)
        
        chi2_cdf_label = QLabel("CDF(χ²) =")
        chi2_cdf_label.setStyleSheet(f"font-size: {widget_text_size}px;")
        self.chi2_cdf_output = QLabel("0.0000")
        self.chi2_cdf_output.setStyleSheet(f"font-size: {widget_text_size}px; padding: 5px; font-weight: bold;")
        cdf_input_layout.addRow(chi2_cdf_label, self.chi2_cdf_output)
        
        cdf_layout.addWidget(cdf_input_widget)
        left_layout.addWidget(cdf_section)
        
        # Section 2: Quantile calculation (α → χ²)
        quantile_section = QWidget()
        quantile_layout = QVBoxLayout()
        quantile_section.setLayout(quantile_layout)
        
        quantile_label = QLabel("Quantil (Inverse)")
        quantile_label.setStyleSheet(f"font-weight: bold; font-size: {title_fontsize}px; padding: 5px;")
        quantile_layout.addWidget(quantile_label)
        
        quantile_input_widget = QWidget()
        quantile_input_layout = QFormLayout()
        quantile_input_widget.setLayout(quantile_input_layout)
        
        chi2_alpha_label = QLabel("α:")
        chi2_alpha_label.setStyleSheet(f"font-size: {widget_text_size}px;")
        self.chi2_alpha_input = QDoubleSpinBox()
        self.chi2_alpha_input.setRange(0.0001, 0.9999)
        self.chi2_alpha_input.setValue(0.5)
        self.chi2_alpha_input.setDecimals(4)
        self.chi2_alpha_input.setSingleStep(0.01)
        self.chi2_alpha_input.setFixedWidth(120)
        self.chi2_alpha_input.setStyleSheet(f"font-size: {widget_text_size}px;")
        self.chi2_alpha_input.valueChanged.connect(self.on_chi2_alpha_changed)
        quantile_input_layout.addRow(chi2_alpha_label, self.chi2_alpha_input)
        
        chi2_quantile_label = QLabel("χ² =")
        chi2_quantile_label.setStyleSheet(f"font-size: {widget_text_size}px;")
        self.chi2_quantile_output = QLabel("0.0000")
        self.chi2_quantile_output.setStyleSheet(f"font-size: {widget_text_size}px; padding: 5px; font-weight: bold;")
        quantile_input_layout.addRow(chi2_quantile_label, self.chi2_quantile_output)
        
        quantile_layout.addWidget(quantile_input_widget)
        left_layout.addWidget(quantile_section)
        
        left_layout.addStretch()
        # Set maximum width for left column to prevent it from expanding too much
        left_column.setMaximumWidth(500)
        main_layout.addWidget(left_column)
        
        # RIGHT COLUMN: Plots
        right_column = QWidget()
        right_layout = QVBoxLayout()
        right_column.setLayout(right_layout)
        
        # Create Chi-squared plot widget
        self.chi2_plot = ChiSquaredPlot()
        right_layout.addWidget(self.chi2_plot)
        
        main_layout.addWidget(right_column)
        
        # Initialize values
        self.on_chi2_df_changed()
        self.on_chi2_changed()
        self.on_chi2_alpha_changed()
        
        return tab_widget
    
    def on_z_changed(self):
        """Calculate Φ(z) when z value changes"""
        z = self.z_input.value()
        phi_z = stats.norm.cdf(z)
        self.phi_output.setText(f"{phi_z:.6f}")
        # Update plot with z value
        if hasattr(self, 'normal_plot'):
            self.normal_plot.set_z_value(z)
    
    def on_alpha_changed(self):
        """Calculate quantile z when α value changes"""
        alpha = self.alpha_input.value()
        z_quantile = stats.norm.ppf(alpha)
        self.z_quantile_output.setText(f"{z_quantile:.6f}")
        # Update plot with alpha value
        if hasattr(self, 'normal_plot'):
            self.normal_plot.set_alpha_value(alpha)
    
    def on_student_df_changed(self):
        """Handle Student-t degrees of freedom changes"""
        df = self.student_df_input.value()
        if hasattr(self, 'student_plot'):
            self.student_plot.set_degrees_of_freedom(df)
        # Recalculate values with new df
        self.on_student_t_changed()
        self.on_student_alpha_changed()
    
    def on_student_t_changed(self):
        """Calculate CDF(t) when t value changes"""
        t = self.student_t_input.value()
        df = self.student_df_input.value()
        cdf_t = stats.t.cdf(t, df)
        self.student_cdf_output.setText(f"{cdf_t:.6f}")
        # Update plot with t value
        if hasattr(self, 'student_plot'):
            self.student_plot.set_z_value(t)
    
    def on_student_alpha_changed(self):
        """Calculate quantile t when α value changes"""
        alpha = self.student_alpha_input.value()
        df = self.student_df_input.value()
        t_quantile = stats.t.ppf(alpha, df)
        self.student_t_quantile_output.setText(f"{t_quantile:.6f}")
        # Update plot with alpha value
        if hasattr(self, 'student_plot'):
            self.student_plot.set_alpha_value(alpha)
    
    def on_chi2_df_changed(self):
        """Handle Chi-squared degrees of freedom changes"""
        df = self.chi2_df_input.value()
        if hasattr(self, 'chi2_plot'):
            self.chi2_plot.set_degrees_of_freedom(df)
        # Recalculate values with new df
        self.on_chi2_changed()
        self.on_chi2_alpha_changed()
    
    def on_chi2_changed(self):
        """Calculate CDF(χ²) when χ² value changes"""
        chi2 = self.chi2_input.value()
        df = self.chi2_df_input.value()
        cdf_chi2 = stats.chi2.cdf(chi2, df)
        self.chi2_cdf_output.setText(f"{cdf_chi2:.6f}")
        # Update plot with χ² value
        if hasattr(self, 'chi2_plot'):
            self.chi2_plot.set_z_value(chi2)
    
    def on_chi2_alpha_changed(self):
        """Calculate quantile χ² when α value changes"""
        alpha = self.chi2_alpha_input.value()
        df = self.chi2_df_input.value()
        chi2_quantile = stats.chi2.ppf(alpha, df)
        self.chi2_quantile_output.setText(f"{chi2_quantile:.6f}")
        # Update plot with alpha value
        if hasattr(self, 'chi2_plot'):
            self.chi2_plot.set_alpha_value(alpha)
    
    def on_parameters_changed(self):
        """Handle parameter changes"""
        mean = self.mean_spinbox.value()
        std = self.std_spinbox.value()
        self.plot_widget.set_parameters(mean, std)
    
    def on_confidence_level_changed(self):
        """Handle confidence level changes"""
        confidence_level = self.confidence_level_spinbox.value()
        self.plot_widget.set_confidence_level(confidence_level)
    
    def draw_samples(self):
        """Draw samples from the normal distribution and display in table"""
        n = self.n_samples_spinbox.value()
        mean = self.mean_spinbox.value()
        std = self.std_spinbox.value()
        
        # Generate samples from normal distribution
        samples = np.random.normal(loc=mean, scale=std, size=n)
        
        # Calculate sample statistics
        sample_mean = np.mean(samples)
        # Sample standard deviation (using n-1 for unbiased estimate)
        sample_std = np.std(samples, ddof=1) if n > 1 else 0.0
        
        # Store sample statistics
        self.current_sample_mean = sample_mean
        self.current_sample_std = sample_std
        self.current_n_samples = n
        
        # Update table - transposed: columns are samples, rows are "Sample" and "Value"
        self.samples_table.setColumnCount(n)
        for i, value in enumerate(samples):
            # Row 0: Sample number
            sample_item = QTableWidgetItem(str(i + 1))
            sample_item.setTextAlignment(Qt.AlignCenter)
            self.samples_table.setItem(0, i, sample_item)
            
            # Row 1: Sample value
            value_item = QTableWidgetItem(f"{value:.6f}")
            value_item.setTextAlignment(Qt.AlignCenter)
            self.samples_table.setItem(1, i, value_item)
        
        # Resize columns to content
        self.samples_table.resizeColumnsToContents()
        
        # Update sample statistics labels
        self.update_statistics_labels(sample_mean, sample_std)
    
    def plot_estimated_distribution(self):
        """Plot the estimated normal distribution using current sample statistics"""
        # Auto-draw samples if checkbox is checked
        if self.auto_draw_checkbox.isChecked():
            self.draw_samples()
        
        if self.current_sample_mean is not None and self.current_sample_std is not None:
            self.plot_widget.add_estimated_distribution(
                self.current_sample_mean, 
                self.current_sample_std
            )
    
    def plot_confidence_interval(self):
        """Plot the confidence interval using current sample statistics"""
        # Auto-draw samples if checkbox is checked
        if self.auto_draw_checkbox.isChecked():
            self.draw_samples()
        
        if self.current_sample_mean is not None and self.current_sample_std is not None and self.current_n_samples is not None:
            confidence_level = self.confidence_level_spinbox.value()
            self.plot_widget.add_confidence_interval(
                self.current_sample_mean,
                self.current_sample_std,
                self.current_n_samples,
                confidence_level
            )
    
    def clear_estimated_distributions(self):
        """Clear all estimated distributions from the plot"""
        self.plot_widget.clear_estimated_distributions()
        self.plot_widget.clear_confidence_intervals()
    
    def update_statistics_labels(self, sample_mean, sample_std):
        """Update the statistics labels with LaTeX rendering"""
        # Create LaTeX strings with formulas and values
        mean_latex = r"$\bar{x} = \frac{1}{n}\sum_{i=1}^{n} x_i = $" + f"{sample_mean:.6f}"
        std_latex = r"$S = \sqrt{\frac{1}{n-1}\sum_{i=1}^{n}(x_i - \bar{x})^2} = $" + f"{sample_std:.6f}"
        
        # Render and set pixmaps
        mean_pixmap = latex_to_pixmap(mean_latex, fontsize=latex_fontsize)
        std_pixmap = latex_to_pixmap(std_latex, fontsize=latex_fontsize)
        
        self.sample_mean_label.setPixmap(mean_pixmap)
        self.sample_std_label.setPixmap(std_pixmap)


def main():
    app = QApplication(sys.argv)
    
    # Set modern Fusion style
    app.setStyle('Fusion')
    
    # Apply modern stylesheet with light theme
    app.setStyleSheet("""
        QMainWindow {
            background-color: #f5f5f5;
        }
        QWidget {
            background-color: #ffffff;
            color: #2c3e50;
        }
        QTabWidget::pane {
            border: 1px solid #d0d0d0;
            background-color: #ffffff;
            border-radius: 4px;
        }
        QTabBar::tab {
            background-color: #e8e8e8;
            color: #2c3e50;
            border: 1px solid #d0d0d0;
            padding: 10px 20px;
            margin-right: 2px;
            border-top-left-radius: 4px;
            border-top-right-radius: 4px;
            font-size: 16px;
            min-width: 100px;
        }
        QTabBar::tab:selected {
            background-color: #ffffff;
            border-bottom-color: #ffffff;
        }
        QTabBar::tab:hover {
            background-color: #f0f0f0;
        }
        QPushButton {
            background-color: #3498db;
            color: white;
            border: none;
            padding: 8px 16px;
            border-radius: 4px;
            font-size: 18px;
            font-weight: bold;
        }
        QPushButton:hover {
            background-color: #2980b9;
        }
        QPushButton:pressed {
            background-color: #21618c;
        }
        QDoubleSpinBox, QSpinBox {
            background-color: #ffffff;
            border: 1px solid #d0d0d0;
            border-radius: 4px;
            padding: 4px;
            font-size: 18px;
        }
        QDoubleSpinBox:focus, QSpinBox:focus {
            border: 2px solid #3498db;
        }
        QLabel {
            color: #2c3e50;
        }
        QCheckBox {
            color: #2c3e50;
            font-size: 18px;
        }
        QCheckBox::indicator {
            width: 18px;
            height: 18px;
            border: 2px solid #3498db;
            border-radius: 3px;
            background-color: #ffffff;
        }
        QCheckBox::indicator:checked {
            background-color: #3498db;
            border-color: #3498db;
        }
        QTableWidget {
            background-color: #ffffff;
            border: 1px solid #d0d0d0;
            border-radius: 4px;
            gridline-color: #e8e8e8;
        }
        QTableWidget::item {
            padding: 4px;
        }
        QTableWidget::item:selected {
            background-color: #3498db;
            color: white;
        }
        QFormLayout {
            spacing: 10px;
        }
    """)
    
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    import sys
    # If --save-icon argument is passed, save icon and exit
    if len(sys.argv) > 1 and sys.argv[1] == "--save-icon":
        save_icon_to_file()
        sys.exit(0)
    main()

