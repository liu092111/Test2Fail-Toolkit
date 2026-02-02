from fpdf import FPDF
from fpdf.enums import XPos, YPos
from math import floor
import textwrap
import os
import tempfile
from datetime import datetime
import numpy as np

class ReportBuilder:
    def __init__(self, text_buffer, figures, data_info=None):
        self.text_buffer = text_buffer
        self.figures = figures
        self.data_info = data_info or {}
        self.pdf = FPDF()
        self.page_count = 0
        
        # Figure descriptions (8 figures total)
        # Figure 1-3, 6-10 from Description.docx
        # Removed: Figure 4 (Weibull Probability Paper), Figure 5
        self.figure_descriptions = [
            {
                'title': 'Histogram of Lifetime Data',
                'description': '''This figure illustrates the distribution of the observed failure times collected from the test samples. Each bar in the histogram represents the frequency of failures occurring within a specific range of lifetime cycles. By examining the overall shape of the histogram, we can identify how the failures are distributed over time - for instance, whether most components fail early (indicating early-life issues), steadily (indicating random failures), or predominantly at later stages (indicating wear-out mechanisms).''',
                'technical_details': '''Understanding the failure-time distribution allows engineers to:

Assess failure behavior - For example, a right-skewed distribution with a long tail may suggest a small portion of components last significantly longer than average, while a steep early peak could imply process or material inconsistency.

Choose an appropriate statistical model - The histogram's shape provides visual cues for model selection (e.g., Weibull for wear-out failures, Exponential for constant failure rates).

Guide reliability improvements - If failures are concentrated in early cycles, process control or burn-in testing may be needed; if they appear in later cycles, focus should shift toward material fatigue, corrosion, or long-term degradation mechanisms.

This figure thus establishes the empirical baseline of the product's lifetime characteristics, forming the foundation for quantitative modeling and predictive reliability assessment in later stages.'''
            },
            {
                'title': 'Probability Density Function (PDF) Comparison',
                'description': '''This figure compares three lifetime models, Weibull, Lognormal, and Exponential. Each fitted to the observed failure data using the Maximum Likelihood Estimation (MLE) method. MLE is a statistical approach that finds the most plausible model parameters so that the fitted curve best represents how failures occur over time.

Each colored curve represents a probability density function (PDF), which describes the likelihood that a component will fail at a given point in its lifetime. The Weibull distribution is widely used for mechanical parts because it can represent early-life, random, or wear-out failures depending on its shape. The Lognormal distribution often models chemical or environmental degradation, where aging accumulates gradually. The Exponential distribution assumes a constant failure rate and is typical for electronic or random-stress failures.

Once the models are fitted, the Mean Time To Failure (MTTF) can be calculated directly from their estimated parameters.''',
                'technical_details': '''This comparison highlights how different assumptions lead to different reliability interpretations. By observing which model best fits the data:

Model Selection Insight - Identifying which distribution best fits the data helps engineers infer whether failures are due to material aging, process variability, or random stress.

Quantitative Lifetime Estimation - The fitted model provides a consistent way to compute MTTF, transforming test results into measurable reliability metrics.

Design & Clarity Guidance - Comparing these model curves visually demonstrates how mathematical modeling translates raw test data into practical engineering understanding.'''
            },
            {
                'title': 'Survival Function Comparison',
                'description': '''This figure presents the Survival Function, denoted as S(t), for the analyzed failure data compared against the fitted probability distributions (e.g., Weibull, Lognormal, and Exponential). The curve represents the probability that a component will perform its required function without failure for a specific period. Starting at a probability of 1.0 (100%), the curve monotonically decreases over time, visually depicting the reliability decay of the product. The model that best aligns with the empirical data provides the most accurate baseline for predicting the remaining population at any given cycle count.''',
                'technical_details': '''The Survival Function is the primary tool for communicating reliability to stakeholders and guides several critical decisions:

Determine Warranty Periods - By pinpointing the time at which the survival probability remains high (e.g., 90% reliability), engineers can set warranty limits that minimize financial risk from returns.

Plan Maintenance Schedules - The slope of the curve indicates how quickly reliability is lost. A steep drop suggests a need for preventive maintenance before the "knee" of the curve is reached.

Predict Fleet Availability - For large-scale deployments, this function estimates the percentage of units that will remain operational at a future date, aiding in spare parts inventory planning.'''
            },
            {
                'title': 'Bootstrap Distribution of Beta (Shape Parameter)',
                'description': '''This histogram illustrates the uncertainty associated with the Weibull Shape Parameter (Beta), generated using the Bootstrap Resampling method (e.g., 1000 iterations). Instead of calculating a single static value, this technique simulates hundreds of "virtual experiments" by resampling the original data with replacement. The resulting distribution shows the range of probable values for Beta. The red vertical line indicates the mean estimated Beta, while the dashed green lines define the 95% Confidence Interval (CI) (spanning from the 2.5th to the 97.5th percentile). This visualization transforms a single point estimate into a probability distribution, revealing the statistical stability of the calculated failure mode.''',
                'technical_details': '''Quantifying the uncertainty of the shape parameter is critical for validating the physics of failure:

Confirm Failure Mode Statistically - A Beta value greater than 1.0 implies wear-out. However, this chart proves it rigorously. If the lower bound of the 95% CI is also strictly greater than 1.0, engineers can claim with 95% statistical confidence that the failure mechanism is indeed wear-out, ruling out random variation.

Assess Data Quality - The width of the distribution reflects the "tightness" of the data. A narrow, sharp peak indicates high consistency and sufficient sample size, whereas a wide, flat distribution suggests noisy data or too few samples, warning that the model parameters may be volatile.

Support Robust Decision Making - By understanding the potential range of Beta, engineering teams can base their maintenance strategies not just on the "average" scenario, but on conservative estimates that account for sampling error.'''
            },
            {
                'title': 'Bootstrap Distribution of Eta (Scale Parameter)',
                'description': '''This histogram depicts the uncertainty distribution of the Weibull Scale Parameter (Eta), also known as the Characteristic Life. Mathematically, Eta represents the time at which 63.2% of the population is expected to fail. By applying bootstrap resampling, this chart reveals the potential variability in the product's longevity due to sampling error. The central red line marks the mean estimated Characteristic Life, while the green dashed lines delineate the 95% Confidence Interval. This interval provides a realistic "best-case" and "worst-case" scenario for the product's lifespan.''',
                'technical_details': '''Understanding the range of the Scale Parameter is essential for logistics and financial planning:

Plan Spare Parts Inventory - The lower bound of the confidence interval (2.5% percentile) is critical for supply chain management. It indicates the earliest timeframe where a significant volume of replacements might be needed, preventing stockouts.

Assess Manufacturing Consistency - A narrow distribution for Eta implies a highly controlled manufacturing process where unit-to-unit variation is minimal. A wide spread suggests process instability or inconsistent raw materials, leading to unpredictable product lifetimes.

Compare Supplier Performance - When evaluating components from different vendors, comparing their Eta distributions allows for a statistical decision. Even if two suppliers have the same "average" life, the one with the narrower confidence interval is the superior choice due to its predictability.'''
            },
            {
                'title': 'Monte Carlo Lifetime Predictions',
                'description': '''This figure presents the results of a Monte Carlo Simulation (e.g., 10,000 runs) used to forecast the expected lifetime distribution of the population. Unlike simple calculations that rely on single parameter estimates, this simulation inputs the range of probable Weibull parameters (Beta and Eta) derived from the bootstrap analysis to generate thousands of hypothetical failure times. The resulting histogram shows the probability density of these predicted outcomes. Key reliability milestones are explicitly marked: B10 Life (the time by which 10% of units are expected to fail) and B50 Life (the median life, where 50% of units have failed).''',
                'technical_details': '''This predictive model translates statistical data into actionable business and engineering metrics:

Define Warranty Risks (B10 Focus) - The B10 value is the industry standard for defining warranty periods. Setting a warranty term below this threshold ensures that fewer than 10% of products will fail in the field, keeping replacement costs within the budget.

Schedule Preventive Maintenance (B50 Focus) - The B50 value indicates the "average" life expectancy. Maintenance teams can use this metric to plan fleet-wide overhauls or end-of-life replacements before the bulk of the population reaches wear-out.

Quantify Tail Risk - By visualizing the full spread of outcomes, engineers can assess the risk of "early outliers" (left tail) versus "long-surviving units" (right tail), ensuring the system design accommodates the full variability of the manufacturing process.'''
            },
            {
                'title': 'Histogram Comparison: Simulated vs Actual Data',
                'description': '''This figure presents a visual validation of the fitted model by overlaying the histogram of the Actual Observed Data (blue bars) with the Simulated Data (yellow bars) generated from the Weibull model. The red dashed line represents the theoretical Probability Density Function (PDF). The height of the bars corresponds to the probability density, allowing for a direct comparison of the distribution shapes. A high degree of overlap between the blue and yellow areas indicates that the simulation accurately replicates the physical failure behavior observed in reality.''',
                'technical_details': '''Visualizing the fit is a crucial first step in model validation:

Verify Distribution Shape - It confirms whether the model correctly captures key characteristics such as the peak failure time and the spread (variance). For instance, if the simulated data (yellow) accurately tracks the tail of the actual data (blue), the model is suitable for predicting long-term wear-out.

Detect Systemic Biases - Significant gaps between the actual and simulated histograms would reveal where the model under- or over-estimates risk (e.g., missing early-life failures), prompting a review of the chosen distribution.'''
            },
            {
                'title': 'CDF Comparison & KS Test',
                'description': '''This figure illustrates the quantitative Goodness-of-Fit using the Cumulative Distribution Function (CDF). It compares the Empirical CDF (blue stepped line, representing actual data) against the Theoretical CDF (red smooth curve, representing the fitted model). The plot includes the results of the Kolmogorov-Smirnov (KS) Test, a non-parametric statistical method. The "KS Distance" marks the maximum vertical divergence between the two curves, quantifying the error. The displayed p-value indicates the statistical significance of the fit; a higher p-value suggests that the model is consistent with the observed data.''',
                'technical_details': '''The CDF comparison provides the statistical evidence required for final model acceptance:

Statistical Acceptance Criteria - The p-value serves as the objective pass/fail metric. Typically, a p-value > 0.05 implies there is no significant evidence to reject the model, validating its use for reliability predictions.

Assess Prediction Accuracy - The proximity of the blue and red curves, particularly in the upper percentiles, assures engineers that the model is reliable for calculating critical metrics like warranty limits (B10 life) and mean life (MTTF).'''
            }
        ]
        
    def setup_pdf(self):
        self.pdf.set_margins(left=20, top=15, right=20)
        self.pdf.set_auto_page_break(auto=True, margin=20)
        self.pdf.set_font("Arial", size=10)
        self.effective_width = self.pdf.w - self.pdf.l_margin - self.pdf.r_margin

    def _clean_text(self, text, preserve_math_symbols=False):
        if not isinstance(text, str):
            text = str(text)
        if not preserve_math_symbols:
            text = text.encode('ascii', 'ignore').decode('ascii')
        replacements = {
            '"': '"', '"': '"', ''': "'", ''': "'", '–': '-', '—': '-',
            '…': '...', '°': ' degrees', '±': '+/-', '×': 'x', '÷': '/',
            '≤': '<=', '≥': '>=', '≠': '!=', '≈': '~=', '→': '->'
        }
        if not preserve_math_symbols:
            math_replacements = {
                'β': 'beta', 'η': 'eta', 'σ': 'sigma', 'μ': 'mu', 'λ': 'lambda', 'Φ': 'Phi', 'Γ': 'Gamma'
            }
            replacements.update(math_replacements)
        for old, new in replacements.items():
            text = text.replace(old, new)
        return text

    def add_header_section(self):
        self.pdf.set_font('Arial', '', 9)
        self.pdf.cell(0, 5, f'Report Date: {datetime.now().strftime("%Y-%m-%d")}', 0, 1, 'L')
        self.pdf.ln(3)
        self.pdf.set_font('Arial', 'B', 12)
        self.pdf.set_fill_color(240, 240, 240)
        self.pdf.cell(0, 8, 'Lifetime Analysis System - Statistical Reliability Assessment', 1, 1, 'C', True)
        self.pdf.ln(2)
        self.pdf.set_font('Arial', 'B', 11)
        self.pdf.cell(0, 6, 'TECHNICAL ANALYSIS REPORT', 1, 1, 'C', True)
        self.pdf.ln(5)

    def add_basic_information_table(self):
        metrics = self._extract_key_metrics()
        self.pdf.set_font('Arial', 'B', 10)
        self.pdf.set_fill_color(200, 200, 200)
        self.pdf.cell(self.effective_width * 0.3, 8, '1) Analysis title:', 1, 0, 'L', True)
        self.pdf.set_font('Arial', '', 10)
        self.pdf.cell(self.effective_width * 0.7, 8, 'Reliability Distribution Lifetime Analysis', 1, 1, 'L')
        self.pdf.ln(3)

    def add_objectives_section(self):
        self.pdf.set_font('Arial', 'B', 10)
        self.pdf.set_fill_color(200, 200, 200)
        self.pdf.cell(0, 8, '2) Main objectives:', 1, 1, 'L', True)
        self.pdf.set_font('Arial', '', 9)
        objectives_text = """- Perform statistical distribution fitting to identify optimal failure model
- Estimate Weibull parameters (shape and scale) with confidence intervals
- Generate reliability predictions and lifetime percentiles (B10, B50, B95)
- Validate model accuracy through goodness-of-fit testing
- Provide engineering recommendations for maintenance and design decisions"""
        lines = objectives_text.strip().split('\n')
        cell_height = len(lines) * 4 + 4
        y_start = self.pdf.get_y()
        self.pdf.cell(0, cell_height, '', 1, 1, 'L')
        self.pdf.set_y(y_start + 2)
        self.pdf.set_x(self.pdf.l_margin + 2)
        for line in lines:
            self.pdf.cell(0, 4, line.strip(), 0, 1, 'L')
            self.pdf.set_x(self.pdf.l_margin + 2)
        self.pdf.ln(3)

    def add_analysis_methods_section(self):
        self.pdf.set_font('Arial', 'B', 10)
        self.pdf.set_fill_color(200, 200, 200)
        self.pdf.cell(0, 8, '3) Statistical methods:', 1, 1, 'L', True)
        self.pdf.set_font('Arial', '', 9)
        methods_text = """- Maximum Likelihood Estimation (MLE) for parameter fitting
- Bootstrap resampling (1000 iterations) for uncertainty quantification
- Monte Carlo simulation (10,000 samples) for lifetime prediction
- Kolmogorov-Smirnov test for model validation"""
        lines = methods_text.strip().split('\n')
        cell_height = len(lines) * 4 + 4
        y_start = self.pdf.get_y()
        self.pdf.cell(0, cell_height, '', 1, 1, 'L')
        self.pdf.set_y(y_start + 2)
        self.pdf.set_x(self.pdf.l_margin + 2)
        for line in lines:
            self.pdf.cell(0, 4, line.strip(), 0, 1, 'L')
            self.pdf.set_x(self.pdf.l_margin + 2)
        self.pdf.ln(3)

    def add_results_summary_table(self):
        self.pdf.set_font('Arial', 'B', 10)
        self.pdf.set_fill_color(200, 200, 200)
        self.pdf.cell(0, 8, '4) Summary of analysis results:', 1, 1, 'L', True)
        metrics = self._extract_key_metrics()
        
        # Two column layout (removed CI column)
        col1_width = self.effective_width * 0.5
        col2_width = self.effective_width * 0.5
        
        self.pdf.set_font('Arial', 'B', 9)
        self.pdf.set_fill_color(220, 220, 220)
        self.pdf.cell(col1_width, 6, 'Parameter', 1, 0, 'C', True)
        self.pdf.cell(col2_width, 6, 'Estimated Value', 1, 1, 'C', True)
        
        # Format reliability as percentage
        reliability_value = metrics.get('reliability_at_cycles', 'N/A')
        if reliability_value != 'N/A':
            try:
                reliability_pct = float(reliability_value) * 100
                reliability_display = f'{reliability_pct:.1f}%'
            except:
                reliability_display = reliability_value
        else:
            reliability_display = 'N/A'
        
        expected_cycles = metrics.get('expected_cycles', 'N/A')
        
        # Reordered: MTTF first (red), then Reliability at Expected Cycles (red), B10, B50, B95, Shape, Scale, Model Validation
        results_data = [
            ('Mean Time To Failure (MTTF)', metrics.get('MTTF', 'N/A') + ' cycles', True),  # Red text
            (f'Reliability at {expected_cycles} Cycles', reliability_display, True),  # Red text
            ('B10 Life (10% failure)', metrics.get('B10', 'N/A') + ' cycles', False),
            ('B50 Life (median)', metrics.get('B50', 'N/A') + ' cycles', False),
            ('B95 Life (95% failure)', metrics.get('B95', 'N/A') + ' cycles', False),
            ('Shape Parameter (Beta)', metrics.get('beta', 'N/A'), False),
            ('Scale Parameter (Eta)', metrics.get('eta', 'N/A'), False),
            ('Model Validation (KS Test)', 'p-value: ' + metrics.get('p_value', 'N/A'), False)
        ]
        
        self.pdf.set_font('Arial', '', 8)
        for param, value, is_red in results_data:
            if is_red:
                # Set red color for MTTF row
                self.pdf.set_text_color(255, 0, 0)
            else:
                # Reset to black color
                self.pdf.set_text_color(0, 0, 0)
            
            self.pdf.cell(col1_width, 5, param, 1, 0, 'L')
            self.pdf.cell(col2_width, 5, str(value), 1, 1, 'C')
        
        # Reset text color to black
        self.pdf.set_text_color(0, 0, 0)
        self.pdf.ln(5)

    def add_flowchart_section(self):
        """Add flowchart image section to the PDF report."""
        # Add title for flowchart
        self.pdf.set_font('Arial', 'B', 10)
        self.pdf.set_fill_color(200, 200, 200)
        self.pdf.cell(0, 8, '5) Flowchart', 1, 1, 'L', True)
        
        # Add flowchart image from img/flowchart.png
        flowchart_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'img', 'flowchart.png')
        
        if os.path.exists(flowchart_path):
            try:
                # Calculate image dimensions to fit inside the table cell
                img_width = min(150, self.effective_width - 10)
                
                # First, we need to know the image height to draw the table cell
                # Get image info using PIL if available, otherwise estimate
                try:
                    from PIL import Image
                    with Image.open(flowchart_path) as img:
                        orig_width, orig_height = img.size
                        img_height = (img_width / orig_width) * orig_height
                except:
                    # Estimate height based on typical aspect ratio
                    img_height = img_width * 0.6
                
                # Draw the table cell border (full width)
                cell_height = img_height + 10  # Add padding
                y_start = self.pdf.get_y()
                self.pdf.cell(0, cell_height, '', 1, 1, 'C')
                
                # Position image centered inside the table cell
                x_pos = (self.pdf.w - img_width) / 2
                y_pos = y_start + 5  # 5mm padding from top of cell
                
                # Add the image inside the cell
                self.pdf.image(flowchart_path, x=x_pos, y=y_pos, w=img_width)
                
            except Exception as e:
                self.pdf.set_font('Arial', 'I', 9)
                self.pdf.cell(0, 10, f'[Flowchart image could not be loaded: {str(e)}]', 1, 1, 'C')
        else:
            self.pdf.set_font('Arial', 'I', 9)
            self.pdf.cell(0, 30, '[Flowchart image not found]', 1, 1, 'C')
        
        self.pdf.ln(5)

    def add_figure_with_description(self, fig_index, fig_buf):
        """Add a figure with its description to the PDF report."""
        if fig_index >= len(self.figure_descriptions):
            return
        
        fig_info = self.figure_descriptions[fig_index]
        
        # Check if we need a new page
        if fig_index == 0:
            if self.pdf.get_y() > 120:
                self.pdf.add_page()
        else:
            if self.pdf.get_y() > 200:
                self.pdf.add_page()
        
        # Add figure title
        self.pdf.set_font('Arial', 'B', 11)
        self.pdf.cell(0, 8, f"Figure {fig_index + 1}: {fig_info['title']}", 0, 1, 'L')
        self.pdf.ln(1)
        
        # Add the figure image
        try:
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as temp_file:
                fig_buf.seek(0)
                temp_file.write(fig_buf.read())
                temp_file_path = temp_file.name
            
            # Determine image width based on figure index
            if fig_index == 0:
                img_width = min(140, self.effective_width * 0.85)
            else:
                img_width = min(160, self.effective_width)
            
            x_pos = (self.pdf.w - img_width) / 2
            y_before_image = self.pdf.get_y()
            
            self.pdf.image(temp_file_path, x=x_pos, w=img_width)
            
            # Clean up temp file
            os.unlink(temp_file_path)
            
        except Exception as e:
            self.pdf.set_font('Arial', 'I', 9)
            self.pdf.cell(0, 10, f'[Figure could not be loaded: {str(e)}]', 0, 1, 'C')
        
        self.pdf.ln(3)
        
        # Add description
        self.pdf.set_font('Arial', 'B', 10)
        self.pdf.cell(0, 5, 'Description:', 0, 1, 'L')
        self.pdf.ln(3)
        self.pdf.set_font('Arial', '', 9)
        
        description_text = self._clean_text(fig_info['description'])
        # Remove extra blank lines between paragraphs
        description_text = '\n'.join(line for line in description_text.split('\n') if line.strip())
        self.pdf.multi_cell(0, 4, description_text)
        self.pdf.ln(2)
        
        # Add technical details / engineering significance
        if 'technical_details' in fig_info:
            self.pdf.set_font('Arial', 'B', 10)
            self.pdf.cell(0, 5, 'Engineering Significance:', 0, 1, 'L')
            self.pdf.ln(3)
            self.pdf.set_font('Arial', '', 9)
            
            technical_text = self._clean_text(fig_info['technical_details'])
            # Remove extra blank lines between paragraphs
            technical_text = '\n'.join(line for line in technical_text.split('\n') if line.strip())
            self.pdf.multi_cell(0, 4, technical_text)
        
        self.pdf.ln(5)

    def add_footer_info(self):
        """Add footer information to the PDF report."""
        self.pdf.ln(10)
        self.pdf.set_font('Arial', 'I', 8)
        self.pdf.set_text_color(128, 128, 128)
        
        footer_text = f"Generated by Lifetime Analysis System | Report Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        self.pdf.cell(0, 5, footer_text, 0, 1, 'C')
        
        self.pdf.cell(0, 5, "This report is for engineering analysis purposes only.", 0, 1, 'C')
        
        # Reset text color
        self.pdf.set_text_color(0, 0, 0)

    def _extract_key_metrics(self):
        """Extract key metrics from the text buffer for display in tables."""
        metrics = {
            'beta': 'N/A',
            'eta': 'N/A',
            'B10': 'N/A',
            'B50': 'N/A',
            'B95': 'N/A',
            'MTTF': 'N/A',
            'p_value': 'N/A',
            'expected_cycles': 'N/A',
            'reliability_at_cycles': 'N/A'
        }
        
        if not self.text_buffer:
            return metrics
        
        # Handle both list and StringIO types
        if hasattr(self.text_buffer, 'getvalue'):
            text = self.text_buffer.getvalue()
        elif isinstance(self.text_buffer, list):
            text = '\n'.join(str(line) for line in self.text_buffer)
        else:
            text = str(self.text_buffer)
        
        import re
        
        # Track context for parameter extraction (Beta vs Eta sections)
        lines = text.split('\n')
        in_beta_section = False
        in_eta_section = False
        
        for line in lines:
            line_stripped = line.strip()
            
            # Check for section context
            if "Beta (shape)" in line_stripped or "- Beta (shape):" in line_stripped:
                in_beta_section = True
                in_eta_section = False
            elif "Eta (scale)" in line_stripped or "- Eta (scale):" in line_stripped:
                in_beta_section = False
                in_eta_section = True
            elif line_stripped.startswith("===") or "[BOLD14]" in line_stripped:
                # Reset context when entering a new section
                if "[2]" in line_stripped or "Simulation" in line_stripped:
                    in_beta_section = False
                    in_eta_section = False
            
            # Extract Mean values for Beta and Eta
            if "Mean" in line_stripped and ":" in line_stripped:
                mean_match = re.search(r'Mean\s*:\s*([\d.]+)', line_stripped)
                if mean_match:
                    if in_beta_section:
                        metrics['beta'] = mean_match.group(1)
                    elif in_eta_section:
                        metrics['eta'] = mean_match.group(1)
            
            # Extract B10 (10% fail)
            if "B10" in line_stripped and ":" in line_stripped:
                b10_match = re.search(r'B10.*?:\s*([\d.]+)', line_stripped)
                if b10_match:
                    metrics['B10'] = b10_match.group(1)
            
            # Extract B50 (median)
            if "B50" in line_stripped and ":" in line_stripped:
                b50_match = re.search(r'B50.*?:\s*([\d.]+)', line_stripped)
                if b50_match:
                    metrics['B50'] = b50_match.group(1)
            
            # Extract B95 (95% fail)
            if "B95" in line_stripped and ":" in line_stripped:
                b95_match = re.search(r'B95.*?:\s*([\d.]+)', line_stripped)
                if b95_match:
                    metrics['B95'] = b95_match.group(1)
            
            # Extract MTTF (Average)
            if "MTTF" in line_stripped and ":" in line_stripped:
                mttf_match = re.search(r'MTTF.*?:\s*([\d.]+)', line_stripped)
                if mttf_match:
                    metrics['MTTF'] = mttf_match.group(1)
            
            # Extract p-value from KS test
            if "P-value" in line_stripped or "p-value" in line_stripped:
                pvalue_match = re.search(r'[Pp]-?value\s*:\s*([\d.]+)', line_stripped)
                if pvalue_match:
                    metrics['p_value'] = pvalue_match.group(1)
            
            # Extract Expected Cycles from reliability estimation section
            if "Reliability Estimation at" in line_stripped:
                cycles_match = re.search(r'at\s*([\d.]+)\s*Cycles', line_stripped)
                if cycles_match:
                    metrics['expected_cycles'] = cycles_match.group(1)
            
            # Extract Median reliability
            if "Median reliability" in line_stripped:
                reliability_match = re.search(r'Median reliability\s*:\s*([\d.]+)', line_stripped)
                if reliability_match:
                    metrics['reliability_at_cycles'] = reliability_match.group(1)
        
        return metrics

    def build(self):
        """Build the complete PDF report."""
        self.setup_pdf()
        
        # Add first page with header and summary sections
        self.pdf.add_page()
        self.add_header_section()
        self.add_basic_information_table()
        self.add_objectives_section()
        self.add_analysis_methods_section()
        self.add_results_summary_table()
        self.add_flowchart_section()
        
        # Add figures with descriptions (8 figures total)
        for i, fig_buf in enumerate(self.figures):
            if i < len(self.figure_descriptions):
                # Start new page for each figure after the first few
                if i > 0:
                    self.pdf.add_page()
                self.add_figure_with_description(i, fig_buf)
        
        # Add footer
        self.add_footer_info()
        
        return self.pdf

    def save(self, filepath):
        """Save the PDF report to a file."""
        pdf = self.build()
        pdf.output(filepath)
        return filepath
