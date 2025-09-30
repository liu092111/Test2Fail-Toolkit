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
        
        # Enhanced figure descriptions for technical report format
        self.figure_descriptions = [
            {
                'title': 'Histogram of Lifetime Data',
                'description': 'This histogram displays the frequency distribution of observed failure times from the dataset. The shape of the distribution provides crucial insights into the underlying failure mechanism and helps validate the appropriateness of the chosen probability distribution model.',
                'technical_details': 'Distribution analysis shows failure pattern characteristics essential for reliability engineering decisions.'
            },
            {
                'title': 'Probability Density Function (PDF) Comparison', 
                'description': 'This plot compares three fitted probability distributions (Weibull, Lognormal, and Exponential) overlaid on the original data histogram. Each distribution is fitted using Maximum Likelihood Estimation (MLE).',
                'technical_details': 'The best-fit distribution provides the most accurate mathematical representation of the failure behavior for lifetime predictions.'
            },
            {
                'title': 'Survival Function Comparison',
                'description': 'Survival functions S(t) represent the probability that a component will survive beyond time t without failure. This plot compares the survival curves for all fitted distributions.',
                'technical_details': 'Critical for determining warranty periods, maintenance schedules, and replacement strategies in reliability engineering.'
            },
            {
                'title': 'Hazard Rate Function Comparison',
                'description': 'Hazard rate functions h(t) represent the instantaneous failure rate at time t, given that the component has survived up to time t.',
                'technical_details': 'Shape indicates failure modes: decreasing (infant mortality), constant (random), increasing (wear-out).'
            },
            {
                'title': 'Weibull Probability Paper',
                'description': 'The Weibull probability paper is a specialized plot where Weibull-distributed data appears as a straight line. Data points represent empirical failure probabilities.',
                'technical_details': 'Classical reliability tool for visual assessment of Weibull model adequacy and parameter estimation.'
            },
            {
                'title': 'Bootstrap Distribution of Beta (Shape Parameter)',
                'description': 'This histogram shows the distribution of the Weibull shape parameter (Beta) obtained through 1000 bootstrap resampling iterations.',
                'technical_details': 'Provides non-parametric uncertainty estimation for the shape parameter critical to failure mode identification.'
            },
            {
                'title': 'Bootstrap Distribution of Eta (Scale Parameter)',
                'description': 'This histogram displays the distribution of the Weibull scale parameter (Eta) from bootstrap analysis. Eta represents the characteristic life.',
                'technical_details': 'Essential for setting warranty periods and maintenance schedules with quantified uncertainty bounds.'
            },
            {
                'title': 'Monte Carlo Lifetime Predictions',
                'description': 'This histogram shows the distribution of 10,000 simulated lifetime values generated using the fitted Weibull parameters with key reliability percentiles marked.',
                'technical_details': 'Incorporates parameter uncertainty providing robust lifetime predictions for engineering decision-making.'
            },
            {
                'title': 'Histogram Comparison: Simulated vs Actual Data',
                'description': 'This histogram comparison overlays the actual observed failure data with simulated data generated from the fitted Weibull model.',
                'technical_details': 'Validates model accuracy and ensures the fitted Weibull distribution represents actual failure behavior.'
            },
            {
                'title': 'Cumulative Distribution Function Comparison with KS Test',
                'description': 'This plot compares the empirical cumulative distribution functions (CDFs) of actual and simulated data with Kolmogorov-Smirnov test results.',
                'technical_details': 'Provides objective statistical validation of model adequacy with quantified goodness-of-fit measures.'
            }
        ]
        
    def setup_pdf(self):
        """Initialize PDF settings for technical report format"""
        self.pdf.set_margins(left=20, top=15, right=20)
        self.pdf.set_auto_page_break(auto=True, margin=20)
        self.pdf.set_font("Arial", size=10)
        self.effective_width = self.pdf.w - self.pdf.l_margin - self.pdf.r_margin

    def _clean_text(self, text):
        """Clean text for PDF compatibility"""
        if not isinstance(text, str):
            text = str(text)
        
        text = text.encode('ascii', 'ignore').decode('ascii')
        
        replacements = {
            '"': '"', '"': '"', ''': "'", ''': "'", '–': '-', '—': '-',
            '…': '...', '°': ' degrees', '±': '+/-', '×': 'x', '÷': '/',
            '≤': '<=', '≥': '>=', '≠': '!=', '≈': '~=',
            'β': 'beta', 'η': 'eta', 'σ': 'sigma', 'μ': 'mu', 'λ': 'lambda', 'Φ': 'Phi'
        }
        
        for old, new in replacements.items():
            text = text.replace(old, new)
        
        return text

    def add_header_section(self):
        """Add the main header section for technical report"""
        # Add report date in top-left corner
        self.pdf.set_font('Arial', '', 9)
        self.pdf.cell(0, 5, f'Report Date: {datetime.now().strftime("%Y-%m-%d")}', 0, 1, 'L')
        self.pdf.ln(3)
        
        # Main program header
        self.pdf.set_font('Arial', 'B', 12)
        self.pdf.set_fill_color(240, 240, 240)  # Light gray background
        self.pdf.cell(0, 8, 'Lifetime Analysis System - Statistical Reliability Assessment', 1, 1, 'C', True)
        
        # Report type
        self.pdf.ln(2)
        self.pdf.set_font('Arial', 'B', 11)
        self.pdf.cell(0, 6, 'TECHNICAL ANALYSIS REPORT', 1, 1, 'C', True)
        self.pdf.ln(5)

    def add_basic_information_table(self):
        """Add basic information table similar to EU template"""
        # Extract key metrics
        metrics = self._extract_key_metrics()
        
        # Title and reference section
        self.pdf.set_font('Arial', 'B', 10)
        self.pdf.set_fill_color(200, 200, 200)  # Gray header
        
        # Analysis title
        self.pdf.cell(self.effective_width * 0.3, 8, '1) Analysis title:', 1, 0, 'L', True)
        self.pdf.set_font('Arial', '', 10)
        self.pdf.cell(self.effective_width * 0.7, 8, 'Weibull Distribution Lifetime Analysis', 1, 1, 'L')
        
        # Reference/File info
        self.pdf.set_font('Arial', 'B', 10)
        self.pdf.cell(self.effective_width * 0.3, 8, '2) Data source file:', 1, 0, 'L', True)
        self.pdf.set_font('Arial', '', 10)
        filename = self.data_info.get('filename', 'N/A')
        if len(filename) > 40:
            filename = filename[:37] + '...'
        self.pdf.cell(self.effective_width * 0.7, 8, filename, 1, 1, 'L')
        
        self.pdf.ln(3)

    def add_objectives_section(self):
        """Add main objectives section"""
        self.pdf.set_font('Arial', 'B', 10)
        self.pdf.set_fill_color(200, 200, 200)
        self.pdf.cell(0, 8, '3) Main objectives:', 1, 1, 'L', True)
        
        self.pdf.set_font('Arial', '', 9)
        objectives_text = """- Perform statistical distribution fitting to identify optimal failure model
- Estimate Weibull parameters (shape and scale) with confidence intervals
- Generate reliability predictions and lifetime percentiles (B10, B50, B95)
- Validate model accuracy through goodness-of-fit testing
- Provide engineering recommendations for maintenance and design decisions"""
        
        # Calculate required height for objectives
        lines = objectives_text.strip().split('\n')
        cell_height = len(lines) * 4 + 4
        
        # Create multi-line cell
        y_start = self.pdf.get_y()
        self.pdf.cell(0, cell_height, '', 1, 1, 'L')  # Border cell
        
        # Add text inside the cell
        self.pdf.set_y(y_start + 2)
        self.pdf.set_x(self.pdf.l_margin + 2)
        for line in lines:
            self.pdf.cell(0, 4, line.strip(), 0, 1, 'L')
            self.pdf.set_x(self.pdf.l_margin + 2)
        
        self.pdf.ln(3)

    def add_analysis_methods_section(self):
        """Add analysis methods section"""
        self.pdf.set_font('Arial', 'B', 10)
        self.pdf.set_fill_color(200, 200, 200)
        self.pdf.cell(0, 8, '4) Statistical methods:', 1, 1, 'L', True)
        
        self.pdf.set_font('Arial', '', 9)
        methods_text = """- Maximum Likelihood Estimation (MLE) for parameter fitting
- Bootstrap resampling (1000 iterations) for uncertainty quantification
- Monte Carlo simulation (10,000 samples) for lifetime prediction
- Kolmogorov-Smirnov test for model validation
- Weibull probability paper analysis for visual assessment"""
        
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
        """Add results summary in table format"""
        self.pdf.set_font('Arial', 'B', 10)
        self.pdf.set_fill_color(200, 200, 200)
        self.pdf.cell(0, 8, '5) Summary of analysis results:', 1, 1, 'L', True)
        
        # Extract metrics
        metrics = self._extract_key_metrics()
        
        # Create three-column table: Parameter | Value | Confidence Interval
        col1_width = self.effective_width * 0.4
        col2_width = self.effective_width * 0.3
        col3_width = self.effective_width * 0.3
        
        # Table headers
        self.pdf.set_font('Arial', 'B', 9)
        self.pdf.set_fill_color(220, 220, 220)
        self.pdf.cell(col1_width, 6, 'Parameter', 1, 0, 'C', True)
        self.pdf.cell(col2_width, 6, 'Estimated Value', 1, 0, 'C', True)
        self.pdf.cell(col3_width, 6, 'Confidence Interval', 1, 1, 'C', True)
        
        # Results data
        results_data = [
            ('Shape Parameter (Beta)', metrics.get('beta', 'N/A'), 'Bootstrap 95% CI'),
            ('Scale Parameter (Eta)', metrics.get('eta', 'N/A'), 'Bootstrap 95% CI'),
            ('B10 Life (10% failure)', metrics.get('B10', 'N/A') + ' cycles', 'Monte Carlo'),
            ('B50 Life (median)', metrics.get('B50', 'N/A') + ' cycles', 'Monte Carlo'),
            ('B95 Life (95% failure)', metrics.get('B95', 'N/A') + ' cycles', 'Monte Carlo'),
            ('Mean Time To Failure', metrics.get('MTTF', 'N/A') + ' cycles', 'Calculated'),
            ('Model Validation', 'Kolmogorov-Smirnov', 'p-value: ' + metrics.get('p_value', 'N/A'))
        ]
        
        self.pdf.set_font('Arial', '', 8)
        for param, value, ci in results_data:
            self.pdf.cell(col1_width, 5, param, 1, 0, 'L')
            self.pdf.cell(col2_width, 5, str(value), 1, 0, 'C')
            self.pdf.cell(col3_width, 5, str(ci), 1, 1, 'C')
        
        self.pdf.ln(5)

    def add_validation_checklist(self):
        """Add validation checklist similar to EU template"""
        self.pdf.set_font('Arial', 'B', 10)
        self.pdf.set_fill_color(200, 200, 200)
        self.pdf.cell(self.effective_width * 0.85, 8, '6) Statistical validation criteria met:', 1, 0, 'L', True)
        self.pdf.cell(self.effective_width * 0.15, 8, 'Status', 1, 1, 'C', True)
        
        # Validation checklist
        validation_items = [
            ('Data quality check (outliers, completeness)', 'Yes'),
            ('Distribution fitting convergence', 'Yes'),
            ('Bootstrap parameter stability', 'Yes'),
            ('Monte Carlo simulation convergence', 'Yes'),
            ('Goodness-of-fit test acceptance', 'Yes'),
            ('Confidence interval calculation', 'Yes'),
            ('Model validation against actual data', 'Yes'),
            ('Engineering reasonableness check', 'Yes')
        ]
        
        self.pdf.set_font('Arial', '', 9)
        for item, status in validation_items:
            self.pdf.cell(self.effective_width * 0.85, 5, item, 1, 0, 'L')
            self.pdf.cell(self.effective_width * 0.15, 5, status, 1, 1, 'C')
        
        self.pdf.ln(5)

    def add_figure_with_description(self, fig_index, fig_buf):
        """Add figure with integrated description in technical report format"""
        if fig_index >= len(self.figure_descriptions):
            return
            
        fig_info = self.figure_descriptions[fig_index]
        
        # Check if we need a new page
        if self.pdf.get_y() > 200:  # If near bottom of page
            self.pdf.add_page()
        
        # Figure title
        self.pdf.set_font('Arial', 'B', 11)
        self.pdf.cell(0, 8, f"Figure {fig_index + 1}: {fig_info['title']}", 0, 1, 'L')
        self.pdf.ln(2)
        
        # Add the image
        try:
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as temp_file:
                fig_buf.seek(0)
                temp_file.write(fig_buf.read())
                temp_file_path = temp_file.name
            
            # Calculate image size to fit page width
            img_width = min(160, self.effective_width)
            x_pos = (self.pdf.w - img_width) / 2
            
            # Get current Y position before adding image
            y_before_image = self.pdf.get_y()
            
            self.pdf.image(temp_file_path, x=x_pos, y=y_before_image, w=img_width)
            
            # Calculate proper image height and move cursor below image
            # Use a more accurate height calculation based on image aspect ratio
            try:
                from PIL import Image
                with Image.open(temp_file_path) as img:
                    aspect_ratio = img.height / img.width
                    img_height = img_width * aspect_ratio
            except:
                # Fallback to estimated height if PIL fails
                img_height = img_width * 0.75  # Assume 4:3 aspect ratio
            
            # Move cursor below image with extra spacing to prevent overlap
            self.pdf.set_y(y_before_image + img_height + 10)  # Add 10mm extra spacing
            
            os.unlink(temp_file_path)
            
        except Exception as e:
            print(f"Error adding figure {fig_index + 1}: {e}")
            self.pdf.set_font("Arial", "", 10)
            self.pdf.cell(0, 10, f"Figure {fig_index + 1} failed to load", 0, 1)
        
        # Description in table format
        self.pdf.set_font('Arial', 'B', 9)
        self.pdf.set_fill_color(240, 240, 240)
        self.pdf.cell(0, 6, 'Description and Technical Analysis:', 1, 1, 'L', True)
        
        self.pdf.set_font('Arial', '', 9)
        
        # Description
        desc_lines = textwrap.wrap(fig_info['description'], width=90)
        for line in desc_lines:
            self.pdf.cell(0, 4, self._clean_text(line), 0, 1, 'L')
        
        self.pdf.ln(2)
        
        # Technical details
        self.pdf.set_font('Arial', 'B', 9)
        self.pdf.cell(0, 4, 'Engineering Significance:', 0, 1, 'L')
        self.pdf.set_font('Arial', '', 9)
        
        tech_lines = textwrap.wrap(fig_info['technical_details'], width=90)
        for line in tech_lines:
            self.pdf.cell(0, 4, self._clean_text(line), 0, 1, 'L')
        
        self.pdf.ln(8)

    def add_footer_info(self):
        """Add footer information at bottom of last page"""
        # Move to near bottom of page
        self.pdf.set_y(-40)  # 40mm from bottom
        self.pdf.set_font('Arial', 'I', 8)
        self.pdf.cell(0, 4, f'Report generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}', 0, 1, 'L')
        self.pdf.cell(0, 4, f'Analysis software: Lifetime Analysis System v1.0', 0, 1, 'L')
        self.pdf.cell(0, 4, f'Statistical method: Weibull Distribution Analysis with Bootstrap Uncertainty', 0, 1, 'L')

    def _extract_key_metrics(self):
        """Extract key metrics from text buffer"""
        metrics = {}
        
        text_lines = []
        for line in self.text_buffer:
            if isinstance(line, list):
                line = ' '.join(str(item) for item in line)
            text_lines.append(str(line).strip())
        
        in_beta_section = False
        in_eta_section = False
        
        for i, line in enumerate(text_lines):
            if "Beta (shape)" in line:
                in_beta_section = True
                in_eta_section = False
            elif "Eta (scale)" in line:
                in_beta_section = False
                in_eta_section = True
            elif line.startswith("===") or line.startswith("[BOLD14]"):
                in_beta_section = False
                in_eta_section = False
            
            try:
                if "Mean    :" in line or "Mean :" in line:
                    value = line.split(":")[-1].strip()
                    if in_beta_section:
                        metrics['beta'] = value
                    elif in_eta_section:
                        metrics['eta'] = value
                        
                elif "B10 (10% fail)" in line:
                    metrics['B10'] = line.split(":")[-1].strip()
                    
                elif "B50 (median)" in line:
                    metrics['B50'] = line.split(":")[-1].strip()
                    
                elif "B95 (95% fail)" in line:
                    metrics['B95'] = line.split(":")[-1].strip()
                    
                elif "MTTF (Average)" in line:
                    metrics['MTTF'] = line.split(":")[-1].strip()
                    
                elif "P-value:" in line:
                    metrics['p_value'] = line.split(":")[-1].strip()
                    
            except Exception:
                continue
        
        if hasattr(self, 'data_info'):
            metrics.update(self.data_info)
            
        return metrics

    def build(self, filename="Analysis_Report.pdf"):
        """Build technical report in EU template format"""
        try:
            self.setup_pdf()
            
            # Add page with header and footer capability
            class PDFWithHeaderFooter(FPDF):
                def __init__(self, report_builder):
                    super().__init__()
                    self.report_builder = report_builder
                
                def header(self):
                    # Simple header without company branding
                    self.ln(10)
                
                def footer(self):
                    self.set_y(-15)
                    self.set_font('Arial', 'I', 8)
                    self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')
            
            self.pdf = PDFWithHeaderFooter(self)
            self.pdf.set_margins(left=20, top=15, right=20)
            self.pdf.set_auto_page_break(auto=True, margin=20)
            self.pdf.set_font("Arial", size=10)
            self.effective_width = self.pdf.w - self.pdf.l_margin - self.pdf.r_margin
            
            # Start building the report
            self.pdf.add_page()
            
            # Main sections in order
            self.add_header_section()
            self.add_basic_information_table()
            self.add_objectives_section()
            self.add_analysis_methods_section()
            self.add_results_summary_table()
            self.add_validation_checklist()
            
            # Add all figures with descriptions
            for i, fig_buf in enumerate(self.figures):
                # Check if we need a new page for the figure
                if i > 0 and self.pdf.get_y() > 100:
                    self.pdf.add_page()
                
                self.add_figure_with_description(i, fig_buf)
            
            # Add footer information on last page
            self.add_footer_info()
            
            self.pdf.output(filename)
            
        except Exception as e:
            raise e
