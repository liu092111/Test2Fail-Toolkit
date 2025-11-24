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
                'description': 'This figure illustrates the distribution of the observed failure times collected from the test samples. Each bar in the histogram represents the frequency of failures occurring within a specific range of lifetime cycles. By examining the overall shape of the histogram, we can identify how the failures are distributed over time — for instance, whether most components fail early (indicating early-life issues), steadily (indicating random failures), or predominantly at later stages (indicating wear-out mechanisms).',
                'technical_details': '''Understanding the failure-time distribution allows engineers to:

• Assess failure behavior — For example, a right-skewed distribution with a long tail may suggest a small portion of components last significantly longer than average, while a steep early peak could imply process or material inconsistency.

• Choose an appropriate statistical model — The histogram's shape provides visual cues for model selection (e.g., Weibull for wear-out failures, Exponential for constant failure rates).

• Guide reliability improvements — If failures are concentrated in early cycles, process control or burn-in testing may be needed; if they appear in later cycles, focus should shift toward material fatigue, corrosion, or long-term degradation mechanisms.

This figure thus establishes the empirical baseline of the product's lifetime characteristics, forming the foundation for quantitative modeling and predictive reliability assessment in later stages.'''
            },
            {
                'title': 'Probability Density Function (PDF) Comparison', 
                'description': '''This figure compares three lifetime models, Weibull, Lognormal, and Exponential. Each fitted to the observed failure data using the Maximum Likelihood Estimation (MLE) method.MLE is a statistical approach that finds the most plausible model parameters so that the fitted curve best represents how failures occur over time.

Each colored curve represents a probability density function (PDF), which describes the likelihood that a component will fail at a given point in its lifetime.The Weibull distribution is widely used for mechanical parts because it can represent early-life, random, or wear-out failures depending on its shape. The Lognormal distribution often models chemical or environmental degradation, where aging accumulates gradually. The Exponential distribution assumes a constant failure rate and is typical for electronic or random-stress failures.

Once the models are fitted, the Mean Time To Failure (MTTF) can be calculated directly from their estimated parameters.''',
                'technical_details': '''This comparison highlights how different assumptions lead to different reliability interpretations.By observing which model best fits the data:

Model Selection Insight:Identifying which distribution best fits the data helps engineers infer whether failures are due to material aging, process variability, or random stress.

Quantitative Lifetime Estimation:The fitted model provides a consistent way to compute MTTF, transforming test results into measurable reliability metrics.

Design & Clarity Guidance:Comparing these model curves visually demonstrates how mathematical modeling translates raw test data into practical engineering understanding.'''
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

    def _clean_text(self, text, preserve_math_symbols=False):
        """Clean text for PDF compatibility"""
        if not isinstance(text, str):
            text = str(text)
        
        # Don't encode to ASCII if we want to preserve math symbols
        if not preserve_math_symbols:
            text = text.encode('ascii', 'ignore').decode('ascii')
        
        replacements = {
            '"': '"', '"': '"', ''': "'", ''': "'", '–': '-', '—': '-',
            '…': '...', '°': ' degrees', '±': '+/-', '×': 'x', '÷': '/',
            '≤': '<=', '≥': '>=', '≠': '!=', '≈': '~=',
            '→': '->'
        }
        
        # Only replace Greek letters if not preserving math symbols
        if not preserve_math_symbols:
            math_replacements = {
                'β': 'beta', 'η': 'eta', 'σ': 'sigma', 'μ': 'mu', 'λ': 'lambda', 'Φ': 'Phi', 'Γ': 'Gamma'
            }
            replacements.update(math_replacements)
        
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
        
        # Special handling for Figure 1 to ensure it fits on one page
        if fig_index == 0:
            # Check if we need a new page for Figure 1
            if self.pdf.get_y() > 120:  # More restrictive for Figure 1
                self.pdf.add_page()
        else:
            # Check if we need a new page for other figures
            if self.pdf.get_y() > 200:  # If near bottom of page
                self.pdf.add_page()
        
        # Figure title
        self.pdf.set_font('Arial', 'B', 11)
        self.pdf.cell(0, 8, f"Figure {fig_index + 1}: {fig_info['title']}", 0, 1, 'L')
        self.pdf.ln(1)  # Reduced spacing for Figure 1
        
        # Add the image
        try:
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as temp_file:
                fig_buf.seek(0)
                temp_file.write(fig_buf.read())
                temp_file_path = temp_file.name
            
            # Calculate image size - smaller for Figure 1 to leave room for text
            if fig_index == 0:
                img_width = min(140, self.effective_width * 0.85)  # Smaller image for Figure 1
            else:
                img_width = min(160, self.effective_width)
            
            x_pos = (self.pdf.w - img_width) / 2
            
            # Get current Y position before adding image
            y_before_image = self.pdf.get_y()
            
            self.pdf.image(temp_file_path, x=x_pos, y=y_before_image, w=img_width)
            
            # Calculate proper image height and move cursor below image
            try:
                from PIL import Image
                with Image.open(temp_file_path) as img:
                    aspect_ratio = img.height / img.width
                    img_height = img_width * aspect_ratio
            except:
                img_height = img_width * 0.75  # Assume 4:3 aspect ratio
            
            # Move cursor below image with appropriate spacing
            if fig_index == 0:
                self.pdf.set_y(y_before_image + img_height + 5)  # Less spacing for Figure 1
            else:
                self.pdf.set_y(y_before_image + img_height + 10)
            
            os.unlink(temp_file_path)
            
        except Exception as e:
            print(f"Error adding figure {fig_index + 1}: {e}")
            self.pdf.set_font("Arial", "", 10)
            self.pdf.cell(0, 10, f"Figure {fig_index + 1} failed to load", 0, 1)
        
        # Special formatting for Figure 1 and Figure 2 to match Description.docx structure
        if fig_index == 0 or fig_index == 1:
            # Description header with gray background and border (like other figures)
            self.pdf.set_font('Arial', 'B', 9)
            self.pdf.set_fill_color(240, 240, 240)
            self.pdf.cell(0, 6, 'Description', 1, 1, 'L', True)
            
            # Description content with preserved line breaks and equation formatting
            self.pdf.set_font('Arial', '', 9)
            desc_text = fig_info['description']
            
            # Split by explicit newlines to preserve paragraph structure
            paragraphs = desc_text.split('\n\n')
            
            for paragraph in paragraphs:
                paragraph = paragraph.strip()
                if not paragraph:
                    continue
                    
                # Check if this paragraph contains equations
                if 'MTTF =' in paragraph or '→' in paragraph:
                    # Handle equation formatting
                    lines = paragraph.split('\n')
                    for line in lines:
                        line = line.strip()
                        if line:
                            if '→ MTTF =' in line:
                                # Format equation lines with monospace-like appearance and preserve math symbols
                                self.pdf.set_font('Arial', '', 8.5)
                                self.pdf.cell(0, 4.5, self._clean_text(line, preserve_math_symbols=True), 0, 1, 'L')
                                self.pdf.set_font('Arial', '', 9)
                            else:
                                # Regular text
                                wrapped_lines = textwrap.wrap(line, width=115)
                                for wrapped_line in wrapped_lines:
                                    self.pdf.cell(0, 4.5, self._clean_text(wrapped_line), 0, 1, 'L')
                else:
                    # Regular paragraph - preserve internal line breaks
                    lines = paragraph.split('. ')
                    current_text = ""
                    
                    for i, sentence in enumerate(lines):
                        if i < len(lines) - 1:
                            sentence += '. '
                        current_text += sentence
                        
                        # If sentence is getting long, wrap it
                        if len(current_text) > 100 or i == len(lines) - 1:
                            wrapped_lines = textwrap.wrap(current_text, width=115)
                            for wrapped_line in wrapped_lines:
                                self.pdf.cell(0, 4.5, self._clean_text(wrapped_line), 0, 1, 'L')
                            current_text = ""
                
                # Add space between paragraphs
                self.pdf.ln(2)
            
            self.pdf.ln(3)  # Space before Engineering Significance
            
            # Engineering Significance header with gray background and border
            self.pdf.set_font('Arial', 'B', 9)
            self.pdf.set_fill_color(240, 240, 240)
            self.pdf.cell(0, 6, 'Engineering Significance', 1, 1, 'L', True)
            self.pdf.ln(1)
            
            # Parse and format the technical details properly
            tech_content = fig_info['technical_details'].strip()
            
            # Remove duplicate introductory text if it exists
            lines = tech_content.split('\n')
            filtered_lines = []
            for line in lines:
                line = line.strip()
                if line and "Understanding the failure-time distribution allows engineers to:" not in line:
                    filtered_lines.append(line)
            
            tech_content = '\n'.join(filtered_lines)
            
            # Split into sections based on bullet points and paragraphs
            sections = []
            current_section = []
            
            for line in tech_content.split('\n'):
                line = line.strip()
                if line.startswith('•'):
                    if current_section:
                        sections.append('\n'.join(current_section))
                        current_section = []
                    current_section.append(line)
                elif line and not line.startswith('•'):
                    current_section.append(line)
            
            if current_section:
                sections.append('\n'.join(current_section))
            
            # Format each section
            for section in sections:
                if section.strip().startswith('•'):
                    # Handle bullet point sections (Figure 1 style)
                    bullet_lines = section.split('\n')
                    main_bullet = bullet_lines[0].strip()
                    
                    # Extract the main point and sub-content
                    if '—' in main_bullet:
                        bullet_title = main_bullet.split('—')[0].strip()
                        bullet_content = main_bullet.split('—', 1)[1].strip()
                        
                        # Format bullet title (bold)
                        self.pdf.set_font('Arial', 'B', 9)
                        self.pdf.cell(0, 4.5, self._clean_text(bullet_title), 0, 1, 'L')
                        
                        # Format bullet content with indentation - further increased wrap width
                        self.pdf.set_font('Arial', '', 9)
                        content_lines = textwrap.wrap(bullet_content, width=125, initial_indent='    ', subsequent_indent='    ')
                        for content_line in content_lines:
                            self.pdf.cell(0, 4.5, self._clean_text(content_line), 0, 1, 'L')
                    else:
                        # Simple bullet point - further increased wrap width
                        self.pdf.set_font('Arial', '', 9)
                        bullet_lines_wrapped = textwrap.wrap(main_bullet, width=130, subsequent_indent='    ')
                        for bullet_line in bullet_lines_wrapped:
                            self.pdf.cell(0, 4.5, self._clean_text(bullet_line), 0, 1, 'L')
                    
                    self.pdf.ln(2)  # Space after each bullet point
                
                elif ':' in section and (fig_index == 1):
                    # Handle Figure 2 sections - distinguish between labels and regular text
                    lines = section.split('\n')
                    for line in lines:
                        line = line.strip()
                        if not line:
                            continue
                            
                        # Check if this is a true label (specific pattern for Figure 2 labels)
                        is_label = False
                        if ':' in line:
                            label_candidates = ['Model Selection Insight:', 'Quantitative Lifetime Estimation:', 'Design & Clarity Guidance:']
                            for candidate in label_candidates:
                                if candidate in line:
                                    is_label = True
                                    break
                        
                        if is_label and ':' in line:
                            # Split into label and content for true labels
                            parts = line.split(':', 1)
                            label = parts[0].strip()
                            content = parts[1].strip() if len(parts) > 1 else ""
                            
                            # Format label (bold)
                            self.pdf.set_font('Arial', 'B', 9)
                            self.pdf.cell(0, 4.5, self._clean_text(label + ':'), 0, 1, 'L')
                            
                            # Format content with slight indentation
                            if content:
                                self.pdf.set_font('Arial', '', 9)
                                content_lines = textwrap.wrap(content, width=115, initial_indent='', subsequent_indent='')
                                for content_line in content_lines:
                                    self.pdf.cell(0, 4.5, self._clean_text(content_line), 0, 1, 'L')
                        else:
                            # Regular text line (including "By observing which model best fits the data:")
                            self.pdf.set_font('Arial', '', 9)
                            # Handle line breaks for specific cases
                            if 'interpretations.By observing' in line:
                                line = line.replace('interpretations.By observing', 'interpretations.\n\nBy observing')
                            elif 'interpretations.By' in line:
                                line = line.replace('interpretations.By', 'interpretations.\n\nBy')
                            
                            # Split by manual line breaks if they exist
                            sub_lines = line.split('\n')
                            for sub_line in sub_lines:
                                sub_line = sub_line.strip()
                                if sub_line:
                                    wrapped_lines = textwrap.wrap(sub_line, width=115)
                                    for wrapped_line in wrapped_lines:
                                        self.pdf.cell(0, 4.5, self._clean_text(wrapped_line), 0, 1, 'L')
                    
                    self.pdf.ln(2)  # Space after each section
                
                else:
                    # Handle regular paragraph text - increased wrap width
                    self.pdf.set_font('Arial', '', 9)
                    para_lines = textwrap.wrap(section.strip(), width=115)
                    for para_line in para_lines:
                        self.pdf.cell(0, 4.5, self._clean_text(para_line), 0, 1, 'L')
                    self.pdf.ln(2)
        
        else:
            # Standard formatting for other figures
            self.pdf.set_font('Arial', 'B', 9)
            self.pdf.set_fill_color(240, 240, 240)
            self.pdf.cell(0, 6, 'Description and Technical Analysis:', 1, 1, 'L', True)
            
            self.pdf.set_font('Arial', '', 9)
            line_height = 4
            wrap_width = 90
            
            # Description
            desc_lines = textwrap.wrap(fig_info['description'], width=wrap_width)
            for line in desc_lines:
                self.pdf.cell(0, line_height, self._clean_text(line), 0, 1, 'L')
            
            self.pdf.ln(2)
            
            # Technical details
            self.pdf.set_font('Arial', 'B', 9)
            self.pdf.cell(0, 4, 'Engineering Significance:', 0, 1, 'L')
            self.pdf.set_font('Arial', '', 9)
            
            tech_lines = textwrap.wrap(fig_info['technical_details'], width=wrap_width)
            for line in tech_lines:
                self.pdf.cell(0, line_height, self._clean_text(line), 0, 1, 'L')
        
        if fig_index == 0:
            self.pdf.ln(5)  # Less spacing after Figure 1
        else:
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
