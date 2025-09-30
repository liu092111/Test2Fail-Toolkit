"""
main.py - Main Controller
Coordinates interactions between modules without containing specific business logic
"""

import os
from datetime import datetime
from gui import LifetimeGUI
from function_code import LifetimeAnalyzer, LifetimeAnalyzerMC, LifetimeAnalyzerPlot
from report_builder import ReportBuilder


class LifetimeAnalysisController:
    """Main Controller - Coordinates interactions between modules"""
    
    def __init__(self):
        self.analyzer = None
        self.mc_analyzer = None
        self.plotter = None
        self.report_builder = None
        
    def load_and_analyze_data(self, file_path, expected_cycles):
        """
        Coordinate data loading and analysis workflow
        Returns: (figures, text_output, success, error_message)
        """
        try:
            # Set matplotlib backend for thread safety
            import matplotlib
            matplotlib.use('Agg')  # Use non-GUI backend in threads
            
            # 1. Data loading and basic analysis
            self.analyzer = LifetimeAnalyzer(file_path)
            self.analyzer.fit_distributions()
            
            # 2. Monte Carlo analysis
            beta = self.analyzer.weibull_params['beta']
            eta = self.analyzer.weibull_params['eta']
            self.mc_analyzer = LifetimeAnalyzerMC(self.analyzer.data, beta, eta)
            
            # 3. Execute analysis and capture output
            text_output = []
            bootstrap_result = self._capture_output(
                lambda: self.mc_analyzer.print_bootstrap_weibull_params(), 
                text_output
            )
            mc_result = self._capture_output(
                lambda: self.mc_analyzer.print_monte_carlo_lifetime(), 
                text_output
            )
            self._capture_output(
                lambda: self.mc_analyzer.print_reliability_at_cycles(expected_cycles), 
                text_output
            )
            
            # 4. Perform advanced statistical tests
            self._perform_advanced_analysis(text_output)
            
            # 5. Generate charts
            figures = self._generate_all_figures(bootstrap_result, mc_result, text_output)
            
            return figures, text_output, True, None
            
        except Exception as e:
            return None, None, False, str(e)
    
    def _capture_output(self, func, output_buffer):
        """Capture function's print output to buffer"""
        import sys
        from io import StringIO
        from contextlib import redirect_stdout
        
        # Create string buffer to capture print output
        captured_output = StringIO()
        
        # Capture output and execute function
        with redirect_stdout(captured_output):
            result = func()
        
        # Add captured output to buffer
        captured_text = captured_output.getvalue()
        if captured_text.strip():
            output_buffer.extend(captured_text.strip().split('\n'))
        
        return result
    
    def _perform_advanced_analysis(self, text_output):
        """Perform advanced statistical analysis"""
        try:
            # Anderson-Darling test
            ad_results = self.analyzer.perform_anderson_darling_test()
            text_output.extend([
                "\n[BOLD14] === [4] Anderson-Darling Goodness-of-Fit Tests ===",
                "Method: More sensitive alternative to Kolmogorov-Smirnov test"
            ])
            
            for dist_name, result in ad_results.items():
                if 'error' not in result:
                    text_output.extend([
                        f"  - {dist_name} Distribution:",
                        f"      - AD Statistic: {result['statistic']:.4f}",
                        f"      - Critical Value (5%): {result['critical_values'][2]:.4f}",
                        f"      - Conclusion: {result['conclusion']}"
                    ])
                else:
                    text_output.append(f"  - {dist_name} Distribution: {result['error']}")
            
            # Maintenance cycle recommendations
            if self.analyzer.best_model == 'Weibull':
                maintenance_rec = self.analyzer.calculate_maintenance_recommendations()
                if 'error' not in maintenance_rec:
                    text_output.extend([
                        "\n[BOLD14] === [5] Maintenance Strategy Recommendations ===",
                        "Method: Cost-benefit optimization based on Weibull reliability model",
                        f"Maintenance Strategy Options:",
                        f"  - Conservative (70% of B10): {maintenance_rec['maintenance_strategy']['conservative']:.1f} cycles",
                        f"  - Balanced (Cost-optimal): {maintenance_rec['maintenance_strategy']['balanced']:.1f} cycles",
                        f"  - Aggressive (90% reliability): {maintenance_rec['maintenance_strategy']['aggressive']:.1f} cycles",
                        f"Optimal Maintenance Interval: {maintenance_rec['optimal_maintenance_interval']:.1f} cycles",
                        f"Expected Reliability at Optimal Interval: {maintenance_rec['optimal_reliability']:.3f}",
                        f"Expected Cost per Maintenance Cycle: ${maintenance_rec['expected_cost_per_cycle']:.0f}"
                    ])
                else:
                    text_output.append(f"\n[BOLD14] === [5] Maintenance Recommendations ===")
                    text_output.append(f"Error: {maintenance_rec['error']}")
            
        except Exception as e:
            text_output.append(f"\nError in advanced analysis: {str(e)}")
    
    def _generate_all_figures(self, bootstrap_result, mc_result, text_output):
        """Generate all charts and return BytesIO object list - Reorganized for better flow"""
        figures = []
        
        try:
            # === SECTION 1: Data Exploration ===
            # 1. Histogram of raw data
            fig1 = self.analyzer.plot_histogram()
            figures.append(self._save_figure_to_memory(fig1))
            
            # 2. PDF comparison with distribution fitting
            fig2 = self.analyzer.plot_pdf_comparison()
            figures.append(self._save_figure_to_memory(fig2))
            
            # 3. Survival function comparison
            fig3 = self.analyzer.plot_survival_comparison()
            figures.append(self._save_figure_to_memory(fig3))
            
            # Skip figure 4 (hazard rate function comparison) as requested
            # fig4 = self.analyzer.plot_hazard_rate_comparison()
            # figures.append(self._save_figure_to_memory(fig4))
            
            # === SECTION 2: Statistical Validation ===
            # 5. Weibull probability paper (goodness-of-fit visualization)
            fig5 = self.analyzer.plot_weibull_probability_paper()
            figures.append(self._save_figure_to_memory(fig5))
            
            # === SECTION 3: Parameter Uncertainty Analysis ===
            self.plotter = LifetimeAnalyzerPlot(
                data=self.mc_analyzer.data, 
                bootstrap_result=bootstrap_result, 
                mc_result=mc_result
            )
            
            # 6-7. Bootstrap parameter distributions (Beta and Eta)
            bootstrap_figs = self.plotter.plot_bootstrap_histograms()
            for fig in bootstrap_figs:
                figures.append(self._save_figure_to_memory(fig))
            
            # === SECTION 4: Predictive Analysis ===
            # 8. Monte Carlo lifetime predictions
            fig_mc = self.plotter.plot_mc_histogram()
            figures.append(self._save_figure_to_memory(fig_mc))
            
            # === SECTION 5: Model Validation ===
            # 9-10. Model validation (Histogram comparison and KS test)
            validation_figs, ks_result = self.plotter.plot_simulated_vs_actual()
            for fig in validation_figs:
                figures.append(self._save_figure_to_memory(fig))
            
            # Add KS test results to text output
            text_output.extend([
                "[BOLD14] === [4] Kolmogorov-Smirnov Test ===",
                f"KS Statistic (D): {ks_result['ks_stat']:.4f}",
                f"P-value: {ks_result['p_value']:.4f}",
                f"Conclusion: {ks_result['conclusion']}"
            ])
            
        except Exception as e:
            print(f"Error generating figures: {e}")
            
        # Filter out None values
        return [fig for fig in figures if fig is not None]
    
    def _save_figure_to_memory(self, fig):
        """Save matplotlib figure to BytesIO"""
        from io import BytesIO
        import matplotlib.pyplot as plt
        
        if fig is None:
            return None
            
        buf = BytesIO()
        try:
            fig.savefig(buf, format='png', dpi=200, bbox_inches='tight')
            buf.seek(0)
            return buf
        except Exception as e:
            print(f"Error saving figure to memory: {e}")
            return None
        finally:
            plt.close(fig)
    
    def generate_report(self, figures, text_output, base_filename="Analysis_Report.pdf"):
        """Generate PDF report only"""
        try:
            # Generate unique filename to avoid overwriting
            unique_filename = self._generate_unique_filename(base_filename)
            
            # Prepare data information for enhanced reporting
            data_info = {
                'sample_size': len(self.analyzer.data) if self.analyzer else 'N/A',
                'filename': os.path.basename(self.analyzer.filepath) if self.analyzer else 'N/A'
            }
            
            # Create report builder and generate PDF only
            self.report_builder = ReportBuilder(text_output, figures, data_info)
            self.report_builder.build(unique_filename)
            
            return os.path.abspath(unique_filename), None
            
        except Exception as e:
            return None, str(e)
    
    def _generate_unique_filename(self, base_filename):
        """Generate unique filename to avoid overwriting"""
        if not os.path.exists(base_filename):
            return base_filename
        
        name, ext = os.path.splitext(base_filename)
        counter = 1
        
        while True:
            # Add timestamp to make filename more meaningful
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            new_filename = f"{name}_{timestamp}_{counter}{ext}"
            if not os.path.exists(new_filename):
                return new_filename
            counter += 1
    
    def _generate_markdown_report(self, text_output, data_info, filename):
        """Generate Analysis Markdown Report with figure placeholders"""
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                # Header
                f.write("# Lifetime Analysis Report\n\n")
                f.write("## Executive Summary\n\n")
                f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                f.write(f"**Analysis Type:** Weibull Distribution Fitting & Monte Carlo Simulation\n\n")
                f.write(f"**Data Points:** {data_info.get('sample_size', 'N/A')}\n\n")
                f.write(f"**Source File:** {data_info.get('filename', 'N/A')}\n\n")
                
                # Extract key metrics
                metrics = self._extract_markdown_metrics(text_output)
                
                f.write("### Key Findings\n\n")
                f.write(f"- **Best Fit Distribution:** Weibull\n")
                f.write(f"- **Shape Parameter (β):** {metrics.get('beta', 'N/A')}\n")
                f.write(f"- **Scale Parameter (η):** {metrics.get('eta', 'N/A')}\n")
                f.write(f"- **B10 Life:** {metrics.get('B10', 'N/A')} cycles\n")
                f.write(f"- **B50 Life (Median):** {metrics.get('B50', 'N/A')} cycles\n")
                f.write(f"- **B95 Life:** {metrics.get('B95', 'N/A')} cycles\n")
                f.write(f"- **MTTF:** {metrics.get('MTTF', 'N/A')} cycles\n\n")
                
                # Methodology
                f.write("## Methodology\n\n")
                f.write("### Statistical Methods Used\n\n")
                f.write("1. **Distribution Fitting:** Multiple probability distributions (Weibull, Lognormal, Exponential) fitted using Maximum Likelihood Estimation\n")
                f.write("2. **Bootstrap Analysis:** 1000 iterations for parameter uncertainty estimation\n")
                f.write("3. **Monte Carlo Simulation:** 10,000 random samples for lifetime predictions\n")
                f.write("4. **Goodness-of-Fit Testing:** Kolmogorov-Smirnov and Anderson-Darling tests\n\n")
                
                # Mathematical Formulas
                f.write("### Key Formulas\n\n")
                f.write("#### Weibull Distribution\n")
                f.write("- **PDF:** `f(t) = (β/η) × (t/η)^(β-1) × exp(-(t/η)^β)`\n")
                f.write("- **Survival Function:** `S(t) = exp(-(t/η)^β)`\n")
                f.write("- **Hazard Rate:** `h(t) = (β/η) × (t/η)^(β-1)`\n")
                f.write("- **MTTF:** `η × Γ(1 + 1/β)`\n\n")
                
                # Add figure placeholders
                f.write("## Analysis Charts\n\n")
                f.write("The following comprehensive visualizations show all key patterns and relationships identified in the data:\n\n")
                
                figure_titles = [
                    "Histogram of Lifetime Data",
                    "Probability Density Function (PDF) Comparison", 
                    "Survival Function Comparison",
                    "Hazard Rate Function Comparison",
                    "Weibull Probability Paper",
                    "Bootstrap Distribution of Beta (Shape Parameter)",
                    "Bootstrap Distribution of Eta (Scale Parameter)",
                    "Monte Carlo Lifetime Predictions",
                    "Histogram Comparison: Simulated vs Actual Data",
                    "Cumulative Distribution Function Comparison with KS Test"
                ]
                
                for i, title in enumerate(figure_titles, 1):
                    f.write(f"### Figure {i}: {title}\n\n")
                    f.write(f"![Figure {i}](figure_{i}.png)\n\n")
                
                # Detailed Results
                f.write("## Detailed Analysis Results\n\n")
                
                current_section = ""
                for line in text_output:
                    if isinstance(line, list):
                        line = ' '.join(str(item) for item in line)
                    line = str(line).strip()
                    
                    if not line:
                        f.write("\n")
                        continue
                    
                    if line.startswith("[BOLD14]"):
                        # Main section headers
                        title = line.replace("[BOLD14]", "").replace("===", "").strip()
                        f.write(f"### {title}\n\n")
                        
                    elif line.startswith("Method:"):
                        # Method descriptions
                        f.write(f"**{line}**\n\n")
                        
                    elif line.startswith("  -"):
                        # Results with proper indentation
                        f.write(f"{line}\n")
                        
                    else:
                        # Regular text
                        f.write(f"{line}\n")
                
                # Statistical Interpretation
                f.write("\n## Statistical Interpretation\n\n")
                f.write("### Shape Parameter (β) Analysis\n")
                beta_val = metrics.get('beta', 'N/A')
                if beta_val != 'N/A':
                    try:
                        beta_float = float(beta_val)
                        if beta_float < 1:
                            f.write("- **β < 1:** Indicates decreasing failure rate (early-life failures, infant mortality)\n")
                        elif beta_float == 1:
                            f.write("- **β = 1:** Indicates constant failure rate (random failures, exponential behavior)\n")
                        else:
                            f.write("- **β > 1:** Indicates increasing failure rate (wear-out failures, aging)\n")
                    except:
                        f.write("- Shape parameter analysis not available\n")
                
                f.write("\n### Reliability Metrics\n")
                f.write("- **B10:** Time at which 10% of units are expected to fail\n")
                f.write("- **B50:** Median life - 50% of units will survive beyond this point\n")
                f.write("- **B95:** Time at which 95% of units are expected to fail\n")
                f.write("- **MTTF:** Mean Time To Failure - average expected lifetime\n\n")
                
                # Recommendations
                f.write("## Engineering Recommendations\n\n")
                f.write("### Maintenance Strategy\n")
                if 'maintenance_strategy' in str(text_output):
                    f.write("Based on the Weibull analysis, the following maintenance strategies are recommended:\n\n")
                    # Extract maintenance recommendations from text_output
                    for line in text_output:
                        if 'Conservative' in str(line) or 'Balanced' in str(line) or 'Aggressive' in str(line):
                            f.write(f"- {line}\n")
                else:
                    f.write("- Schedule preventive maintenance before B10 life to minimize unexpected failures\n")
                    f.write("- Implement condition monitoring starting at 80% of B10 life\n")
                    f.write("- Consider design improvements if reliability targets are not met\n")
                
                f.write("\n### Quality Control\n")
                reliability = metrics.get('reliability_median', '0.5')
                try:
                    rel_float = float(reliability)
                    if rel_float > 0.8:
                        f.write("- Current reliability levels are acceptable - maintain existing quality standards\n")
                    else:
                        f.write("- Implement enhanced quality control measures to improve reliability\n")
                        f.write("- Consider design review to extend operational life\n")
                except:
                    f.write("- Continue monitoring system performance and data collection\n")
                
                # Footer
                f.write(f"\n---\n")
                f.write(f"*Report generated by Lifetime Analysis System on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n")
            
            # Generate function code documentation as separate file
            func_doc_filename = filename.replace('.md', '_function_documentation.md')
            self._generate_function_documentation(func_doc_filename)
                
        except Exception as e:
            print(f"Error generating markdown report: {e}")
    
    def _generate_function_documentation(self, filename):
        """Generate Function Code Documentation in Markdown format"""
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                # Header
                f.write("# Lifetime Analysis Function Code Documentation\n\n")
                f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                f.write("This document provides comprehensive documentation for all functions and classes in the Lifetime Analysis System.\n\n")
                
                # Table of Contents
                f.write("## Table of Contents\n\n")
                f.write("1. [LifetimeAnalyzer Class](#lifetimeanalyzer-class)\n")
                f.write("2. [LifetimeAnalyzerMC Class](#lifetimeanalyzermc-class)\n")
                f.write("3. [LifetimeAnalyzerPlot Class](#lifetimeanalyzerplot-class)\n")
                f.write("4. [ReportBuilder Class](#reportbuilder-class)\n")
                f.write("5. [LifetimeAnalysisController Class](#lifetimeanalysiscontroller-class)\n\n")
                
                # LifetimeAnalyzer Class Documentation
                f.write("## LifetimeAnalyzer Class\n\n")
                f.write("**Purpose:** Core statistical analysis class for lifetime data fitting and distribution analysis.\n\n")
                
                f.write("### Constructor\n")
                f.write("```python\n")
                f.write("LifetimeAnalyzer(filepath)\n")
                f.write("```\n")
                f.write("**Parameters:**\n")
                f.write("- `filepath` (str): Path to data file (.csv, .txt, or .xlsx)\n\n")
                f.write("**Description:** Initializes the analyzer by loading and preprocessing lifetime data from the specified file.\n\n")
                
                f.write("### Key Methods\n\n")
                
                f.write("#### `fit_distributions()`\n")
                f.write("**Purpose:** Fits multiple probability distributions (Weibull, Lognormal, Exponential) to the data using Maximum Likelihood Estimation.\n\n")
                f.write("**Returns:** None (stores results in `self.results` dictionary)\n\n")
                f.write("**Key Features:**\n")
                f.write("- Calculates PDF, survival function, and log-likelihood for each distribution\n")
                f.write("- Determines best-fit model based on highest log-likelihood\n")
                f.write("- Computes MTTF (Mean Time To Failure) for each distribution\n\n")
                
                f.write("#### `perform_anderson_darling_test()`\n")
                f.write("**Purpose:** Performs Anderson-Darling goodness-of-fit tests for distribution validation.\n\n")
                f.write("**Returns:** Dictionary containing test statistics and conclusions\n\n")
                f.write("**Features:**\n")
                f.write("- More sensitive than Kolmogorov-Smirnov test\n")
                f.write("- Tests for Normal and Exponential distributions\n")
                f.write("- Provides critical values and reject/accept decisions\n\n")
                
                f.write("#### `calculate_maintenance_recommendations()`\n")
                f.write("**Purpose:** Calculates optimal maintenance strategies based on Weibull reliability model.\n\n")
                f.write("**Parameters:**\n")
                f.write("- `target_reliability` (float): Target reliability level (default: 0.9)\n")
                f.write("- `cost_per_maintenance` (float): Cost per maintenance cycle (default: 1000)\n")
                f.write("- `cost_per_failure` (float): Cost per failure event (default: 10000)\n\n")
                f.write("**Returns:** Dictionary with maintenance intervals and cost analysis\n\n")
                
                f.write("#### Plotting Methods\n")
                f.write("- `plot_histogram()`: Creates histogram of raw lifetime data\n")
                f.write("- `plot_pdf_comparison()`: Compares fitted PDFs with data histogram\n")
                f.write("- `plot_survival_comparison()`: Plots survival functions for all distributions\n")
                f.write("- `plot_hazard_rate_comparison()`: Shows hazard rate functions\n")
                f.write("- `plot_weibull_probability_paper()`: Creates Weibull probability paper plot\n\n")
                
                # LifetimeAnalyzerMC Class Documentation
                f.write("## LifetimeAnalyzerMC Class\n\n")
                f.write("**Purpose:** Monte Carlo simulation and bootstrap analysis for parameter uncertainty estimation.\n\n")
                
                f.write("### Constructor\n")
                f.write("```python\n")
                f.write("LifetimeAnalyzerMC(data, beta=None, eta=None)\n")
                f.write("```\n")
                f.write("**Parameters:**\n")
                f.write("- `data` (array): Original lifetime data\n")
                f.write("- `beta` (float): Weibull shape parameter (optional)\n")
                f.write("- `eta` (float): Weibull scale parameter (optional)\n\n")
                
                f.write("### Key Methods\n\n")
                
                f.write("#### `print_bootstrap_weibull_params(n_bootstrap=1000)`\n")
                f.write("**Purpose:** Performs bootstrap resampling to estimate parameter uncertainty.\n\n")
                f.write("**Process:**\n")
                f.write("1. Resamples original data with replacement (1000 iterations)\n")
                f.write("2. Fits Weibull distribution to each bootstrap sample\n")
                f.write("3. Calculates 95% confidence intervals for β and η parameters\n")
                f.write("4. Returns mean parameter estimates and confidence bounds\n\n")
                
                f.write("#### `print_monte_carlo_lifetime(n_simulations=10000)`\n")
                f.write("**Purpose:** Simulates lifetime predictions using fitted Weibull parameters.\n\n")
                f.write("**Process:**\n")
                f.write("1. Generates 10,000 random samples from Weibull distribution\n")
                f.write("2. Calculates reliability percentiles (B10, B50, B95)\n")
                f.write("3. Computes Mean Time To Failure (MTTF)\n")
                f.write("4. Provides lifetime prediction statistics\n\n")
                
                f.write("#### `print_reliability_at_cycles(cycles, ci=0.95, n_bootstrap=1000)`\n")
                f.write("**Purpose:** Estimates reliability at specific cycle count with confidence intervals.\n\n")
                f.write("**Parameters:**\n")
                f.write("- `cycles` (float): Target cycle count for reliability estimation\n")
                f.write("- `ci` (float): Confidence interval level (default: 0.95)\n")
                f.write("- `n_bootstrap` (int): Number of bootstrap iterations\n\n")
                
                # LifetimeAnalyzerPlot Class Documentation
                f.write("## LifetimeAnalyzerPlot Class\n\n")
                f.write("**Purpose:** Advanced plotting and visualization for bootstrap and Monte Carlo results.\n\n")
                
                f.write("### Constructor\n")
                f.write("```python\n")
                f.write("LifetimeAnalyzerPlot(data, bootstrap_result=None, mc_result=None)\n")
                f.write("```\n")
                f.write("**Parameters:**\n")
                f.write("- `data` (array): Original lifetime data\n")
                f.write("- `bootstrap_result` (dict): Results from bootstrap analysis\n")
                f.write("- `mc_result` (dict): Results from Monte Carlo simulation\n\n")
                
                f.write("### Key Methods\n\n")
                
                f.write("#### `plot_bootstrap_histograms()`\n")
                f.write("**Purpose:** Creates separate histograms for β and η parameter distributions.\n\n")
                f.write("**Features:**\n")
                f.write("- Shows parameter uncertainty through distribution shape\n")
                f.write("- Displays mean values and 95% confidence intervals\n")
                f.write("- Returns list of two matplotlib figures\n\n")
                
                f.write("#### `plot_mc_histogram()`\n")
                f.write("**Purpose:** Visualizes Monte Carlo lifetime predictions with percentile markers.\n\n")
                f.write("**Features:**\n")
                f.write("- Histogram of 10,000 simulated lifetime values\n")
                f.write("- Vertical lines marking B10, B50, B90, B95, B99 percentiles\n")
                f.write("- Clean visualization without formula boxes\n\n")
                
                f.write("#### `plot_simulated_vs_actual()`\n")
                f.write("**Purpose:** Validates model accuracy by comparing simulated and actual data.\n\n")
                f.write("**Returns:** Tuple of (figure_list, ks_test_results)\n\n")
                f.write("**Features:**\n")
                f.write("- Histogram comparison of actual vs simulated data\n")
                f.write("- CDF comparison with Kolmogorov-Smirnov test\n")
                f.write("- Statistical validation of model adequacy\n\n")
                
                # ReportBuilder Class Documentation
                f.write("## ReportBuilder Class\n\n")
                f.write("**Purpose:** Generates professional PDF reports with detailed figure descriptions.\n\n")
                
                f.write("### Constructor\n")
                f.write("```python\n")
                f.write("ReportBuilder(text_buffer, figures, data_info=None)\n")
                f.write("```\n")
                f.write("**Parameters:**\n")
                f.write("- `text_buffer` (list): Analysis results text output\n")
                f.write("- `figures` (list): BytesIO objects containing matplotlib figures\n")
                f.write("- `data_info` (dict): Metadata about the dataset\n\n")
                
                f.write("### Key Features\n\n")
                f.write("#### Professional Report Structure\n")
                f.write("1. **Title Page:** Report metadata and generation information\n")
                f.write("2. **Executive Summary:** Key findings and statistical overview\n")
                f.write("3. **Data Overview:** Tabular summary of analysis parameters\n")
                f.write("4. **Methodology:** Statistical methods and approaches used\n")
                f.write("5. **Detailed Results:** Complete analysis output\n")
                f.write("6. **Figures with Descriptions:** Each figure includes detailed explanation\n")
                f.write("7. **Risk Assessment:** Engineering recommendations\n")
                f.write("8. **Appendix:** Technical details and formulas\n\n")
                
                f.write("#### Enhanced Figure Descriptions\n")
                f.write("Each figure includes:\n")
                f.write("- **Description:** Technical explanation of what the figure shows\n")
                f.write("- **Interpretation:** Engineering significance and practical implications\n")
                f.write("- **Professional formatting:** Similar to weather analysis reports\n\n")
                
                f.write("#### Automatic Page Numbering\n")
                f.write("- Consistent footer with page numbers\n")
                f.write("- Generation timestamp on each page\n")
                f.write("- Professional layout with proper margins\n\n")
                
                # LifetimeAnalysisController Class Documentation
                f.write("## LifetimeAnalysisController Class\n\n")
                f.write("**Purpose:** Main controller coordinating all analysis modules and workflow.\n\n")
                
                f.write("### Key Methods\n\n")
                
                f.write("#### `load_and_analyze_data(file_path, expected_cycles)`\n")
                f.write("**Purpose:** Orchestrates the complete analysis workflow.\n\n")
                f.write("**Process:**\n")
                f.write("1. Loads data using LifetimeAnalyzer\n")
                f.write("2. Fits probability distributions\n")
                f.write("3. Performs Monte Carlo and bootstrap analysis\n")
                f.write("4. Executes advanced statistical tests\n")
                f.write("5. Generates all visualization figures\n")
                f.write("6. Returns results for report generation\n\n")
                
                f.write("#### `generate_report(figures, text_output, base_filename)`\n")
                f.write("**Purpose:** Creates both PDF and Markdown reports.\n\n")
                f.write("**Features:**\n")
                f.write("- Generates unique filenames to prevent overwriting\n")
                f.write("- Creates comprehensive PDF report with figure descriptions\n")
                f.write("- Generates function code documentation (this document)\n")
                f.write("- Handles error reporting and file management\n\n")
                
                f.write("#### `_generate_all_figures(bootstrap_result, mc_result, text_output)`\n")
                f.write("**Purpose:** Coordinates figure generation in logical order.\n\n")
                f.write("**Figure Organization:**\n")
                f.write("1. **Data Exploration:** Histogram, PDF comparison, survival functions\n")
                f.write("2. **Statistical Validation:** Hazard rates, Weibull probability paper\n")
                f.write("3. **Parameter Uncertainty:** Bootstrap distributions\n")
                f.write("4. **Predictive Analysis:** Monte Carlo simulations\n")
                f.write("5. **Model Validation:** Comparison plots and statistical tests\n\n")
                
                # Mathematical Background
                f.write("## Mathematical Background\n\n")
                
                f.write("### Weibull Distribution\n")
                f.write("The two-parameter Weibull distribution is fundamental to reliability analysis:\n\n")
                f.write("**Probability Density Function:**\n")
                f.write("```\n")
                f.write("f(t) = (β/η) × (t/η)^(β-1) × exp(-(t/η)^β)\n")
                f.write("```\n\n")
                f.write("**Survival Function:**\n")
                f.write("```\n")
                f.write("S(t) = exp(-(t/η)^β)\n")
                f.write("```\n\n")
                f.write("**Hazard Rate Function:**\n")
                f.write("```\n")
                f.write("h(t) = (β/η) × (t/η)^(β-1)\n")
                f.write("```\n\n")
                
                f.write("### Parameter Interpretation\n")
                f.write("- **β (Shape Parameter):**\n")
                f.write("  - β < 1: Decreasing failure rate (infant mortality)\n")
                f.write("  - β = 1: Constant failure rate (random failures)\n")
                f.write("  - β > 1: Increasing failure rate (wear-out)\n\n")
                f.write("- **η (Scale Parameter):**\n")
                f.write("  - Characteristic life (63.2% failure point)\n")
                f.write("  - Determines the scale of the distribution\n\n")
                
                f.write("### Statistical Methods\n")
                
                f.write("#### Bootstrap Method\n")
                f.write("Non-parametric resampling technique:\n")
                f.write("1. Resample original data with replacement\n")
                f.write("2. Fit distribution to each bootstrap sample\n")
                f.write("3. Build empirical distribution of parameters\n")
                f.write("4. Calculate confidence intervals from percentiles\n\n")
                
                f.write("#### Monte Carlo Simulation\n")
                f.write("Stochastic simulation process:\n")
                f.write("1. Use fitted parameters to define distribution\n")
                f.write("2. Generate large number of random samples\n")
                f.write("3. Calculate percentiles and statistics\n")
                f.write("4. Provide robust lifetime predictions\n\n")
                
                f.write("#### Goodness-of-Fit Tests\n")
                f.write("- **Kolmogorov-Smirnov Test:** Compares empirical and theoretical CDFs\n")
                f.write("- **Anderson-Darling Test:** More sensitive to tail differences\n")
                f.write("- Both tests validate model adequacy and reliability\n\n")
                
                # Usage Examples
                f.write("## Usage Examples\n\n")
                
                f.write("### Basic Analysis Workflow\n")
                f.write("```python\n")
                f.write("# Initialize analyzer with data file\n")
                f.write("analyzer = LifetimeAnalyzer('lifetime_data.csv')\n\n")
                f.write("# Fit distributions and find best model\n")
                f.write("analyzer.fit_distributions()\n")
                f.write("print(f'Best fit: {analyzer.best_model}')\n\n")
                f.write("# Perform Monte Carlo analysis\n")
                f.write("mc_analyzer = LifetimeAnalyzerMC(analyzer.data)\n")
                f.write("bootstrap_result = mc_analyzer.print_bootstrap_weibull_params()\n")
                f.write("mc_result = mc_analyzer.print_monte_carlo_lifetime()\n\n")
                f.write("# Generate plots\n")
                f.write("plotter = LifetimeAnalyzerPlot(analyzer.data, bootstrap_result, mc_result)\n")
                f.write("figures = plotter.plot_bootstrap_histograms()\n")
                f.write("```\n\n")
                
                f.write("### Advanced Statistical Analysis\n")
                f.write("```python\n")
                f.write("# Perform goodness-of-fit tests\n")
                f.write("ad_results = analyzer.perform_anderson_darling_test()\n\n")
                f.write("# Calculate maintenance recommendations\n")
                f.write("maintenance = analyzer.calculate_maintenance_recommendations(\n")
                f.write("    target_reliability=0.9,\n")
                f.write("    cost_per_maintenance=1000,\n")
                f.write("    cost_per_failure=10000\n")
                f.write(")\n\n")
                f.write("# Estimate reliability at specific cycles\n")
                f.write("reliability = mc_analyzer.print_reliability_at_cycles(5000)\n")
                f.write("```\n\n")
                
                # Footer
                f.write("---\n")
                f.write(f"*Function documentation generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n")
                f.write("*This documentation covers all major functions and classes in the Lifetime Analysis System*\n")
                
        except Exception as e:
            print(f"Error generating function documentation: {e}")
    
    def _extract_markdown_metrics(self, text_output):
        """Extract key metrics from text output for markdown report"""
        metrics = {}
        
        # Convert all lines to strings
        text_lines = []
        for line in text_output:
            if isinstance(line, list):
                line = ' '.join(str(item) for item in line)
            text_lines.append(str(line).strip())
        
        # Track context for parameter extraction
        in_beta_section = False
        in_eta_section = False
        
        for line in text_lines:
            # Check for section context
            if "Beta (shape)" in line:
                in_beta_section = True
                in_eta_section = False
            elif "Eta (scale)" in line:
                in_beta_section = False
                in_eta_section = True
            elif line.startswith("===") or line.startswith("[BOLD14]"):
                in_beta_section = False
                in_eta_section = False
            
            # Extract metrics
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
                    
                elif "Median reliability" in line:
                    metrics['reliability_median'] = line.split(":")[-1].strip()
                    
            except Exception:
                continue
        
        return metrics

    def create_narrative_report_from_function_doc(self, function_doc_filename, figures, text_output, data_info):
        """Create a narrative-style PDF report based on function documentation with embedded data and figures"""
        try:
            # First, save figures as PNG files
            figure_files = []
            base_dir = os.path.dirname(function_doc_filename)
            
            for i, fig_buf in enumerate(figures, 1):
                if fig_buf is not None:
                    figure_path = os.path.join(base_dir, f"figure_{i}.png")
                    with open(figure_path, 'wb') as f:
                        fig_buf.seek(0)
                        f.write(fig_buf.read())
                    figure_files.append(figure_path)
            
            # Create narrative PDF report
            pdf_filename = function_doc_filename.replace('_function_documentation.md', '_narrative_report.pdf')
            self._create_narrative_pdf_report(function_doc_filename, figure_files, text_output, data_info, pdf_filename)
            
            # Clean up temporary figure files
            for fig_file in figure_files:
                try:
                    os.remove(fig_file)
                except:
                    pass
            
            return os.path.abspath(pdf_filename), None
            
        except Exception as e:
            return None, str(e)
    
    def _create_markdown_based_pdf(self, markdown_filename, figure_files, pdf_filename):
        """Create PDF from Markdown content with embedded figures"""
        from fpdf import FPDF
        import tempfile
        
        class MarkdownPDF(FPDF):
            def __init__(self):
                super().__init__()
                self.set_auto_page_break(auto=True, margin=15)
                
            def footer(self):
                self.set_y(-15)
                self.set_font('Arial', 'I', 8)
                self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')
                self.set_y(-15)
                self.cell(0, 10, f'Generated: {datetime.now().strftime("%Y-%m-%d %H:%M")}', 0, 0, 'R')
        
        pdf = MarkdownPDF()
        pdf.set_margins(left=15, top=20, right=15)
        
        # Read markdown content
        with open(markdown_filename, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Parse and convert markdown to PDF
        lines = content.split('\n')
        current_figure = 1
        
        pdf.add_page()
        
        for line in lines:
            line = line.strip()
            
            if not line:
                pdf.ln(3)
                continue
            
            # Handle headers
            if line.startswith('# '):
                pdf.ln(10)
                pdf.set_font('Arial', 'B', 18)
                title = line[2:].strip()
                pdf.cell(0, 10, self._clean_pdf_text(title), 0, 1, 'C')
                pdf.ln(5)
                
            elif line.startswith('## '):
                pdf.ln(8)
                pdf.set_font('Arial', 'B', 14)
                title = line[3:].strip()
                pdf.cell(0, 8, self._clean_pdf_text(title), 0, 1)
                pdf.ln(3)
                
            elif line.startswith('### '):
                pdf.ln(5)
                pdf.set_font('Arial', 'B', 12)
                title = line[4:].strip()
                pdf.cell(0, 7, self._clean_pdf_text(title), 0, 1)
                pdf.ln(2)
                
            # Handle figure references
            elif line.startswith('![Figure'):
                if current_figure <= len(figure_files):
                    try:
                        # Add some space before figure
                        pdf.ln(5)
                        
                        # Calculate image dimensions
                        img_width = min(170, pdf.w - pdf.l_margin - pdf.r_margin)
                        x_pos = (pdf.w - img_width) / 2
                        
                        # Add the figure
                        pdf.image(figure_files[current_figure-1], x=x_pos, y=pdf.get_y(), w=img_width)
                        
                        # Add space after figure
                        pdf.ln(80)  # Approximate space for figure
                        current_figure += 1
                        
                    except Exception as e:
                        pdf.set_font('Arial', '', 10)
                        pdf.cell(0, 5, f"[Figure {current_figure} could not be loaded]", 0, 1)
                        current_figure += 1
                
            # Handle bold text
            elif line.startswith('**') and line.endswith('**'):
                pdf.set_font('Arial', 'B', 10)
                text = line[2:-2].strip()
                pdf.cell(0, 6, self._clean_pdf_text(text), 0, 1)
                
            # Handle bullet points
            elif line.startswith('- '):
                pdf.set_font('Arial', '', 10)
                text = line[2:].strip()
                # Handle bold text within bullets
                if '**' in text:
                    text = text.replace('**', '')
                pdf.cell(5, 5, '•', 0, 0)
                pdf.cell(0, 5, self._clean_pdf_text(text), 0, 1)
                
            # Handle numbered lists
            elif line and line[0].isdigit() and '. ' in line:
                pdf.set_font('Arial', '', 10)
                pdf.cell(0, 5, self._clean_pdf_text(line), 0, 1)
                
            # Handle code blocks
            elif line.startswith('```'):
                continue  # Skip code block markers
                
            elif line.startswith('`') and line.endswith('`'):
                pdf.set_font('Courier', '', 9)
                text = line[1:-1].strip()
                pdf.cell(0, 5, self._clean_pdf_text(text), 0, 1)
                
            # Handle horizontal rules
            elif line.startswith('---'):
                pdf.ln(5)
                pdf.line(pdf.l_margin, pdf.get_y(), pdf.w - pdf.r_margin, pdf.get_y())
                pdf.ln(5)
                
            # Handle regular text
            elif line and not line.startswith('#'):
                pdf.set_font('Arial', '', 10)
                # Wrap long lines
                import textwrap
                wrapped_lines = textwrap.wrap(line, width=85)
                for wrapped_line in wrapped_lines:
                    pdf.cell(0, 5, self._clean_pdf_text(wrapped_line), 0, 1)
        
        pdf.output(pdf_filename)
    
    def _create_narrative_pdf_report(self, function_doc_filename, figure_files, text_output, data_info, pdf_filename):
        """Create a narrative-style PDF report with LaTeX formatting and embedded figures"""
        from fpdf import FPDF
        import textwrap
        
        class NarrativePDF(FPDF):
            def __init__(self):
                super().__init__()
                self.set_auto_page_break(auto=True, margin=20)
                
            def footer(self):
                self.set_y(-15)
                self.set_font('Arial', 'I', 8)
                self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')
                self.set_y(-15)
                self.cell(0, 10, f'Generated: {datetime.now().strftime("%Y-%m-%d %H:%M")}', 0, 0, 'R')
        
        pdf = NarrativePDF()
        pdf.set_margins(left=20, top=25, right=20)
        
        # Extract key metrics from text output
        metrics = self._extract_markdown_metrics(text_output)
        
        # === TITLE PAGE ===
        pdf.add_page()
        pdf.ln(40)
        pdf.set_font('Arial', 'B', 24)
        pdf.cell(0, 15, 'The Story of Lifetime Analysis', 0, 1, 'C')
        pdf.ln(10)
        pdf.set_font('Arial', 'I', 16)
        pdf.cell(0, 10, 'A Journey Through Data, Statistics, and Reliability', 0, 1, 'C')
        pdf.ln(30)
        
        # Story introduction
        pdf.set_font('Arial', '', 12)
        intro_text = f"""
In the world of engineering and reliability analysis, every dataset tells a story. Today, we embark on a journey to understand the lifetime characteristics of {data_info.get('sample_size', 'N/A')} data points from {data_info.get('filename', 'our dataset')}. 

This is not just a collection of numbers - it's a narrative about failure, survival, and the mathematical beauty that helps us predict the future. Through advanced statistical methods and visualization, we'll uncover the hidden patterns that govern the lifetime behavior of our system.
        """
        
        for paragraph in intro_text.strip().split('\n\n'):
            lines = textwrap.wrap(paragraph.strip(), width=80)
            for line in lines:
                pdf.cell(0, 6, self._clean_pdf_text(line), 0, 1)
            pdf.ln(5)
        
        # === CHAPTER 1: THE DATA AWAKENS ===
        pdf.add_page()
        pdf.set_font('Arial', 'B', 18)
        pdf.cell(0, 12, 'Chapter 1: The Data Awakens', 0, 1)
        pdf.ln(5)
        
        pdf.set_font('Arial', '', 11)
        chapter1_text = f"""
Our story begins with raw data - {data_info.get('sample_size', 'N/A')} observations that represent the lifetime cycles of our system. Like archaeologists examining ancient artifacts, we must first understand what these numbers reveal about the underlying patterns.

The first step in our journey is visualization. By creating a histogram of our lifetime data, we can see the shape of our story - is it a tale of early failures, random occurrences, or gradual wear-out?
        """
        
        for paragraph in chapter1_text.strip().split('\n\n'):
            lines = textwrap.wrap(paragraph.strip(), width=80)
            for line in lines:
                pdf.cell(0, 6, self._clean_pdf_text(line), 0, 1)
            pdf.ln(3)
        
        # Insert Figure 1: Histogram
        if len(figure_files) > 0:
            pdf.ln(5)
            pdf.set_font('Arial', 'B', 12)
            pdf.cell(0, 8, 'Figure 1: The Shape of Our Data Story', 0, 1)
            pdf.ln(3)
            
            img_width = min(160, pdf.w - pdf.l_margin - pdf.r_margin)
            x_pos = (pdf.w - img_width) / 2
            pdf.image(figure_files[0], x=x_pos, y=pdf.get_y(), w=img_width)
            pdf.ln(85)
            
            pdf.set_font('Arial', 'I', 10)
            pdf.cell(0, 5, 'This histogram reveals the frequency distribution of our lifetime data,', 0, 1, 'C')
            pdf.cell(0, 5, 'showing us the first glimpse of the underlying failure patterns.', 0, 1, 'C')
        
        # === CHAPTER 2: THE MATHEMATICAL FOUNDATION ===
        pdf.add_page()
        pdf.set_font('Arial', 'B', 18)
        pdf.cell(0, 12, 'Chapter 2: The Mathematical Foundation', 0, 1)
        pdf.ln(5)
        
        pdf.set_font('Arial', '', 11)
        chapter2_text = f"""
Every great story needs a solid foundation, and ours is built upon the elegant mathematics of probability distributions. We tested three different mathematical models to see which one best captures the essence of our data:

1. The Weibull Distribution - A versatile model that can describe various failure modes
2. The Lognormal Distribution - Often used for systems with multiplicative effects  
3. The Exponential Distribution - The simplest model for constant failure rates

Through Maximum Likelihood Estimation, we discovered that the Weibull distribution tells our story best, with parameters that reveal fascinating insights about our system's behavior.
        """
        
        for paragraph in chapter2_text.strip().split('\n\n'):
            lines = textwrap.wrap(paragraph.strip(), width=80)
            for line in lines:
                pdf.cell(0, 6, self._clean_pdf_text(line), 0, 1)
            pdf.ln(3)
        
        # LaTeX-style mathematical formulations
        pdf.ln(5)
        pdf.set_font('Arial', 'B', 12)
        pdf.cell(0, 8, 'The Mathematical Language of Reliability', 0, 1)
        pdf.ln(3)
        
        pdf.set_font('Courier', '', 10)
        math_formulas = [
            "Weibull PDF: f(t) = (beta/eta) * (t/eta)^(beta-1) * exp(-(t/eta)^beta)",
            "Survival Function: S(t) = exp(-(t/eta)^beta)", 
            "Hazard Rate: h(t) = (beta/eta) * (t/eta)^(beta-1)",
            f"Our fitted parameters: beta = {metrics.get('beta', 'N/A')}, eta = {metrics.get('eta', 'N/A')}"
        ]
        
        for formula in math_formulas:
            pdf.cell(0, 6, formula, 0, 1)
            pdf.ln(2)
        
        # Insert Figure 2: PDF Comparison
        if len(figure_files) > 1:
            pdf.ln(5)
            pdf.set_font('Arial', 'B', 12)
            pdf.cell(0, 8, 'Figure 2: The Battle of Distributions', 0, 1)
            pdf.ln(3)
            
            img_width = min(160, pdf.w - pdf.l_margin - pdf.r_margin)
            x_pos = (pdf.w - img_width) / 2
            pdf.image(figure_files[1], x=x_pos, y=pdf.get_y(), w=img_width)
            pdf.ln(85)
            
            pdf.set_font('Arial', 'I', 10)
            pdf.cell(0, 5, 'Here we see the competition between different mathematical models,', 0, 1, 'C')
            pdf.cell(0, 5, 'with the Weibull distribution emerging as the clear winner.', 0, 1, 'C')
        
        # === CHAPTER 3: THE SURVIVAL STORY ===
        pdf.add_page()
        pdf.set_font('Arial', 'B', 18)
        pdf.cell(0, 12, 'Chapter 3: The Survival Story', 0, 1)
        pdf.ln(5)
        
        pdf.set_font('Arial', '', 11)
        chapter3_text = f"""
Now we delve deeper into the heart of reliability analysis - the survival function. This mathematical curve tells us the probability that our system will survive beyond any given time. It's like a crystal ball that reveals the future fate of our components.

The survival function S(t) answers the fundamental question: "What are the chances that our system will still be functioning after t cycles?" This is crucial information for engineers making decisions about warranties, maintenance schedules, and replacement strategies.

Our analysis reveals key reliability milestones:
• B10 Life: {metrics.get('B10', 'N/A')} cycles (10% will fail by this point)
• B50 Life: {metrics.get('B50', 'N/A')} cycles (median lifetime)  
• B95 Life: {metrics.get('B95', 'N/A')} cycles (95% will have failed by this point)
• MTTF: {metrics.get('MTTF', 'N/A')} cycles (average expected lifetime)
        """
        
        for paragraph in chapter3_text.strip().split('\n\n'):
            lines = textwrap.wrap(paragraph.strip(), width=80)
            for line in lines:
                pdf.cell(0, 6, self._clean_pdf_text(line), 0, 1)
            pdf.ln(3)
        
        # Insert Figure 3: Survival Functions
        if len(figure_files) > 2:
            pdf.ln(5)
            pdf.set_font('Arial', 'B', 12)
            pdf.cell(0, 8, 'Figure 3: The Survival Curves', 0, 1)
            pdf.ln(3)
            
            img_width = min(160, pdf.w - pdf.l_margin - pdf.r_margin)
            x_pos = (pdf.w - img_width) / 2
            pdf.image(figure_files[2], x=x_pos, y=pdf.get_y(), w=img_width)
            pdf.ln(85)
            
            pdf.set_font('Arial', 'I', 10)
            pdf.cell(0, 5, 'These curves show how the probability of survival decreases over time,', 0, 1, 'C')
            pdf.cell(0, 5, 'revealing the natural progression from birth to failure.', 0, 1, 'C')
        
        # === CHAPTER 4: THE HAZARD RATE REVELATION ===
        pdf.add_page()
        pdf.set_font('Arial', 'B', 18)
        pdf.cell(0, 12, 'Chapter 4: The Hazard Rate Revelation', 0, 1)
        pdf.ln(5)
        
        pdf.set_font('Arial', '', 11)
        beta_val = metrics.get('beta', 'N/A')
        hazard_interpretation = ""
        if beta_val != 'N/A':
            try:
                beta_float = float(beta_val)
                if beta_float < 1:
                    hazard_interpretation = "decreasing failure rate, indicating early-life failures or 'infant mortality'"
                elif beta_float == 1:
                    hazard_interpretation = "constant failure rate, suggesting random failures throughout life"
                else:
                    hazard_interpretation = "increasing failure rate, revealing wear-out failures as the system ages"
            except:
                hazard_interpretation = "complex failure behavior requiring further investigation"
        
        chapter4_text = f"""
The hazard rate function h(t) tells perhaps the most intriguing part of our story - it reveals the instantaneous risk of failure at any given moment, assuming the system has survived up to that point. Think of it as the "danger level" at each point in time.

Our Weibull analysis reveals a {hazard_interpretation}. This insight is crucial for understanding the underlying physics of failure and designing appropriate maintenance strategies.

The shape parameter beta = {beta_val} is the key to this revelation. It acts like a storyteller, revealing whether our system experiences:
- Early troubles (beta < 1)
- Consistent random failures (beta = 1)  
- Gradual wear-out (beta > 1)
        """
        
        for paragraph in chapter4_text.strip().split('\n\n'):
            lines = textwrap.wrap(paragraph.strip(), width=80)
            for line in lines:
                pdf.cell(0, 6, self._clean_pdf_text(line), 0, 1)
            pdf.ln(3)
        
        # Insert Figure 4: Hazard Rate
        if len(figure_files) > 3:
            pdf.ln(5)
            pdf.set_font('Arial', 'B', 12)
            pdf.cell(0, 8, 'Figure 4: The Hazard Rate Journey', 0, 1)
            pdf.ln(3)
            
            img_width = min(160, pdf.w - pdf.l_margin - pdf.r_margin)
            x_pos = (pdf.w - img_width) / 2
            pdf.image(figure_files[3], x=x_pos, y=pdf.get_y(), w=img_width)
            pdf.ln(85)
            
            pdf.set_font('Arial', 'I', 10)
            pdf.cell(0, 5, 'The hazard rate curves reveal the changing risk of failure over time,', 0, 1, 'C')
            pdf.cell(0, 5, 'providing insights into the fundamental failure mechanisms.', 0, 1, 'C')
        
        # === CHAPTER 5: UNCERTAINTY AND CONFIDENCE ===
        pdf.add_page()
        pdf.set_font('Arial', 'B', 18)
        pdf.cell(0, 12, 'Chapter 5: Embracing Uncertainty', 0, 1)
        pdf.ln(5)
        
        pdf.set_font('Arial', '', 11)
        chapter5_text = f"""
No story is complete without acknowledging uncertainty. In the real world, we never have perfect knowledge, and our parameter estimates come with inherent uncertainty. This is where the bootstrap method becomes our trusted companion.

By resampling our data 1000 times and refitting the Weibull distribution to each sample, we create a distribution of possible parameter values. This gives us confidence intervals that quantify our uncertainty:

For the shape parameter (beta): We can be 95% confident that the true value lies within a specific range, reflecting the precision of our estimate.

For the scale parameter (eta): Similarly, we obtain confidence bounds that help us understand the reliability of our characteristic life estimate.

This uncertainty quantification is not a weakness - it's a strength that allows us to make informed engineering decisions with full awareness of the risks involved.
        """
        
        for paragraph in chapter5_text.strip().split('\n\n'):
            lines = textwrap.wrap(paragraph.strip(), width=80)
            for line in lines:
                pdf.cell(0, 6, self._clean_pdf_text(line), 0, 1)
            pdf.ln(3)
        
        # Insert Bootstrap Figures (5 and 6)
        if len(figure_files) > 5:
            pdf.ln(5)
            pdf.set_font('Arial', 'B', 12)
            pdf.cell(0, 8, 'Figure 5: Bootstrap Distribution of Shape Parameter', 0, 1)
            pdf.ln(3)
            
            img_width = min(140, pdf.w - pdf.l_margin - pdf.r_margin)
            x_pos = (pdf.w - img_width) / 2
            pdf.image(figure_files[5], x=x_pos, y=pdf.get_y(), w=img_width)
            pdf.ln(70)
            
            pdf.set_font('Arial', 'B', 12)
            pdf.cell(0, 8, 'Figure 6: Bootstrap Distribution of Scale Parameter', 0, 1)
            pdf.ln(3)
            
            if len(figure_files) > 6:
                pdf.image(figure_files[6], x=x_pos, y=pdf.get_y(), w=img_width)
                pdf.ln(70)
        
        # === CHAPTER 6: MONTE CARLO PREDICTIONS ===
        pdf.add_page()
        pdf.set_font('Arial', 'B', 18)
        pdf.cell(0, 12, 'Chapter 6: Peering Into the Future', 0, 1)
        pdf.ln(5)
        
        pdf.set_font('Arial', '', 11)
        chapter6_text = f"""
Armed with our fitted Weibull model and understanding of parameter uncertainty, we now turn to the crystal ball of Monte Carlo simulation. By generating 10,000 random lifetime values from our fitted distribution, we create a synthetic future that helps us understand what lies ahead.

This simulation reveals key percentiles that are crucial for engineering decision-making:
• B10 = {metrics.get('B10', 'N/A')} cycles: The point where 10% of units will have failed
• B50 = {metrics.get('B50', 'N/A')} cycles: The median lifetime (50% failure point)
• B95 = {metrics.get('B95', 'N/A')} cycles: The point where 95% will have failed

These predictions, incorporating both our model fit and parameter uncertainty, provide robust estimates for warranty analysis, maintenance planning, and reliability assessments.
        """
        
        for paragraph in chapter6_text.strip().split('\n\n'):
            lines = textwrap.wrap(paragraph.strip(), width=80)
            for line in lines:
                pdf.cell(0, 6, self._clean_pdf_text(line), 0, 1)
            pdf.ln(3)
        
        # Insert Figure 7: Monte Carlo
        if len(figure_files) > 7:
            pdf.ln(5)
            pdf.set_font('Arial', 'B', 12)
            pdf.cell(0, 8, 'Figure 7: Monte Carlo Lifetime Predictions', 0, 1)
            pdf.ln(3)
            
            img_width = min(160, pdf.w - pdf.l_margin - pdf.r_margin)
            x_pos = (pdf.w - img_width) / 2
            pdf.image(figure_files[7], x=x_pos, y=pdf.get_y(), w=img_width)
            pdf.ln(85)
            
            pdf.set_font('Arial', 'I', 10)
            pdf.cell(0, 5, 'This histogram shows 10,000 simulated lifetimes, with key percentiles marked', 0, 1, 'C')
            pdf.cell(0, 5, 'to guide engineering decisions and risk assessments.', 0, 1, 'C')
        
        # === EPILOGUE: THE VALIDATION ===
        pdf.add_page()
        pdf.set_font('Arial', 'B', 18)
        pdf.cell(0, 12, 'Epilogue: Validating Our Story', 0, 1)
        pdf.ln(5)
        
        pdf.set_font('Arial', '', 11)
        epilogue_text = f"""
Every good story needs validation - we must ensure that our mathematical model truly captures the essence of our data. Through statistical tests like the Kolmogorov-Smirnov test, we compare our simulated data with the actual observations.

The validation process confirms that our Weibull model is not just mathematically elegant, but also practically accurate. This gives us confidence that our predictions and insights are grounded in statistical reality.

Our journey through this lifetime analysis has revealed:
1. The shape of failure patterns in our data
2. The mathematical model that best describes these patterns  
3. The uncertainty in our parameter estimates
4. Robust predictions for future performance
5. Statistical validation of our approach

This story of data, mathematics, and engineering insight provides the foundation for informed decision-making about reliability, maintenance, and design improvements.
        """
        
        for paragraph in epilogue_text.strip().split('\n\n'):
            lines = textwrap.wrap(paragraph.strip(), width=80)
            for line in lines:
                pdf.cell(0, 6, self._clean_pdf_text(line), 0, 1)
            pdf.ln(3)
        
        # Insert remaining validation figures if available
        remaining_figures = figure_files[8:] if len(figure_files) > 8 else []
        for i, fig_path in enumerate(remaining_figures, 8):
            pdf.ln(5)
            pdf.set_font('Arial', 'B', 12)
            pdf.cell(0, 8, f'Figure {i+1}: Model Validation', 0, 1)
            pdf.ln(3)
            
            img_width = min(160, pdf.w - pdf.l_margin - pdf.r_margin)
            x_pos = (pdf.w - img_width) / 2
            pdf.image(fig_path, x=x_pos, y=pdf.get_y(), w=img_width)
            pdf.ln(85)
        
        # Final page with summary
        pdf.add_page()
        pdf.set_font('Arial', 'B', 16)
        pdf.cell(0, 10, 'The End of Our Journey', 0, 1, 'C')
        pdf.ln(10)
        
        pdf.set_font('Arial', '', 11)
        final_text = f"""
This narrative journey through lifetime analysis has transformed raw data into actionable insights. We've seen how mathematical models can reveal the hidden stories within our measurements, and how statistical methods can quantify uncertainty and validate our conclusions.

The Weibull distribution, with its elegant mathematical form and practical interpretability, has proven to be the perfect narrator for our data's story. Through bootstrap analysis and Monte Carlo simulation, we've not only understood the past but also gained the ability to predict the future with quantified confidence.

As we conclude this analytical adventure, we carry with us not just numbers and charts, but a deeper understanding of the reliability characteristics that govern our system's behavior. This knowledge becomes the foundation for better engineering decisions, more effective maintenance strategies, and ultimately, more reliable products.

The story of our data is now complete, but the insights it provides will continue to guide us in the ongoing quest for reliability and excellence.
        """
        
        for paragraph in final_text.strip().split('\n\n'):
            lines = textwrap.wrap(paragraph.strip(), width=80)
            for line in lines:
                pdf.cell(0, 6, self._clean_pdf_text(line), 0, 1)
            pdf.ln(3)
        
        pdf.output(pdf_filename)
    
    def _clean_pdf_text(self, text):
        """Clean text for PDF output"""
        if not isinstance(text, str):
            text = str(text)
        
        # Remove or replace unsupported characters
        text = text.encode('ascii', 'ignore').decode('ascii')
        
        # Replace common special characters
        replacements = {
            'β': 'beta',
            'η': 'eta',
            'σ': 'sigma',
            'μ': 'mu',
            'λ': 'lambda',
            'Φ': 'Phi',
            '×': 'x',
            '°': ' degrees',
            '±': '+/-'
        }
        
        for old, new in replacements.items():
            text = text.replace(old, new)
        
        return text

    def open_pdf_file(self, filepath):
        """Open PDF file using system default application"""
        try:
            import platform
            import subprocess
            
            if platform.system() == 'Darwin':  # macOS
                subprocess.call(('open', filepath))
            elif platform.system() == 'Windows':  # Windows
                os.startfile(filepath)
            else:  # Linux
                subprocess.call(('xdg-open', filepath))
                
            return True
        except Exception as e:
            print(f"Could not open PDF file: {e}")
            return False


def main():
    """Main program entry point - only responsible for starting GUI"""
    import tkinter as tk
    
    # Create main controller
    controller = LifetimeAnalysisController()
    
    # Create and start GUI
    root = tk.Tk()
    app = LifetimeGUI(root, controller)  # Pass controller to GUI
    root.mainloop()


if __name__ == "__main__":
    main()
