import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')
from scipy.stats import weibull_min, lognorm, expon, ks_2samp, anderson
from scipy.special import gamma
from statsmodels.distributions.empirical_distribution import ECDF

class LifetimeAnalyzer:
    def __init__(self, filepath):
        self.filepath = filepath
        ext = os.path.splitext(filepath)[-1].lower()

        if ext == ".xlsx":
            df = pd.read_excel(filepath, header=None)
        elif ext in [".csv", ".txt"]:
            df = pd.read_csv(filepath, header=None)
        else:
            raise ValueError("Unsupported file format. Please upload .csv, .txt, or .xlsx files.")

        # Combine all columns into 1D array, supporting row/column formats
        flat_data = pd.to_numeric(df.values.flatten(), errors='coerce')
        self.data = flat_data[~np.isnan(flat_data)]  # Remove NaN values
        if len(self.data) == 0:
            raise ValueError("No valid numeric data found in the file.")

        self.x = np.linspace(min(self.data), max(self.data), 200)
        self.results = {}
        self.best_model = None

    def fit_distributions(self):
        # === Weibull ===
        wb_params = weibull_min.fit(self.data, floc=0)
        wb_beta, _, wb_eta = wb_params
        wb_pdf = weibull_min.pdf(self.x, *wb_params)
        wb_sf = weibull_min.sf(self.x, *wb_params)
        wb_ll = np.sum(weibull_min.logpdf(self.data, *wb_params))
        wb_mttf = wb_eta * gamma(1 + 1 / wb_beta)
        wb_label = rf"Weibull: $S(t)=\exp(-(\frac{{t}}{{\eta}})^{{\beta}})$, " \
                   rf"$\beta={wb_beta:.2f}$, $\eta={wb_eta:.2f}$, MTTF={wb_mttf:.2f}"

        self.results['Weibull'] = {
            'pdf': wb_pdf, 'sf': wb_sf, 'll': wb_ll,
            'mttf': wb_mttf, 'label': wb_label, 'color': 'orange'
        }
        self.weibull_params = {
            'beta': wb_beta,
            'eta': wb_eta
        }


        # === Lognormal ===
        ln_params = lognorm.fit(self.data, floc=0)
        ln_sigma, _, ln_scale = ln_params
        ln_mu = np.log(ln_scale)
        ln_pdf = lognorm.pdf(self.x, *ln_params)
        ln_sf = lognorm.sf(self.x, *ln_params)
        ln_ll = np.sum(lognorm.logpdf(self.data, *ln_params))
        ln_mttf = np.exp(ln_mu + 0.5 * ln_sigma**2)
        ln_label = rf"Lognormal: $S(t)=1-\Phi((\ln t-\mu)/\sigma)$, " \
                   rf"$\mu={ln_mu:.2f}$, $\sigma={ln_sigma:.2f}$, MTTF={ln_mttf:.2f}"

        self.results['Lognormal'] = {
            'pdf': ln_pdf, 'sf': ln_sf, 'll': ln_ll,
            'mttf': ln_mttf, 'label': ln_label, 'color': 'green'
        }

        # === Exponential ===
        ex_params = expon.fit(self.data, floc=0)
        ex_lambda = 1 / ex_params[1] if ex_params[1] != 0 else 1 / np.mean(self.data)
        ex_pdf = expon.pdf(self.x, *ex_params)
        ex_sf = expon.sf(self.x, *ex_params)
        ex_ll = np.sum(expon.logpdf(self.data, *ex_params))
        ex_mttf = 1 / ex_lambda
        ex_label = rf"Exponential: $S(t)=\exp(-\lambda t)$, " \
                   rf"$\lambda={ex_lambda:.2f}$, MTTF={ex_mttf:.2f}"

        self.results['Exponential'] = {
            'pdf': ex_pdf, 'sf': ex_sf, 'll': ex_ll,
            'mttf': ex_mttf, 'label': ex_label, 'color': 'red'
        }

        # === Determine Best Fit ===
        self.best_model = max(self.results, key=lambda k: self.results[k]['ll'])
        
    def perform_anderson_darling_test(self):
        """Perform Anderson-Darling goodness-of-fit test"""
        ad_results = {}
        
        # Anderson-Darling test for normal distribution
        try:
            # Standardize data for normality test
            standardized_data = (self.data - np.mean(self.data)) / np.std(self.data)
            ad_stat, critical_values, significance_level = anderson(standardized_data, dist='norm')
            
            # Determine whether to reject null hypothesis
            reject_null = ad_stat > critical_values[2]  # Use 5% significance level
            
            ad_results['Normal'] = {
                'statistic': ad_stat,
                'critical_values': critical_values,
                'significance_levels': [15, 10, 5, 2.5, 1],
                'reject_null': reject_null,
                'conclusion': 'Reject normality' if reject_null else 'Cannot reject normality'
            }
        except Exception as e:
            ad_results['Normal'] = {'error': str(e)}
        
        # Anderson-Darling test for exponential distribution
        try:
            # Transform data using MLE estimated parameters
            ex_params = expon.fit(self.data, floc=0)
            transformed_data = expon.cdf(self.data, *ex_params)
            # Transform to standard uniform then to standard exponential distribution
            exp_transformed = -np.log(1 - transformed_data + 1e-10)  # Avoid log(0)
            
            ad_stat, critical_values, significance_level = anderson(exp_transformed, dist='expon')
            reject_null = ad_stat > critical_values[2]
            
            ad_results['Exponential'] = {
                'statistic': ad_stat,
                'critical_values': critical_values,
                'significance_levels': [15, 10, 5, 2.5, 1],
                'reject_null': reject_null,
                'conclusion': 'Reject exponential fit' if reject_null else 'Cannot reject exponential fit'
            }
        except Exception as e:
            ad_results['Exponential'] = {'error': str(e)}
        
        return ad_results
    
    def calculate_maintenance_recommendations(self, target_reliability=0.9, cost_per_maintenance=1000, 
                                           cost_per_failure=10000):
        """Calculate maintenance cycle recommendations"""
        if self.best_model != 'Weibull':
            return {'error': 'Maintenance recommendations are only available for Weibull distribution'}
        
        wb_params = weibull_min.fit(self.data, floc=0)
        beta, _, eta = wb_params
        
        # Calculate time corresponding to target reliability
        target_time = eta * (-np.log(target_reliability))**(1/beta)
        
        # Calculate costs for different maintenance strategies
        maintenance_intervals = np.linspace(target_time * 0.5, target_time * 1.5, 20)
        total_costs = []
        
        for interval in maintenance_intervals:
            # Failure probability within maintenance interval
            failure_prob = 1 - np.exp(-(interval/eta)**beta)
            
            # Expected cost = maintenance cost + failure cost × failure probability
            expected_cost = cost_per_maintenance + cost_per_failure * failure_prob
            total_costs.append(expected_cost)
        
        # Find optimal maintenance interval
        optimal_index = np.argmin(total_costs)
        optimal_interval = maintenance_intervals[optimal_index]
        optimal_cost = total_costs[optimal_index]
        optimal_reliability = np.exp(-(optimal_interval/eta)**beta)
        
        # Calculate common maintenance indicators
        b10_life = eta * (-np.log(0.9))**(1/beta)
        b50_life = eta * (-np.log(0.5))**(1/beta)
        
        recommendations = {
            'optimal_maintenance_interval': optimal_interval,
            'optimal_reliability': optimal_reliability,
            'expected_cost_per_cycle': optimal_cost,
            'target_reliability_interval': target_time,
            'b10_maintenance_interval': b10_life * 0.8,  # 80% of B10 life
            'b50_maintenance_interval': b50_life * 0.6,  # 60% of B50 life
            'maintenance_strategy': {
                'conservative': b10_life * 0.7,
                'balanced': optimal_interval,
                'aggressive': target_time
            },
            'cost_analysis': {
                'intervals': maintenance_intervals.tolist(),
                'costs': total_costs
            }
        }
        
        return recommendations

    def plot_histogram(self):
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist(self.data, bins=30, density=True, alpha=0.5, color='skyblue', edgecolor='black')
        ax.set_xlabel("Life Cycles", fontsize=12)
        ax.set_ylabel("Probability Density", fontsize=12)
        ax.set_title("Histogram of Lifetime Data", fontsize=14, fontweight='bold')
        ax.grid(True, linestyle='--', alpha=0.6)
        fig.tight_layout()
        return fig

    def plot_pdf_comparison(self):
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist(self.data, bins=30, density=True, alpha=0.5)
        for name, res in self.results.items():
            ax.plot(self.x, res['pdf'], color=res['color'], label=res['label'])
            ax.axvline(res['mttf'], color=res['color'], linestyle='--', alpha=0.6)
        ax.set_xlabel("Lifetime Cycles")
        ax.set_ylabel("Frequency")
        ax.set_title("PDF Distribution Comparison")
        ax.legend(title=f"Best Fit: {self.best_model}", fontsize=10)
        ax.grid(True)
        fig.tight_layout()
        return fig

    def plot_survival_comparison(self):
        fig, ax = plt.subplots(figsize=(10, 6))
        for name, res in self.results.items():
            ax.plot(self.x, res['sf'], color=res['color'], label=res['label'], linewidth=2)
            ax.axvline(res['mttf'], color=res['color'], linestyle='--', alpha=0.6, linewidth=1)
        
        ax.set_xlabel("Lifetime Cycles", fontsize=12)
        ax.set_ylabel("Survival Probability", fontsize=12)
        ax.set_title("Survival Function Comparison", fontsize=14, fontweight='bold')
        ax.legend(title=f"Best Fit: {self.best_model}", fontsize=10, loc='upper right')
        ax.grid(True, alpha=0.3)
        
        fig.tight_layout()
        return fig

    def plot_hazard_rate_comparison(self):
        """Plot hazard rate function comparison"""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        for name, res in self.results.items():
            if name == 'Weibull':
                # Weibull hazard rate function: h(t) = (beta/eta) * (t/eta)^(beta-1)
                wb_params = weibull_min.fit(self.data, floc=0)
                beta, _, eta = wb_params
                hazard = (beta/eta) * (self.x/eta)**(beta-1)
                ax.plot(self.x, hazard, color=res['color'], label=f"{name}: $h(t) = \\frac{{\\beta}}{{\\eta}}\\left(\\frac{{t}}{{\\eta}}\\right)^{{\\beta-1}}$", linewidth=2)
                
            elif name == 'Exponential':
                # Exponential distribution hazard rate function: h(t) = lambda (constant)
                ex_params = expon.fit(self.data, floc=0)
                lambda_rate = 1 / ex_params[1] if ex_params[1] != 0 else 1 / np.mean(self.data)
                hazard = np.full_like(self.x, lambda_rate)
                ax.plot(self.x, hazard, color=res['color'], label=f"{name}: $h(t) = \\lambda = {lambda_rate:.3f}$", linewidth=2)
                
            elif name == 'Lognormal':
                # Lognormal distribution hazard rate function: h(t) = f(t) / S(t)
                ln_params = lognorm.fit(self.data, floc=0)
                pdf_vals = lognorm.pdf(self.x, *ln_params)
                sf_vals = lognorm.sf(self.x, *ln_params)
                hazard = np.divide(pdf_vals, sf_vals, out=np.zeros_like(pdf_vals), where=sf_vals!=0)
                ax.plot(self.x, hazard, color=res['color'], label=f"{name}: $h(t) = \\frac{{f(t)}}{{S(t)}}$", linewidth=2)
        
        ax.set_xlabel("Lifetime Cycles", fontsize=12)
        ax.set_ylabel("Hazard Rate", fontsize=12)
        ax.set_title("Hazard Rate Function Comparison", fontsize=14, fontweight='bold')
        ax.legend(title=f"Best Fit: {self.best_model}", fontsize=10, loc='upper right')
        ax.grid(True, alpha=0.3)
        
        fig.tight_layout()
        return fig

    def plot_weibull_probability_paper(self):
        """Plot Weibull probability paper"""
        if self.best_model != 'Weibull':
            print("Warning: Best fit model is not Weibull. Weibull probability paper may not be appropriate.")
        
        # Calculate empirical distribution function
        sorted_data = np.sort(self.data)
        n = len(sorted_data)
        # Use median rank estimation
        empirical_cdf = (np.arange(1, n+1) - 0.3) / (n + 0.4)
        
        # Weibull probability paper transformation: ln(-ln(1-F)) vs ln(t)
        ln_t = np.log(sorted_data)
        ln_ln_inv_reliability = np.log(-np.log(1 - empirical_cdf))
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Plot data points
        ax.scatter(ln_t, ln_ln_inv_reliability, alpha=0.6, s=50, color='blue', 
                  label='Observed Data', edgecolors='black', linewidth=0.5)
        
        # Fit line (Weibull parameters)
        wb_params = weibull_min.fit(self.data, floc=0)
        beta, _, eta = wb_params
        
        # Theoretical line: ln(-ln(1-F)) = beta * ln(t) - beta * ln(eta)
        theoretical_ln_ln = beta * ln_t - beta * np.log(eta)
        ax.plot(ln_t, theoretical_ln_ln, 'r-', linewidth=2, 
               label=f'Weibull Fit: β={beta:.2f}, η={eta:.2f}')
        
        # Add characteristic life line (eta)
        eta_line_x = np.log(eta)
        eta_line_y = np.log(-np.log(1 - 0.632))  # ln(-ln(1-0.632)) ≈ 0
        ax.axvline(eta_line_x, color='green', linestyle='--', alpha=0.7, 
                  label=f'Characteristic Life (η={eta:.1f})')
        ax.axhline(0, color='gray', linestyle=':', alpha=0.5)
        
        ax.set_xlabel('ln(Lifetime)', fontsize=12)
        ax.set_ylabel('ln(-ln(1-F))', fontsize=12)
        ax.set_title('Weibull Probability Paper', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10, loc='lower right')
        ax.grid(True, alpha=0.3)
        
        # Add secondary tick labels
        ax2 = ax.twiny()
        ax2.set_xlim(ax.get_xlim())
        lifetime_ticks = [10, 50, 100, 500, 1000, 5000, 10000]
        ln_lifetime_ticks = [np.log(t) for t in lifetime_ticks if min(self.data) <= t <= max(self.data)]
        if ln_lifetime_ticks:
            ax2.set_xticks(ln_lifetime_ticks)
            ax2.set_xticklabels([f'{int(np.exp(t))}' for t in ln_lifetime_ticks])
            ax2.set_xlabel('Lifetime Cycles', fontsize=12)
        
        fig.tight_layout()
        return fig

class LifetimeAnalyzerMC:
    def __init__(self, data, beta=None, eta=None):
        self.data = np.array(data)
        self.beta = beta
        self.eta = eta

    def print_bootstrap_weibull_params(self, n_bootstrap=1000):
        n_samples = len(self.data)
        bootstrap_params = []
        for _ in range(n_bootstrap):
            sample = np.random.choice(self.data, size=n_samples, replace=True)
            params = weibull_min.fit(sample, floc=0)
            bootstrap_params.append(params)
        bootstrap_params = np.array(bootstrap_params)
        beta_samples = bootstrap_params[:, 0]
        eta_samples = bootstrap_params[:, 2]
        beta_ci = np.percentile(beta_samples, [2.5, 97.5])
        eta_ci = np.percentile(eta_samples, [2.5, 97.5])
        beta_mean = np.mean(beta_samples)
        eta_mean = np.mean(eta_samples)
        print("\n[BOLD14] === [1] Bootstrap Analysis - Parameter Estimation ===")
        print(f"Method: Resampling from original dataset (n = {n_samples}) with replacement, {n_bootstrap} iterations")
        print(f"Estimated Weibull parameters:")
        print(f"  - Beta (shape):")
        print(f"      - 95% CI  : [{beta_ci[0]:.2f}, {beta_ci[1]:.2f}]")
        print(f"      - Mean    : {beta_mean:.2f}")
        print(f"  - Eta (scale):")
        print(f"      - 95% CI  : [{eta_ci[0]:.2f}, {eta_ci[1]:.2f}]")
        print(f"      - Mean    : {eta_mean:.2f}")

        self.beta = beta_mean
        self.eta = eta_mean
        

        return {
            'beta_ci': beta_ci,
            'eta_ci': eta_ci,
            'beta_mean': beta_mean,
            'eta_mean': eta_mean,
            'beta_samples': beta_samples,
            'eta_samples': eta_samples
        }

    def print_monte_carlo_lifetime(self, beta=None, eta=None, n_simulations=10000):
        beta = beta if beta is not None else self.beta
        eta = eta if eta is not None else self.eta
        simulated_data = weibull_min.rvs(beta, scale=eta, size=n_simulations)
        percentiles = np.percentile(simulated_data, [10, 50, 90, 95, 99])
        mttf = np.mean(simulated_data)
        print("\n[BOLD14] === [2] Simulation - Lifetime Prediction ===")
        print(f"Method: Simulated {n_simulations:,} failure times from Weibull(beta={beta:.2f}, eta={eta:.2f})")
        print(f"Predicted lifecycle percentiles:")
        print(f"  - B10 (10% fail)  : {percentiles[0]:.2f}")
        print(f"  - B50 (median)    : {percentiles[1]:.2f}")
        print(f"  - B95 (95% fail)  : {percentiles[3]:.2f}")
        print(f"  - MTTF (Average)  : {mttf:.2f}")
        return {
            'B10': percentiles[0],
            'B50': percentiles[1],
            'B95': percentiles[3],
            'MTTF': mttf,
            'simulated_data': simulated_data
        }

    def print_reliability_at_cycles(self, cycles, beta=None, eta=None, ci=0.95, n_bootstrap=1000):
        beta = beta if beta is not None else self.beta
        eta = eta if eta is not None else self.eta
        reliabilities = []
        for _ in range(n_bootstrap):
            sample = np.random.choice(self.data, size=len(self.data), replace=True)
            params = weibull_min.fit(sample, floc=0)
            b, _, e = params
            reliability = np.exp(-(cycles / e) ** b)
            reliabilities.append(reliability)
        reliabilities = np.array(reliabilities)
        lower = np.percentile(reliabilities, (1 - ci) / 2 * 100)
        upper = np.percentile(reliabilities, (1 + ci) / 2 * 100)
        median = np.median(reliabilities)
        print(f"\n[BOLD14] === [3] Reliability Estimation at {cycles} Cycles ===")
        print(f"Method: Bootstrap reliability curve using {n_bootstrap} resampled datasets")
        print(f"Estimated reliability at cycle = {cycles}:")
        print(f"  - Median reliability     : {median:.3f}")
        print(f"  - {int(ci*100)}% Confidence Interval : [{lower:.3f}, {upper:.3f}]")

        return {
            'reliability_median': median,
            'reliability_lower': lower,
            'reliability_upper': upper
        }

class LifetimeAnalyzerPlot:
    def __init__(self, data, bootstrap_result=None, mc_result=None):
        self.data = np.array(data)
        self.bootstrap_result = bootstrap_result
        self.mc_result = mc_result

    def plot_bootstrap_histograms(self):
        """Split Bootstrap parameter distributions into two separate plots"""
        beta_samples = self.bootstrap_result['beta_samples']
        eta_samples = self.bootstrap_result['eta_samples']
        beta_mean = np.mean(beta_samples)
        beta_ci = np.percentile(beta_samples, [2.5, 97.5])
        eta_mean = np.mean(eta_samples)
        eta_ci = np.percentile(eta_samples, [2.5, 97.5])

        # Create Beta parameter plot
        fig_beta, ax_beta = plt.subplots(figsize=(8, 6))
        ax_beta.hist(beta_samples, bins=30, color='skyblue', edgecolor='black', alpha=0.7)
        ax_beta.axvline(beta_mean, color='red', linestyle='-', linewidth=2, label=f'Mean = {beta_mean:.2f}')
        ax_beta.axvline(beta_ci[0], color='green', linestyle='--', linewidth=2, label=f'2.5% = {beta_ci[0]:.2f}')
        ax_beta.axvline(beta_ci[1], color='green', linestyle='--', linewidth=2, label=f'97.5% = {beta_ci[1]:.2f}')
        ax_beta.set_title('Bootstrap Distribution of Beta (Shape Parameter)', fontsize=14, fontweight='bold')
        ax_beta.set_xlabel('Beta Value', fontsize=12)
        ax_beta.set_ylabel('Frequency', fontsize=12)
        ax_beta.legend(fontsize=10)
        ax_beta.grid(True, alpha=0.3)
        fig_beta.tight_layout()

        # Create Eta parameter plot
        fig_eta, ax_eta = plt.subplots(figsize=(8, 6))
        ax_eta.hist(eta_samples, bins=30, color='lightgreen', edgecolor='black', alpha=0.7)
        ax_eta.axvline(eta_mean, color='red', linestyle='-', linewidth=2, label=f'Mean = {eta_mean:.2f}')
        ax_eta.axvline(eta_ci[0], color='green', linestyle='--', linewidth=2, label=f'2.5% = {eta_ci[0]:.2f}')
        ax_eta.axvline(eta_ci[1], color='green', linestyle='--', linewidth=2, label=f'97.5% = {eta_ci[1]:.2f}')
        ax_eta.set_title('Bootstrap Distribution of Eta (Scale Parameter)', fontsize=14, fontweight='bold')
        ax_eta.set_xlabel('Eta Value', fontsize=12)
        ax_eta.set_ylabel('Frequency', fontsize=12)
        ax_eta.legend(fontsize=10)
        ax_eta.grid(True, alpha=0.3)
        fig_eta.tight_layout()

        return [fig_beta, fig_eta]

    def plot_mc_histogram(self):
        simulated_data = self.mc_result['simulated_data']
        percentiles = [
            self.mc_result['B10'],
            self.mc_result['B50'],
            np.percentile(simulated_data, 90),
            self.mc_result['B95'],
            np.percentile(simulated_data, 99)
        ]
        labels = ['B10', 'B50', 'B90', 'B95', 'B99']

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.hist(simulated_data, bins=50, density=True, alpha=0.6, color='skyblue', edgecolor='black')

        for i, p in enumerate(percentiles):
            ax.axvline(p, color='red', linestyle='--', alpha=0.7, linewidth=1.5)
            y_text = ax.get_ylim()[1] * (0.85 - i * 0.03)
            x_text = p + (ax.get_xlim()[1] - ax.get_xlim()[0]) * 0.01
            ax.text(x_text, y_text, f'{labels[i]}: {p:.1f}', rotation=90,
                    verticalalignment='center', color='red', fontsize=9, fontweight='bold')

        ax.set_title('Monte Carlo Lifetime Predictions', fontsize=14, fontweight='bold')
        ax.set_xlabel('Lifetime Cycles', fontsize=12)
        ax.set_ylabel('Probability Density', fontsize=12)
        ax.grid(True, linestyle='--', alpha=0.3)
        
        fig.tight_layout()
        return fig

    def plot_simulated_vs_actual(self):
        """Split simulated vs actual data comparison into two separate plots"""
        simulated_data = self.mc_result['simulated_data']
        actual_data = self.data
        beta_mean = self.bootstrap_result['beta_mean']
        eta_mean = self.bootstrap_result['eta_mean']
        loc = 0
        x_vals = np.linspace(0, max(max(simulated_data), max(actual_data)), 500)

        # Create histogram comparison
        fig_hist, ax_hist = plt.subplots(figsize=(10, 6))
        ax_hist.hist(actual_data, bins=30, density=True, alpha=0.6, label='Actual Data', 
                    color='steelblue', edgecolor='black')
        ax_hist.hist(simulated_data, bins=50, density=True, alpha=0.4, label='Simulated Data', 
                    color='orange', edgecolor='black')
        ax_hist.plot(x_vals, weibull_min.pdf(x_vals, beta_mean, loc=loc, scale=eta_mean), 
                    'r--', lw=2, label='Theoretical PDF')
        ax_hist.set_title('Histogram Comparison: Simulated vs Actual Data', fontsize=14, fontweight='bold')
        ax_hist.set_xlabel('Lifetime Cycles', fontsize=12)
        ax_hist.set_ylabel('Probability Density', fontsize=12)
        ax_hist.legend(fontsize=10)
        ax_hist.grid(True, linestyle='--', alpha=0.5)
        fig_hist.tight_layout()

        # Create CDF comparison and KS test
        ecdf_actual = ECDF(actual_data)
        ecdf_sim = ECDF(simulated_data)
        ks_stat, p_value = ks_2samp(actual_data, simulated_data)
        x_common = np.linspace(min(actual_data.min(), simulated_data.min()),
                               max(actual_data.max(), simulated_data.max()), 1000)
        y_actual = ECDF(actual_data)(x_common)
        y_sim = ECDF(simulated_data)(x_common)
        diff = np.abs(y_actual - y_sim)
        max_diff_index = np.argmax(diff)
        ks_x = x_common[max_diff_index]
        ks_y1 = y_actual[max_diff_index]
        ks_y2 = y_sim[max_diff_index]

        fig_cdf, ax_cdf = plt.subplots(figsize=(10, 6))
        ax_cdf.plot(ecdf_actual.x, ecdf_actual.y, label='Empirical CDF (Actual)', 
                   lw=2, color='blue')
        ax_cdf.plot(ecdf_sim.x, ecdf_sim.y, label='Empirical CDF (Simulated)', 
                   lw=2, color='orange', linestyle='--')
        ax_cdf.plot(x_vals, weibull_min.cdf(x_vals, beta_mean, loc=loc, scale=eta_mean), 
                   'r-', lw=2, label='Theoretical CDF')
        ax_cdf.vlines(ks_x, ks_y1, ks_y2, color='black', linestyle=':', linewidth=2, 
                     label=f'KS Distance = {ks_stat:.3f}')
        ax_cdf.text(ks_x + (max(actual_data) * 0.05), (ks_y1 + ks_y2) / 2, 
                   f"KS Statistic: {ks_stat:.3f}\np-value: {p_value:.4f}", 
                   color='black', fontsize=10, bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
        ax_cdf.set_title('Cumulative Distribution Function Comparison with KS Test', fontsize=14, fontweight='bold')
        ax_cdf.set_xlabel('Lifetime Cycles', fontsize=12)
        ax_cdf.set_ylabel('Cumulative Probability', fontsize=12)
        ax_cdf.legend(fontsize=10)
        ax_cdf.grid(True, linestyle='--', alpha=0.5)
        fig_cdf.tight_layout()

        ks_result = {
            'ks_stat': ks_stat,
            'p_value': p_value,
            'conclusion': (
                "Simulated and actual data are likely from the same distribution (p > 0.05)."
                if p_value > 0.05
                else "Simulated and actual data are significantly different (p < 0.05)."
            )
        }
        return [fig_hist, fig_cdf], ks_result
