import numpy as np
import pandas as pd
from typing import Tuple, List, Dict
import matplotlib.pyplot as plt
from dataclasses import dataclass
from scipy.special import comb
from scipy.optimize import fsolve
from scipy.optimize import minimize_scalar

@dataclass
class WeibullParameters:
    shape: float      # beta parameter
    nominal: float    # nominal life
    recalc_theta: List[float]  # recalculated theta at different voltages
    af: List[float]  # acceleration factors at different voltages

class ReliabilityDemonstrationPlan:
    def __init__(
        self,
        failure_modes: Dict[str, WeibullParameters],
        test_voltage: float,
        nominal_voltage: float,
        mission_time: float,
        voltage_levels: List[float],
        R_l: float = 0.92,
        R_u: float = 0.97,
        alpha: float = 0.05,
        beta: float = 0.1,
        n_simulations: int = 1000
    ):
        if not failure_modes:
            raise ValueError("At least one failure mode must be provided")
        self.failure_modes = failure_modes
        self.test_voltage = test_voltage
        self.nominal_voltage = nominal_voltage
        self.mission_time = mission_time
        self.voltage_levels = voltage_levels
        self.voltage_index = self._get_voltage_index()
        self.R_l = R_l
        self.R_u = R_u
        self.alpha = alpha
        self.beta = beta
        self.n_simulations = n_simulations
        
        # Calculate test plan parameters
        self.N, self.r, self.actual_alpha, self.actual_beta = self._calculate_test_plan()
        
    def _get_voltage_index(self) -> int:
        """Get the index of the test voltage in voltage levels"""
        try:
            return self.voltage_levels.index(self.test_voltage)
        except ValueError:
            raise ValueError(f"Test voltage {self.test_voltage}V not found in voltage levels {self.voltage_levels}")

    def _calculate_acceleration_factor(self) -> float:
        """Calculate the minimum acceleration factor among all failure modes at the test voltage"""
        try:
            # Get AFs at the test voltage level for each failure mode
            afs = [mode.af[self.voltage_index] for mode in self.failure_modes.values()]
            min_af = min(afs)  # Use minimum AF to observe all failure modes
            print(f"\nAcceleration Factors at {self.test_voltage}V:")
            for name, af in zip(self.failure_modes.keys(), afs):
                print(f"{name}: {af:.2f}")
            print(f"Using minimum AF: {min_af:.2f}")
            return min_af
        except IndexError:
            raise ValueError(f"Missing acceleration factor data for voltage {self.test_voltage}V")

    def _calculate_theta_for_reliability(self, R_target: float) -> Dict[str, float]:
        """
        Calculate theta for each failure mode such that their product reliability equals R_target
        using accelerated mission time
        
        Args:
            R_target: Target reliability (R_l or R_u)
        Returns:
            Dict[str, float]: New theta values for each failure mode
        """
        def solve_for_reliabilities(R_target: float, beta1: float, beta2: float, time: float) -> Tuple[float, float]:
            """
            Find R1 and R2 such that R1 * R2 = R_target
            and the resulting thetas are consistent with the Weibull distributions
            """
            def equations(vars):
                R1, R2 = vars
                # Check if R1 * R2 = R_target
                eq1 = R1 * R2 - R_target
                # Check if ratio of times to reach these reliabilities matches Weibull shapes
                t1 = (-np.log(R1)) ** (1/beta1)
                t2 = (-np.log(R2)) ** (1/beta2)
                eq2 = t1 - t2  # Should be equal at the solution
                return [eq1, eq2]
            
            # Initial guess: equal reliabilities
            R_guess = R_target ** 0.5
            solution = fsolve(equations, [R_guess, R_guess])
            return solution[0], solution[1]
        
        # Calculate minimum acceleration factor to observe all failure modes
        af = self._calculate_acceleration_factor()
        accelerated_time = self.mission_time / af
        
        print(f"\nCalculating thetas for R={R_target:.4f}")
        print(f"Mission time: {self.mission_time} hours")
        print(f"Accelerated time: {accelerated_time:.2f} hours")
        
        # Get shape parameters
        betas = [mode.shape for mode in self.failure_modes.values()]
        names = list(self.failure_modes.keys())
        
        # Solve for individual reliabilities
        R1, R2 = solve_for_reliabilities(R_target, betas[0], betas[1], accelerated_time)
        
        print(f"\nCalculated reliabilities at accelerated time:")
        print(f"{names[0]}: R1 = {R1:.4f}")
        print(f"{names[1]}: R2 = {R2:.4f}")
        print(f"Product R1 * R2 = {R1 * R2:.4f}")
        
        # Calculate thetas based on these reliabilities
        new_thetas = {}
        for name, R, mode in zip(names, [R1, R2], self.failure_modes.values()):
            theta = accelerated_time / (-np.log(R)) ** (1/mode.shape)
            new_thetas[name] = theta
            print(f"\n{name}:")
            print(f"  Original theta at {self.test_voltage}V: {mode.recalc_theta[self.voltage_index]:.2f}")
            print(f"  Recalculated theta for R={R:.4f}: {theta:.2f}")
        
        # Verify reliabilities at accelerated time
        print("\nVerifying reliabilities at accelerated time:")
        system_R = 1.0
        for name, mode in self.failure_modes.items():
            R = np.exp(-(accelerated_time/new_thetas[name])**mode.shape)
            print(f"{name} reliability: {R:.4f}")
            system_R *= R
        
        print(f"System reliability: {system_R:.4f}")
        print(f"Target reliability: {R_target:.4f}")
        print(f"Difference: {abs(system_R - R_target):.6f}")
        
        return new_thetas

    def _calculate_test_plan(self) -> Tuple[int, int, float, float]:
        """Calculate required sample size and acceptance criteria"""
        return calculate_nr_parameters(
            alpha=self.alpha,
            beta=self.beta,
            R_l=self.R_l,
            R_u=self.R_u,
            max_N=200,      # Configurable maximum sample size
            min_N=10,        # Configurable minimum sample size
            deviation_threshold=0.001,  # Configurable acceptance threshold
            progress_interval=50    # Show progress every 50 iterations
        )
    
    def _find_optimal_mission_time(self) -> float:
        """
        Find optimal accelerated mission time where product of reliabilities equals R_u
        by searching between mission_time/max_AF and mission_time/min_AF
        """
        def calculate_system_reliability(t: float) -> float:
            """Calculate system reliability at time t using original thetas"""
            system_R = 1.0
            for mode in self.failure_modes.values():
                theta = mode.recalc_theta[self.voltage_index]
                R = np.exp(-(t/theta)**mode.shape)
                system_R *= R
            return system_R
        
        # Get AFs at test voltage
        afs = [mode.af[self.voltage_index] for mode in self.failure_modes.values()]
        min_af = min(afs)
        max_af = max(afs)
        
        # Define search range for accelerated time
        t_min = self.mission_time / max_af
        t_max = self.mission_time / min_af
        
        def objective(t: float) -> float:
            """Objective function to minimize: difference from R_u"""
            return abs(calculate_system_reliability(t) - self.R_u)
        
        # Use scipy's minimize to find optimal time
        result = minimize_scalar(objective, bounds=(t_min, t_max), method='bounded')
        optimal_time = result.x
        
        # Print detailed results
        print("\nOptimal Mission Time Calculation:")
        print(f"Mission time: {self.mission_time} hours")
        print(f"AF range at {self.test_voltage}V: [{min_af:.2f}, {max_af:.2f}]")
        print(f"Search range: [{t_min:.2f}, {t_max:.2f}] hours")
        print(f"Optimal accelerated time: {optimal_time:.2f} hours")
        
        # Verify reliabilities at optimal time
        print("\nReliabilities at optimal time:")
        system_R = 1.0
        for name, mode in self.failure_modes.items():
            theta = mode.recalc_theta[self.voltage_index]
            R = np.exp(-(optimal_time/theta)**mode.shape)
            print(f"{name}:")
            print(f"  Theta at {self.test_voltage}V: {theta:.2f}")
            print(f"  Reliability: {R:.4f}")
            system_R *= R
        
        print(f"\nSystem reliability: {system_R:.4f}")
        print(f"Target R_u: {self.R_u:.4f}")
        print(f"Difference: {abs(system_R - self.R_u):.6f}")
        
        return optimal_time

    def run_demonstration_test(self) -> Dict:
        """
        Run Monte Carlo simulation of the demonstration test
        """
        results_l = []  # Results for lower reliability bound
        results_u = []  # Results for upper reliability bound
        results_orig = []  # Results for original theta values
        
        # Calculate minimum acceleration factor for R_l and R_u calculations
        af = self._calculate_acceleration_factor()
        accelerated_time = self.mission_time / af
        
        # Calculate thetas for both R_l and R_u using accelerated time
        thetas_l = self._calculate_theta_for_reliability(self.R_l)
        thetas_u = self._calculate_theta_for_reliability(self.R_u)
        
        # Find optimal time for original thetas
        optimal_time = self._find_optimal_mission_time()
        
        print("\nRunning simulation with original thetas:")
        for name, mode in self.failure_modes.items():
            print(f"{name} original theta at {self.test_voltage}V: {mode.recalc_theta[self.voltage_index]:.2f}")
        
        for _ in range(self.n_simulations):
            # Test for R_l (using original accelerated time)
            failures_l = []
            for mode_name, mode in self.failure_modes.items():
                ttf = np.random.weibull(
                    mode.shape, 
                    size=self.N
                ) * thetas_l[mode_name]
                failures_l.append(ttf)
            
            min_ttf_l = np.minimum.reduce(failures_l) if len(failures_l) > 1 else failures_l[0]
            n_failures_l = np.sum(min_ttf_l < accelerated_time)
            results_l.append(n_failures_l)
            
            # Test for R_u (using original accelerated time)
            failures_u = []
            for mode_name, mode in self.failure_modes.items():
                ttf = np.random.weibull(
                    mode.shape, 
                    size=self.N
                ) * thetas_u[mode_name]
                failures_u.append(ttf)
            
            min_ttf_u = np.minimum.reduce(failures_u) if len(failures_u) > 1 else failures_u[0]
            n_failures_u = np.sum(min_ttf_u < accelerated_time)
            results_u.append(n_failures_u)
            
            # Test with original thetas (using optimal time)
            failures_orig = []
            for mode in self.failure_modes.values():
                ttf = np.random.weibull(
                    mode.shape, 
                    size=self.N
                ) * mode.recalc_theta[self.voltage_index]
                failures_orig.append(ttf)
            
            min_ttf_orig = np.minimum.reduce(failures_orig) if len(failures_orig) > 1 else failures_orig[0]
            n_failures_orig = np.sum(min_ttf_orig < optimal_time)
            results_orig.append(n_failures_orig)
        
        return self._analyze_results(results_l, results_u, results_orig)
    
    def _analyze_results(self, results_l: List[int], results_u: List[int], results_orig: List[int]) -> Dict:
        """Analyze simulation results for both bounds and original thetas"""
        results_l_array = np.array(results_l)
        results_u_array = np.array(results_u)
        results_orig_array = np.array(results_orig)
        
        analysis = {
            'lower_bound': {
                'mean_failures': np.mean(results_l_array),
                'std_failures': np.std(results_l_array),
                'acceptance_rate': np.mean(results_l_array <= self.r) * 100,
                'failure_distribution': np.bincount(results_l_array),
            },
            'upper_bound': {
                'mean_failures': np.mean(results_u_array),
                'std_failures': np.std(results_u_array),
                'acceptance_rate': np.mean(results_u_array <= self.r) * 100,
                'failure_distribution': np.bincount(results_u_array),
            },
            'original': {
                'mean_failures': np.mean(results_orig_array),
                'std_failures': np.std(results_orig_array),
                'acceptance_rate': np.mean(results_orig_array <= self.r) * 100,
                'failure_distribution': np.bincount(results_orig_array),
            }
        }
        
        return analysis
    
    def plot_results(self, results: Dict):
        """Generate visualization of results for both bounds and original thetas"""
        fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3, figsize=(18, 10))
        
        # Plot lower bound distribution
        x_l = np.arange(len(results['lower_bound']['failure_distribution']))
        ax1.bar(x_l, results['lower_bound']['failure_distribution'])
        ax1.axvline(self.r, color='r', linestyle='--', label=f'Acceptance criteria (r={self.r})')
        ax1.set_title('Distribution of Failures (R_l)')
        ax1.set_xlabel('Number of Failures')
        ax1.set_ylabel('Frequency')
        ax1.legend()
        
        # Plot upper bound distribution
        x_u = np.arange(len(results['upper_bound']['failure_distribution']))
        ax2.bar(x_u, results['upper_bound']['failure_distribution'])
        ax2.axvline(self.r, color='r', linestyle='--', label=f'Acceptance criteria (r={self.r})')
        ax2.set_title('Distribution of Failures (R_u)')
        ax2.set_xlabel('Number of Failures')
        ax2.set_ylabel('Frequency')
        ax2.legend()
        
        # Plot original theta distribution
        x_orig = np.arange(len(results['original']['failure_distribution']))
        ax3.bar(x_orig, results['original']['failure_distribution'])
        ax3.axvline(self.r, color='r', linestyle='--', label=f'Acceptance criteria (r={self.r})')
        ax3.set_title('Distribution of Failures (Original θ)')
        ax3.set_xlabel('Number of Failures')
        ax3.set_ylabel('Frequency')
        ax3.legend()
        
        # Add summary statistics
        summary_l = f"R_l Acceptance Rate: {results['lower_bound']['acceptance_rate']:.1f}%\n"
        summary_l += f"Mean Failures: {results['lower_bound']['mean_failures']:.1f}\n"
        summary_l += f"Std Dev: {results['lower_bound']['std_failures']:.1f}"
        
        summary_u = f"R_u Acceptance Rate: {results['upper_bound']['acceptance_rate']:.1f}%\n"
        summary_u += f"Mean Failures: {results['upper_bound']['mean_failures']:.1f}\n"
        summary_u += f"Std Dev: {results['upper_bound']['std_failures']:.1f}"
        
        summary_orig = f"Original θ Acceptance Rate: {results['original']['acceptance_rate']:.1f}%\n"
        summary_orig += f"Mean Failures: {results['original']['mean_failures']:.1f}\n"
        summary_orig += f"Std Dev: {results['original']['std_failures']:.1f}"
        
        ax4.text(0.5, 0.5, summary_l, 
                horizontalalignment='center',
                verticalalignment='center',
                transform=ax4.transAxes,
                bbox=dict(facecolor='white', alpha=0.8))
        ax4.axis('off')
        
        ax5.text(0.5, 0.5, summary_u, 
                horizontalalignment='center',
                verticalalignment='center',
                transform=ax5.transAxes,
                bbox=dict(facecolor='white', alpha=0.8))
        ax5.axis('off')
        
        ax6.text(0.5, 0.5, summary_orig, 
                horizontalalignment='center',
                verticalalignment='center',
                transform=ax6.transAxes,
                bbox=dict(facecolor='white', alpha=0.8))
        ax6.axis('off')
        
        plt.tight_layout()
        return fig

def calculate_nr_parameters(
    alpha: float, 
    beta: float, 
    R_l: float, 
    R_u: float,
    max_N: int = 1000,
    min_N: int = 1,
    deviation_threshold: float = 0.001,
    progress_interval: int = 50  # Show progress every X iterations
) -> Tuple[int, int, float, float]:
    """
    Calculate the required sample size (N) and maximum allowable failures (r)
    by minimizing the deviation from target alpha and beta values
    
    Args:
        alpha: Target consumer risk (Type I error)
        beta: Target producer risk (Type II error)
        R_l: Lower reliability bound
        R_u: Upper reliability bound
        max_N: Maximum sample size to consider
        min_N: Minimum sample size to consider
        deviation_threshold: Stop searching if deviation is below this value
        progress_interval: Show progress every X iterations
    
    Returns:
        Tuple[int, int, float, float]: (N, r, actual_alpha, actual_beta)
    """
    def calculate_actual_risks(N: int, r: int) -> Tuple[float, float]:
        """Calculate actual alpha and beta values for given N and r"""
        # Calculate actual beta (producer risk)
        actual_beta = sum(
            comb(N, i) * (1 - R_l) ** i * R_l ** (N - i)
            for i in range(r + 1)
        )
        
        # Calculate actual alpha (consumer risk)
        actual_alpha = 1 - sum(
            comb(N, i) * (1 - R_u) ** i * R_u ** (N - i)
            for i in range(r + 1)
        )
        
        return actual_alpha, actual_beta

    def calculate_deviation(N: int, r: int) -> float:
        """Calculate total deviation from target alpha and beta"""
        actual_alpha, actual_beta = calculate_actual_risks(N, r)
        alpha_deviation = abs(actual_alpha - alpha)
        beta_deviation = abs(actual_beta - beta)
        return alpha_deviation + beta_deviation

    # Validate inputs
    if min_N < 1:
        raise ValueError("min_N must be at least 1")
    if max_N < min_N:
        raise ValueError("max_N must be greater than min_N")
    if R_l >= R_u:
        raise ValueError("R_l must be less than R_u")
    if alpha <= 0 or alpha >= 1 or beta <= 0 or beta >= 1:
        raise ValueError("alpha and beta must be between 0 and 1")

    # Initialize search variables
    best_N = min_N
    best_r = 0
    min_deviation = float('inf')
    best_alpha = 0
    best_beta = 0
    total_iterations = max_N - min_N + 1
    
    print("\nSearching for optimal N and r values...")
    print(f"Parameters: R_l={R_l:.4f}, R_u={R_u:.4f}, alpha={alpha:.4f}, beta={beta:.4f}")
    print(f"Search range: N ∈ [{min_N}, {max_N}]")
    print("-" * 50)

    # Try different values of N
    for N in range(min_N, max_N + 1):
        if N % progress_interval == 0:
            progress = (N - min_N + 1) / total_iterations * 100
            print(f"Progress: {progress:.1f}% (testing N={N})")
        
        # Calculate reasonable range for r based on N
        max_r = min(N, int(N * 0.5))  # Limit r to 50% of N as a reasonable upper bound
        
        # Try different values of r for this N
        for r in range(max_r + 1):
            deviation = calculate_deviation(N, r)
            
            if deviation < min_deviation:
                actual_alpha, actual_beta = calculate_actual_risks(N, r)
                
                # Update best values
                min_deviation = deviation
                best_N = N
                best_r = r
                best_alpha = actual_alpha
                best_beta = actual_beta
                
                # Print when we find a better solution
                print(f"Found better solution: N={N}, r={r}, deviation={deviation:.4f}")

        # If we've found a good enough solution, stop searching
        if min_deviation < deviation_threshold:
            print(f"\nFound solution within deviation threshold ({deviation_threshold})")
            break

    print("\n" + "=" * 50)
    print("Final Results:")
    print("=" * 50)
    print(f"Sample size (N): {best_N}")
    print(f"Allowable failures (r): {best_r}")
    print(f"Target consumer risk (alpha): {alpha:.4f}")
    print(f"Actual consumer risk (alpha): {best_alpha:.4f}")
    print(f"Target producer risk (beta): {beta:.4f}")
    print(f"Actual producer risk (beta): {best_beta:.4f}")
    print(f"Total deviation: {min_deviation:.4f}")
    print("=" * 50)

    return best_N, best_r, best_alpha, best_beta

# Example usage
if __name__ == "__main__":
    # Initialize failure modes from your data
    fm1 = WeibullParameters(
        shape=1.796666667,
        nominal=50000,
        recalc_theta=[17133.22785, 2239.451016, 384.2800754],  # For 6V, 7V, 8V
        af=[11.0966643, 84.89655568, 494.7476853]  # For 6V, 7V, 8V
    )
    
    fm2 = WeibullParameters(
        shape=2.25,
        nominal=60000,
        recalc_theta=[13969.82242, 1972.27131, 361.8003067],  # For 6V, 7V, 8V
        af=[10.12982225, 71.75068525, 391.1323882]  # For 6V, 7V, 8V
    )
    
    # Create demonstration plan
    plan = ReliabilityDemonstrationPlan(
        n_simulations=10000,
        failure_modes={'Flash Memory': fm1, 'USB Controller': fm2},
        test_voltage=7,  # V - User specified test voltage
        nominal_voltage=5,  # V
        mission_time=20000,  # hours
        voltage_levels=[6, 7, 8],  # Available voltage levels
        R_l=0.92,
        R_u=0.97,
        alpha=0.05,
        beta=0.1
    )
    
    # Run demonstration
    results = plan.run_demonstration_test()
    
    # Plot results
    fig = plan.plot_results(results)
    plt.show()
    
    # Print detailed results
    print("\nTest Plan Parameters:")
    print(f"Required Sample Size (N): {plan.N}")
    print(f"Acceptance Number (r): {plan.r}")
    print(f"Actual Consumer Risk (α): {plan.actual_alpha:.4f}")
    print(f"Actual Producer Risk (β): {plan.actual_beta:.4f}")
    
    print("\nSimulation Results:")
    for key, value in results.items():
        if isinstance(value, (int, float)):
            print(f"{key}: {value:.2f}")
