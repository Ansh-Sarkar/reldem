import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import List, Tuple, Dict
import matplotlib.pyplot as plt

@dataclass
class TestParameters:
    """Class to hold configurable test parameters"""
    shape_fm1: float = 2.25  # Weibull shape parameter for Failure Mode 1
    shape_fm2: float = 2.5   # Weibull shape parameter for Failure Mode 2
    scale_fm1: float = 4253.54  # Scale at 7V from your document
    scale_fm2: float = 3938.02  # Scale at 7V from your document
    mission_time: float = 419   # Accelerated mission time
    min_reliability: float = 0.91
    r_u: float = 0.96  # Upper reliability
    r_l: float = 0.9   # Lower reliability
    h0: float = -2.002485457  # Accept line intercept
    h1: float = 2.506600517   # Reject line intercept
    s: float = 0.06579995544  # Slope for both lines

class ReliabilityTest:
    """Class for conducting reliability testing with Weibull distributions"""
    def __init__(self, params: TestParameters):
        self.params = params
        self.results = []

    def weibull_random(self, shape: float, scale: float) -> float:
        """Generate a Weibull random variable"""
        return scale * (-np.log(1 - np.random.random())) ** (1 / shape)

    def run_single_test(self, max_samples: int) -> Dict:
        """Run a single test iteration"""
        data = {
            'S.No': [],
            'F1': [],
            'F2': [],
            'TTF1': [],
            'TTF2': [],
            'System Failure TTF': [],
            'Failed': [],
            'Cumulative Failures': [],
            'ALf': [],
            'RLf': [],
            'Decision': []
        }
        
        cumulative_failures = 0
        for i in range(1, max_samples + 1):
            # Generate failure times for both failure modes
            f1 = np.random.random()
            f2 = np.random.random()
            ttf1 = self.weibull_random(self.params.shape_fm1, self.params.scale_fm1)
            ttf2 = self.weibull_random(self.params.shape_fm2, self.params.scale_fm2)
            system_ttf = min(ttf1, ttf2)
            
            # Check if system failed within mission time
            failed = system_ttf < self.params.mission_time
            if failed:
                cumulative_failures += 1
            
            # Calculate accept/reject lines
            alf = self.params.h0 + self.params.s * i
            rlf = self.params.h1 + self.params.s * i
            
            # Decision logic
            decision = 'Indecisive'
            if alf >= 0:
                decision = 'Accept'
            elif rlf <= cumulative_failures:
                decision = 'Reject'
            
            # Store results
            data['S.No'].append(i)
            data['F1'].append(f1)
            data['F2'].append(f2)
            data['TTF1'].append(ttf1)
            data['TTF2'].append(ttf2)
            data['System Failure TTF'].append(system_ttf)
            data['Failed'].append('Yes' if failed else 'No')
            data['Cumulative Failures'].append(cumulative_failures)
            data['ALf'].append(alf)
            data['RLf'].append(rlf)
            data['Decision'].append(decision)
            
            if decision in ['Accept', 'Reject']:
                break
                
        return data

    def run_multiple_tests(self, num_runs: int, max_samples: int) -> pd.DataFrame:
        """Run multiple test iterations and return results as DataFrame"""
        summary_data = []
        
        for run in range(1, num_runs + 1):
            test_result = self.run_single_test(max_samples)
            self.results.append(test_result)
            
            # Summary for this run
            sample_size = len(test_result['S.No'])
            final_decision = test_result['Decision'][-1]
            percentage = self.params.r_u if final_decision == 'Accept' else self.params.r_l
            
            summary_data.append({
                'Run No': run,
                'Sample Size': sample_size,
                'Result': final_decision,
                '%age Demonstrated': percentage
            })
            
            # Optional: Plot for this run
            self.plot_test(test_result, run)
            
        return pd.DataFrame(summary_data)

    def plot_test(self, test_result: Dict, run_number: int):
        """Plot cumulative failures with accept/reject lines"""
        x_values = test_result['S.No']
        y_values = test_result['Cumulative Failures']
        alf_values = test_result['ALf']
        rlf_values = test_result['RLf']
        
        plt.figure(figsize=(10, 6))
        plt.plot(x_values, y_values, marker='o', linestyle='-', label='Cumulative Failures')
        plt.plot(x_values, alf_values, linestyle='--', label=f'Accept Line: {self.params.h0:.2f} + {self.params.s:.5f}*N')
        plt.plot(x_values, rlf_values, linestyle='--', label=f'Reject Line: {self.params.h1:.2f} + {self.params.s:.5f}*N')
        
        plt.xlabel('Number of Samples (N)')
        plt.ylabel('Cumulative Failures (f)')
        plt.title(f'Reliability Test Run {run_number}')
        plt.legend()
        plt.grid(True)
        plt.show()

def main():
    # Configure test parameters
    params = TestParameters(
        shape_fm1=2.25,
        shape_fm2=2.5,
        scale_fm1=4253.54,  # 7V stress level from your document
        scale_fm2=3938.02,  # 7V stress level from your document
        mission_time=419,
        min_reliability=0.91,
        r_u=0.96,
        r_l=0.9,
        h0=-2.002485457,
        h1=2.506600517,
        s=0.06579995544
    )
    
    # Initialize test
    test = ReliabilityTest(params)
    
    # Run multiple tests
    num_runs = 50
    max_samples = 100
    results_df = test.run_multiple_tests(num_runs, max_samples)
    
    # Print summary
    print("\nTest Parameters:")
    print(f"Shape FM1: {params.shape_fm1}, Scale FM1: {params.scale_fm1}")
    print(f"Shape FM2: {params.shape_fm2}, Scale FM2: {params.scale_fm2}")
    print(f"Mission Time: {params.mission_time} cycles")
    print(f"Minimum Reliability: {params.min_reliability}")
    print(f"Accept Line: ALf = {params.h0} + {params.s}*N")
    print(f"Reject Line: RLf = {params.h1} + {params.s}*N")
    
    print("\nTest Results Summary:")
    print(results_df.to_string(index=False))
    print(f"\nAverage Sample Size: {results_df['Sample Size'].mean():.2f}")
    
    # Save detailed results for first run
    detailed_df = pd.DataFrame(test.results[0])
    detailed_df.to_csv('detailed_test_results_run1.csv', index=False)

if __name__ == "__main__":
    main()