# federated_system.py
"""Core federated learning system with Byzantine workers"""

import numpy as np
from attacks import create_attackers
from aggregators import nnm_preprocessing, get_aggregator
from optimizers import get_optimizer


class FederatedSystem:
    """Federated learning system with Byzantine-robust optimization"""
    
    def __init__(self, n_workers=20, n_byzantine=3, dimension=10, seed=None, gradient_noise=0.00001):
        self.n_workers = n_workers
        self.n_byzantine = n_byzantine
        self.n_honest = n_workers - n_byzantine
        self.dimension = dimension
        self.gradient_noise = gradient_noise
        
        if seed is not None:
            np.random.seed(seed)
        
        self.x = np.ones(dimension) * 1.0
        
        # Fixed shifts for honest workers that sum to zero (unbiased)
        self.worker_shifts = [np.random.randn(dimension) * 1e-6 for _ in range(self.n_honest - 1)]
        self.worker_shifts.append(-np.sum(self.worker_shifts, axis=0))
        
        self.gradient_norms = []
        self.objective_values = []
        
    def compute_local_gradient(self, x, worker_id):
        """
        Compute local gradient: ∇f_i(x) = 4 * x * ||x||^2
        Adds Gaussian noise and fixed shift for heterogeneity
        """
        x_norm = np.linalg.norm(x)
        x_norm_clipped = min(x_norm, 100.0)
        
        gradient = 4.0 * x * (x_norm_clipped ** 2)
        
        noise = np.random.randn(self.dimension) * self.gradient_noise
        gradient = gradient + noise
        
        if worker_id < self.n_honest:
            gradient = gradient + self.worker_shifts[worker_id]
        
        grad_norm = np.linalg.norm(gradient)
        if grad_norm > 1000.0:
            gradient = gradient * (1000.0 / grad_norm)
        
        if not np.all(np.isfinite(gradient)):
            gradient = np.zeros_like(x)
        
        return gradient
    
    def compute_global_objective(self, x):
        """Compute global objective: f(x) = ||x||^4"""
        x_norm = np.linalg.norm(x)
        x_norm_clipped = min(x_norm, 100.0)
        objective = x_norm_clipped ** 4
        return objective
    
    def run_experiment(self, attack_type, aggregator_type, optimizer_type, 
                       n_iterations=1500, lr=0.01, momentum=0.9, verbose=True):
        """Run federated learning experiment"""
        self.x = np.ones(self.dimension) * 1.0
        self.gradient_norms = []
        self.objective_values = []
        
        attackers = create_attackers(attack_type, self.n_byzantine, self.n_workers)
        aggregator = get_aggregator(aggregator_type)
        optimizer = get_optimizer(optimizer_type, lr, momentum)
        
        attack_name = attackers[0].name if attackers else "None"
        
        if verbose:
            print(f"\nRunning experiment:")
            print(f"  Attack: {attack_name}")
            print(f"  Aggregator: {aggregator_type.upper()}")
            print(f"  Optimizer: {optimizer.name}")
            print("-" * 50)
        
        for iteration in range(n_iterations):
            worker_gradients = []
            honest_gradients = []
            
            for worker_id in range(self.n_workers):
                local_grad = self.compute_local_gradient(self.x, worker_id)
                
                if worker_id < self.n_honest:
                    worker_gradients.append(local_grad)
                    honest_gradients.append(local_grad)
                else:
                    attacker_idx = worker_id - self.n_honest
                    malicious_grad = attackers[attacker_idx].attack(
                        local_grad, honest_gradients, iteration
                    )
                    worker_gradients.append(malicious_grad)
            
            mixed_gradients = nnm_preprocessing(worker_gradients, self.n_workers, self.n_byzantine)
            aggregated_grad = aggregator(mixed_gradients, self.n_byzantine)
            update = optimizer.step(aggregated_grad)
            self.x += update
            
            if not np.all(np.isfinite(self.x)):
                if verbose:
                    print(f"  Warning: Non-finite values detected at iteration {iteration}. Stopping.")
                break
            
            grad_norm = np.linalg.norm(aggregated_grad)
            self.gradient_norms.append(grad_norm)
            
            obj_value = self.compute_global_objective(self.x)
            self.objective_values.append(obj_value)
            
            if verbose and (iteration % 100 == 0 or iteration == n_iterations - 1):
                x_norm = np.linalg.norm(self.x)
                print(f"  Iter {iteration:4d}: ||x|| = {x_norm:8.4f}, "
                      f"grad_norm = {grad_norm:8.6f}, obj = {obj_value:10.6f}")
        
        if verbose:
            x_norm = np.linalg.norm(self.x)
            print(f"  Final ||x||: {x_norm:.6f}")
            print(f"  Final objective: {obj_value:.6f}")
            print(f"  Final gradient norm: {grad_norm:.8f}")
        
        return self.gradient_norms, self.objective_values