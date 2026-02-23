# attacks.py
"""Byzantine attack implementations for federated learning"""

import numpy as np


class BitFlippingAttack:
    """Bit flipping attack: reverse gradient direction"""
    
    def __init__(self):
        self.name = "Bit Flipping"
    
    def attack(self, honest_gradient, honest_gradients=None, iteration=0):
        return -honest_gradient


class MimicAttack:
    """Mimic attack: behave honestly initially, then deviate strategically"""
    
    def __init__(self, warmup_rounds=50):
        self.warmup_rounds = warmup_rounds
        self.name = "Mimic"
        self.round_count = 0
    
    def attack(self, honest_gradient, honest_gradients=None, iteration=0):
        self.round_count = iteration
        
        if honest_gradients is None:
            honest_gradients = [honest_gradient]
        
        if self.round_count < self.warmup_rounds:
            return np.mean(honest_gradients)
        else:
            honest_avg = np.mean(honest_gradients)
            return -2 * honest_avg


class ALIEAttack:
    """ALIE attack: v_i = mean - z * std"""

    def __init__(self, n_total=20, f_byzantine=3, z=1.0):
        self.n_total = n_total
        self.f_byzantine = f_byzantine
        self.z = z
        self.name = "ALIE"

    def attack(self, honest_gradient, honest_gradients=None, iteration=0):
        if honest_gradients is None:
            honest_gradients = [honest_gradient]

        stacked = np.array(honest_gradients)
        mu = np.mean(stacked, axis=0)
        std = np.std(stacked, axis=0)
        malicious_gradient = mu - self.z * std

        return malicious_gradient


def create_attackers(attack_type, n_byzantine=3, n_total=20):
    """Create a list of attacker instances based on attack type"""
    attackers = []
    
    for i in range(n_byzantine):
        if attack_type == "bf" or attack_type == "bit_flipping":
            attackers.append(BitFlippingAttack())
        elif attack_type == "mimic":
            attackers.append(MimicAttack(warmup_rounds=50))
        elif attack_type == "alie":
            attackers.append(ALIEAttack(n_total=n_total, f_byzantine=n_byzantine))
        else:
            raise ValueError(f"Unknown attack type: {attack_type}")
    
    return attackers