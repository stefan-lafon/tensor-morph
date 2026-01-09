# @title experimental/simulator.py
import numpy as np
import pandas as pd

"""
Roofline model comparing base latency vs fused latency.
- Memory-bound: Generally favors fusion, except it penalizes cases that become counter-productive by simulating register pressure.
- Compute-bound: Favors unfused ops by applying an orchestration tax for complex kernels in ALU-saturated environments.
"""

class HardwareScenario:
    def __init__(self, name, flops_per_sec, bytes_per_sec, overhead_ms=0.001):
        self.name = name
        self.flops_per_sec = flops_per_sec
        self.bytes_per_sec = bytes_per_sec
        self.overhead = overhead_ms

    def compute_conv_cost(self, h, w, ic, oc, k, is_dw=False):
        if is_dw:
            flops = h * w * ic * (k**2) * 2
            weight_bytes = (k**2) * ic * 4
        else:
            flops = h * w * oc * ic * (k**2) * 2
            weight_bytes = (k**2) * ic * oc * 4
        
        io_bytes = (h * w * ic * 4) + (h * w * oc * 4)
        total_bytes = weight_bytes + io_bytes
        
        t_compute = flops / self.flops_per_sec
        t_memory = total_bytes / self.bytes_per_sec
        
        return max(t_compute, t_memory) + self.overhead

    def compute_pointwise_cost(self, h, w, c):
        flops = h * w * c * 2
        total_bytes = (h * w * c * 4 * 3) # Read/Read/Write
        
        t_compute = flops / self.flops_per_sec
        t_memory = total_bytes / self.bytes_per_sec
        
        return max(t_compute, t_memory) + self.overhead

    def run_sim(self, f):
        h, w, ic, oc, k = f['in_h'], f['in_w'], f['in_c'], f['out_c'], f['kernel']
        is_dw, chain_len = f['is_dw'], f['chain_len']
        
        # Unfused: conv + N pointwise ops.
        t_conv = self.compute_conv_cost(h, w, ic, oc, k, is_dw)
        t_pointwise = chain_len * self.compute_pointwise_cost(h, w, oc)
        base_latency = t_conv + t_pointwise
        
        # Fused: absorb pointwise math, eliminate IO.
        extra_math = chain_len * (h * w * oc * 2)
        if is_dw:
            fused_flops = (h * w * ic * (k**2) * 2) + extra_math
            fused_w_bytes = (k**2) * ic * 4
        else:
            fused_flops = (h * w * oc * ic * (k**2) * 2) + extra_math
            fused_w_bytes = (k**2) * ic * oc * 4
            
        fused_io_bytes = (h * w * ic * 4) + (h * w * oc * 4) 
        t_fused_comp = fused_flops / self.flops_per_sec
        t_fused_mem = (fused_w_bytes + fused_io_bytes) / self.bytes_per_sec
        
        morphed_latency = max(t_fused_comp, t_fused_mem) + self.overhead

        # Hardware-specific penalties for veto logic.
        if "Memory_Bound" in self.name:
            # Register pressure scales with ic and chain depth.
            reg_pressure_factor = (ic * chain_len) / 1024.0
            penalty = 1.0 + (reg_pressure_factor ** 2)
            morphed_latency *= penalty
            
        elif "Compute_Bound" in self.name:
            # Rebalanced tax: 1.1 floor allows small kernels to profit.
            size_factor = (h * w) / (224 * 224)
            penalty = 1.1 + (chain_len * 0.4 * size_factor)
            morphed_latency *= penalty
        
        return base_latency, morphed_latency

def generate_balanced_workload(n=1000, scenario_type="Compute_Bound"):
    # Rebalancing distributions to ensure roughly 20% profitability.
    if scenario_type == "Compute_Bound":
        # Target kernels that are small enough to evade the orchestration tax.
        h_w = np.random.choice([32, 56, 112, 224], n, p=[0.4, 0.3, 0.2, 0.1])
        chains = np.random.randint(1, 4, n)
    else:
        h_w = np.random.choice([56, 112, 224], n)
        chains = np.random.randint(1, 6, n)

    data = {
        'in_h': h_w,
        'in_w': h_w,
        'in_c': np.random.choice([16, 32, 64, 128, 256, 512], n),
        'out_c': np.random.choice([16, 32, 64, 128, 256, 512], n),
        'kernel': np.random.choice([1, 3, 5], n),
        'is_dw': np.random.choice([0, 1], n),
        'chain_len': chains
    }
    return pd.DataFrame(data)