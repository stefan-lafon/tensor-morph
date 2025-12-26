import numpy as np

class HardwarePersonality:
    def __init__(self, name, flops_per_sec, bytes_per_sec, overhead_ms=0.01):
        self.name = name
        self.flops_per_sec = flops_per_sec
        self.bytes_per_sec = bytes_per_sec
        self.overhead = overhead_ms

    def compute_conv_cost(self, h, w, ic, oc, k, is_dw=False):
        # Determine flops and weights based on conv type
        if is_dw:
            flops = h * w * ic * (k**2) * 2
            weight_bytes = (k**2) * ic * 4
        else:
            flops = h * w * oc * ic * (k**2) * 2
            weight_bytes = (k**2) * ic * oc * 4
        
        # IO is just input + output tensors
        io_bytes = (h * w * ic * 4) + (h * w * oc * 4)
        total_bytes = weight_bytes + io_bytes
        
        # Roofline logic: time is dominated by the slowest component
        t_compute = flops / self.flops_per_sec
        t_memory = total_bytes / self.bytes_per_sec
        
        return max(t_compute, t_memory) + self.overhead

    def compute_pointwise_cost(self, h, w, c):
        # Add/Mul etc are basically always bandwidth limited
        flops = h * w * c * 2
        total_bytes = (h * w * c * 4 * 3) # 2 in, 1 out
        
        t_compute = flops / self.flops_per_sec
        t_memory = total_bytes / self.bytes_per_sec
        
        return max(t_compute, t_memory) + self.overhead

    def run_sim(self, f):
        """
        Main entry point for calculating base vs fused latency.
        'f' is a dict matching our schema keys.
        """
        h, w, ic, oc, k = f['in_h'], f['in_w'], f['in_c'], f['out_c'], f['kernel']
        is_dw, chain_len = f['is_dw'], f['chain_len']
        
        # Unfused: conv + N pointwise ops
        t_conv = self.compute_conv_cost(h, w, ic, oc, k, is_dw)
        t_pointwise = chain_len * self.compute_pointwise_cost(h, w, oc)
        base_latency = t_conv + t_pointwise
        
        # Fused: Conv absorbs the pointwise math, eliminates the memory overhead
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
        
        return base_latency, morphed_latency