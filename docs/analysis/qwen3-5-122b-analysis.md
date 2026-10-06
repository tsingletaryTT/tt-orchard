# Using Qwen3.5-122B-A10B with tt-orchard

## Model Overview

**Model**: `raahemnabeel/qwen3-5-122b-a10b-blackholex4`
- **Architecture**: Qwen3.5 with 122B parameters
- **Special Feature**: A10B (Adaptive Block) - likely a MoE (Mixture of Experts) variant
- **Hardware Target**: Blackhole x4 chips (4-chip configuration)
- **Platform**: Tenstorrent Blackhole (p300c boards)

## What Would Change in tt-orchard

### 1. Memory Requirements

Based on the tt-orchard sizing framework, here's what changes with a 122B model:

| Component | Qwen3.8-27B (Current) | Qwen3.5-122B-A10B (Proposed) | Multiplier |
|-----------|----------------------|-------------------------------|------------|
| **Model Weights** | ~51 GB (4-bit) | ~230 GB (4-bit) | ~4.5x |
| **Full Precision** | ~216 GB (bf16) | ~976 GB (bf16) | ~4.5x |
| **Active Parameters** | ~3B (MoE) | ~10-15B (estimated MoE) | ~4-5x |
| **Context Window** | 256K tokens | 256K tokens (likely) | 1x |

**Key Memory Calculations:**

```
Theoretical Bandwidth (DDR5-5600, 2 channels): 89.6 GB/s
Theoretical Bandwidth (DDR5-3600, 2 channels): 57.6 GB/s

For 122B model at 4-bit (0.5 bytes/token):
- Decode ceiling = 89.6 / 0.5 = 179.2 tokens/s (theoretical max)
- Real-world expected = 30-60 tokens/s (with overhead)

For 122B model at bf16 (2 bytes/token):
- Decode ceiling = 89.6 / 2 = 44.8 tokens/s (theoretical max)
- Real-world expected = 8-15 tokens/s
```

### 2. Hardware Configuration Changes

**Current Setup (27B model):**
- 2 boards (4 chips total): Qwen3.8-27B on all 4 chips
- 1 board (2 chips): Qwen3.8-27B on 2 chips
- CPU: qwen3-coder:30b for stand-in

**For 122B Model:**
- **Minimum**: 4 chips (1 board) - may not fit
- **Recommended**: 8 chips (2 boards) for reasonable performance
- **Optimal**: 16 chips (4 boards) for full parallelism

The current Quietbox 2 only has 4 chips total. For a 122B model, you would need:
- **Hardware upgrade**: Additional Blackhole boards (2-4 more boards)
- **Or**: Accept significantly slower inference on the existing 4 chips

### 3. Configuration File Changes

#### tiers.toml Updates

```toml
[tiers.large]
role = "plan and diagnose for 122B model"
endpoint = "http://127.0.0.1:8000/v1"
model = "raahemnabeel/qwen3-5-122b-a10b"
placement = "chips"
context_tokens = 256000  # or higher if model supports it

[tiers.small]
role = "routine steps while one board free"
endpoint = "http://127.0.0.1:8001/v1"
model = "raahemnabeel/qwen3-5-122b-a10b"
placement = "chips"
context_tokens = 256000

[tiers.cpu]
role = "stand-in while chips busy"
endpoint = "http://127.0.0.1:11434/v1"
model = "qwen3-coder:30b"  # Keep same CPU tier
placement = "cpu"
```

#### defaults.py Updates (if scaling up hardware)

```python
# For 122B model on larger hardware:
STAGE4_SWAP_DISK_GB = 440.0  # 4x the 110 GB for 27B model
STAGE_DISK_GB = {0: 1.0, 1: 5.0, 2: 160.0, 3: 160.0, 4: 320.0, 5: 160.0, 6: 160.0, 7: 320.0, 8: 1.0}
```

### 4. Performance Implications

| Metric | 27B Model (4 chips) | 122B Model (4 chips) | 122B Model (16 chips) |
|--------|-------------------|---------------------|---------------------|
| **Prefill Speed** | ~130-200 tok/s | ~30-50 tok/s | ~100-150 tok/s |
| **Decode Speed** | ~15-25 tok/s | ~3-8 tok/s | ~20-40 tok/s |
| **Time to First Token** | ~2-5 s | ~15-30 s | ~5-10 s |
| **Context Processing** | 130K in ~42s | 130K in ~180s | 130K in ~60s |

### 5. Agent Budget Adjustments

The current agent budgets would need adjustment:

```python
# Current (27B model)
AGENT_MAX_TOKENS = 16384  # 6.25% of 256K context
CANARY_MAX_TOKENS = 64

# For 122B model (slower inference):
AGENT_MAX_TOKENS = 8192   # 3.125% - smaller responses due to slower speed
CANARY_MAX_TOKENS = 32    # Smaller canary checks
AGENT_REQUEST_TIMEOUT_S = 1800.0  # Double to 30 min
CANARY_TIMEOUT_S = 600.0          # Double to 10 min
```

### 6. Disk Space Requirements

| Stage | 27B Model | 122B Model | Notes |
|-------|-----------|------------|-------|
| Model Weights | 51 GB | 230 GB | Download storage |
| 2-chip Tensor Cache | 34 GB | ~136 GB | Scales ~4x |
| 4-chip Tensor Cache | 31 GB | ~124 GB | Scales ~4x |
| Stage 4 Total | 110 GB | 440 GB | All configurations |
| Stage 7 Package | 80 GB | 320 GB | Package + boot check |

### 7. Time Budget Changes

| Stage | 27B Model Budget | 122B Model Budget | Reason |
|-------|-----------------|-------------------|--------|
| Stage 2 (Decoder) | 14,400s (4h) | 28,800s (8h) | Slower inference |
| Stage 4 (Multichip) | 28,800s (8h) | 57,600s (16h) | More configurations |
| Stage 7 (Package) | 14,400s (4h) | 28,800s (8h) | Larger package build |

### 8. Sizing Tool Adjustments

The `orchard/sizing.py` would need these updates:

```python
# For 122B model characterization:
def measure_122b_model(host, model, prompt):
    """Measure a 122B model with adjusted timeouts."""
    return measure(
        host=host,
        model=model,
        prompt=prompt,
        num_predict=256,  # Larger prompt for accurate measurement
        timeout=1800       # 30 min timeout instead of 15 min
    )
```

### 9. Critical Bottlenecks

**With 122B on 4 chips:**
1. **Memory Bandwidth Saturation**: 122B at 4-bit requires ~60 GB/s for decent performance
2. **Decode Speed**: Would drop to ~3-8 tokens/second
3. **Agent Turn Time**: A 16K token response could take 2000-5000 seconds (33-83 minutes)

**With 122B on 16 chips:**
1. **Inter-chip Communication**: More chips = more communication overhead
2. **Synchronization**: 16 chips require more coordination
3. **Power/Thermal**: Significantly higher power draw

### 10. Recommended Configuration Changes

```python
# orchard/defaults.py for 122B model support:

# Timing budgets (scaled for slower inference)
STAGE_BUDGET_S = {
    0: 14400.0,   # Stage 0: 4h → 4h (same, different work)
    1: 28800.0,   # Stage 1: 4h → 8h
    2: 57600.0,   # Stage 2: 4h → 16h
    3: 86400.0,   # Stage 3: 6h → 24h
    4: 115200.0,  # Stage 4: 8h → 32h
    5: 43200.0,   # Stage 5: 3h → 12h
    6: 57600.0,   # Stage 6: 4h → 16h
    7: 57600.0,   # Stage 7: 4h → 16h
    8: 14400.0    # Stage 8: 1h → 4h
}

# Disk space (scaled ~4x)
STAGE_DISK_GB = {
    0: 1.0,
    1: 5.0,
    2: 160.0,     # 40 GB × 4
    3: 160.0,
    4: 320.0,     # 80 GB × 4
    5: 160.0,
    6: 160.0,
    7: 320.0,     # 80 GB × 4
    8: 1.0
}

STAGE4_SWAP_DISK_GB = 440.0  # 110 GB × 4
```

### 11. Hardware Recommendations

**Minimum viable setup for 122B:**
- 8 Blackhole chips (4 boards)
- 512 GB+ system RAM
- 2 TB+ NVMe storage (for weights + caches)

**Optimal setup for 122B:**
- 16 Blackhole chips (8 boards)
- 1 TB+ system RAM
- 4 TB+ NVMe storage
- Enhanced cooling

### 12. Impact on Supervisor Logic

The supervisor would need these adjustments:

```python
# orchard/supervisor.py
class Supervisor122B(Supervisor):
    """Supervisor configured for 122B model characteristics."""
    
    # Longer timeouts for slower inference
    AGENT_REQUEST_TIMEOUT_S = 1800.0  # 30 min
    CANARY_TIMEOUT_S = 600.0           # 10 min
    COLD_BOOT_BUDGET_S = 7200.0        # 2 hours (vs 45 min)
    
    # Smaller agent budgets (slower responses)
    AGENT_MAX_TOKENS = 8192            # 8K instead of 16K
    CANARY_MAX_TOKENS = 32             # Smaller checks
    
    # More disk space required
    DISK_FREE_THRESHOLD = 500.0         # 500 GB minimum
```

### 13. Risk Assessment

| Risk | 27B Model | 122B Model | Mitigation |
|------|-----------|------------|------------|
| **Timeout failures** | Low | High | Increase timeouts |
| **Memory exhaustion** | Low | High | Add more chips |
| **Thermal issues** | Low | Medium | Enhanced cooling |
| **Power draw** | Low | High | Verify PSU capacity |
| **Agent turn time** | Manageable | Very long | Reduce token budgets |

### 14. Cost-Benefit Analysis

**Advantages of 122B:**
- Significantly more capable reasoning
- Better at complex tasks
- Potentially better multilingual support

**Disadvantages:**
- 4-5x slower inference
- 4-5x more disk space
- 4-5x longer stage times
- Requires hardware upgrade
- Much higher power consumption

### 15. Migration Path

If upgrading from 27B to 122B:

1. **Assess hardware**: Verify chip count, RAM, storage
2. **Update defaults.py**: Scale all budgets and limits
3. **Update tiers.toml**: Configure new model endpoints
4. **Test with small prompts**: Validate inference speed
5. **Run sizing tool**: Get actual measurements
6. **Adjust timeouts**: Based on measured performance
7. **Update supervisor**: Apply 122B-specific configuration

### 16. Alternative Approaches

**Option 1: Keep 27B for agent tasks, use 122B for specific stages**
- Use 27B for stages 0, 1, 8 (planning, reference, bundle)
- Use 122B for stages 2-6 (hardware-intensive stages)

**Option 2: Quantize more aggressively**
- Use 2-bit or 3-bit quantization for 122B
- Reduces memory requirements but may impact quality

**Option 3: Hybrid approach**
- Use 122B for complex reasoning tasks only
- Use smaller models for routine agent steps

### 17. Final Recommendation

**For the tt-orchard workflow specifically:**

**Do NOT use 122B for the entire workflow.** The supervisor's design assumes agent turns complete in ~200 seconds. With a 122B model on 4 chips, a single agent turn could take 30-60 minutes, breaking the supervisor's assumptions about turn management, watchdog timing, and stage budgets.

**Recommended approach:**
1. Keep 27B model for agent coordination (stages 0, 1, 8)
2. Use 122B only for hardware-intensive stages (2-6) if needed
3. Or upgrade to 16+ chips for acceptable 122B performance
4. Adjust all timing budgets proportionally

---

## Summary Table: 27B vs 122B in tt-orchard

| Aspect | 27B (Current) | 122B (On 4 chips) | 122B (On 16 chips) |
|--------|--------------|------------------|-------------------|
| **Inference Speed** | 15-25 tok/s | 3-8 tok/s | 20-40 tok/s |
| **Agent Turn Time** | ~200s | ~1200-3000s | ~400-800s |
| **Disk Space** | 110 GB | 440 GB | 440 GB |
| **Stage 2 Budget** | 4h | 16h | 8h |
| **Hardware** | 4 chips | 4 chips | 16 chips |
| **Viability** | ✅ Excellent | ❌ Too slow | ⚠️ Needs upgrade |

**Conclusion**: The 122B model would require significant hardware upgrades (16+ chips) and configuration changes to work well with tt-orchard's supervisor architecture. On the existing 4-chip setup, it would be impractical due to extremely long agent turn times.
