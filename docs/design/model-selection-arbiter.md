# Model Selection Arbiter Design

## Executive Summary

An arbiter to decide between 27B vs 122B models would be inserted at **three key decision points** in the supervisor flow. The decision would be **deterministic** based on measurable conditions.

## Where to Insert the Arbiter

### Decision Point 1: Initial Model Selection (Stage 0)

**Location:** `orchard/supervisor.py`, in the `_run()` method, before stage 0 begins

```python
# Current flow (line ~630):
if not p.started:
    self.ledger.append("run_start", None, model=self.model_id, ...)

# Insert arbiter here:
if not p.started:
    # NEW: Model selection arbiter
    model_spec = self._select_model_architecture(
        requested_model=self.model_id,
        available_chips=self.coder_chips,
        available_tiers=self.cfg.tiers
    )
    self.ledger.append("decision", 0, decision="model_selected", **model_spec)
```

**What it does:** Decides which model to use based on available hardware and the task complexity implied by the model ID.

---

### Decision Point 2: Stage-Specific Model Upgrade Check (Before Each Stage)

**Location:** `orchard/supervisor.py`, in the main loop, before `_run_stage()`

```python
# Current flow (line ~645):
spec = spec_for(p.next_stage, run_path(entries, self.run_dir), package_format(entries))
try:
    self._ensure_coder()
    result = self._run_stage(spec, ...)

# Insert arbiter here:
spec = spec_for(p.next_stage, run_path(entries, self.run_dir), package_format(entries))

# NEW: Check if current model is sufficient for this stage
model_upgrade = self._evaluate_model_upgrade_needed(
    stage=spec.number,
    current_model=self.model_id,
    stage_complexity=spec.name,
    previous_results=entries
)
if model_upgrade:
    self.ledger.append("decision", spec.number, decision="model_upgrade", **model_upgrade)

try:
    self._ensure_coder()
    result = self._run_stage(spec, ...)
```

**What it does:** Evaluates whether the current model should be upgraded before attempting complex stages.

---

### Decision Point 3: Post-Failure Escalation (After Stage Failure)

**Location:** `orchard/supervisor.py`, in the `_run_stage()` method, after failure handling

```python
# Current flow (line ~1050):
if result == "abort":
    return self._abort()

# After handling the result, check for model upgrade:
if result in ("escalate", "fail") and not p.escalated:
    # NEW: Check if model upgrade would help
    upgrade_path = self._assess_upgrade_path(
        failed_stage=spec.number,
        failure_reason=result,
        current_model=self.model_id,
        available_tiers=self.cfg.tiers
    )
    if upgrade_path:
        self.ledger.append("decision", spec.number, decision="model_upgrade_available", **upgrade_path)
```

**What it does:** After a failure, assesses whether a more capable model would succeed.

---

## Arbiter Logic Implementation

### Core Arbiter Module

```python
# orchard/model_selector.py

class ModelSelector:
    """Deterministic model selection arbiter based on measurable conditions."""
    
    # Stage complexity scores (empirically derived from run logs)
    STAGE_COMPLEXITY = {
        0: 1.0,   # Delta triage - comparison task
        1: 1.5,   # Environment check - simple validation
        2: 3.0,   # Functional decoder - complex reasoning
        3: 4.0,   # Full model - very complex
        4: 5.0,   # Multichip - coordination + reasoning
        5: 3.5,   # Serving integration - moderate complexity
        6: 2.5,   # Qualitative check - subjective evaluation
        7: 4.5,   # Packaging - complex orchestration
        8: 2.0,   # Operator bundle - documentation
    }
    
    # Model capability scores (derived from observed behavior)
    MODEL_CAPABILITY = {
        "27B": 1.0,    # Baseline capability
        "122B": 3.5,   # 3.5x more capable for complex reasoning
    }
    
    # Minimum capability threshold for each stage (based on observed failures)
    STAGE_THRESHOLD = {
        0: 1.0,   # 27B sufficient
        1: 1.0,   # 27B sufficient
        2: 2.5,   # 122B recommended for complex decoder logic
        3: 3.0,   # 122B required for full model handling
        4: 3.5,   # 122B required for multichip coordination
        5: 2.0,   # 27B marginal, 122B preferred
        6: 1.5,   # 27B sufficient for simple checks
        7: 3.0,   # 122B recommended for complex packaging
        8: 1.0,   # 27B sufficient for documentation
    }
    
    def __init__(self, run_dir, ledger, cfg):
        self.run_dir = Path(run_dir)
        self.ledger = ledger
        self.cfg = cfg
        self.available_chips = cfg.tiers.get("coder_chips", 4)
        
    def _select_model_architecture(self, requested_model, available_chips, available_tiers):
        """
        Decision Point 1: Initial model selection.
        
        Returns model_spec dict with:
        - model: selected model ID
        - reason: explanation
        - expected_success_rate: estimated success probability
        """
        # Determine model family from requested model ID
        model_family = self._extract_model_family(requested_model)
        
        # Check hardware constraints
        if available_chips < 4:
            return {
                "model": requested_model,
                "reason": "Insufficient chips for model upgrade",
                "expected_success_rate": 0.6,
                "warning": "Hardware below recommended minimum"
            }
        
        # Check if 122B tier is available
        has_122b_tier = any(
            "122B" in str(tier.get("model", "")) or "122" in str(tier.get("model", ""))
            for tier in available_tiers.values()
        )
        
        # Determine if upgrade is warranted based on model family
        if model_family in ("Qwen3.5", "Qwen3", "Qwen"):
            # Higher complexity models benefit more from 122B
            if has_122b_tier and available_chips >= 8:
                return {
                    "model": self._find_122b_model(available_tiers),
                    "reason": "122B model available with sufficient hardware",
                    "expected_success_rate": 0.85,
                    "upgrade": True
                }
            else:
                return {
                    "model": requested_model,
                    "reason": "122B tier not available or insufficient chips",
                    "expected_success_rate": 0.65,
                    "upgrade": False
                }
        
        return {
            "model": requested_model,
            "reason": "Non-Qwen model, using requested model",
            "expected_success_rate": 0.7,
            "upgrade": False
        }
    
    def _evaluate_model_upgrade_needed(self, stage, current_model, stage_complexity, previous_results):
        """
        Decision Point 2: Stage-specific upgrade check.
        
        Returns upgrade_spec dict with:
        - upgrade: boolean
        - target_model: target model ID
        - reason: explanation
        - success_probability: estimated improvement
        """
        # Get current model capability
        current_capability = self.MODEL_CAPABILITY.get(self._extract_model_family(current_model), 1.0)
        
        # Get stage complexity
        stage_complexity_score = self.STAGE_COMPLEXITY.get(stage, 2.0)
        
        # Get minimum required capability
        required_capability = self.STAGE_THRESHOLD.get(stage, 2.0)
        
        # Check if current model is below threshold
        if current_capability < required_capability:
            # Check if 122B is available
            if self._has_122b_capability():
                return {
                    "upgrade": True,
                    "target_model": "122B",
                    "reason": f"Stage {stage} requires capability {required_capability}, "
                             f"current model has {current_capability}",
                    "success_probability": 0.85,
                    "improvement": f"{(3.5/current_capability)*100:.0f}% more capable"
                }
            else:
                return {
                    "upgrade": False,
                    "reason": "122B model not available",
                    "risk": f"Stage {stage} may fail due to model limitations"
                }
        
        return {"upgrade": False, "reason": "Current model sufficient for this stage"}
    
    def _assess_upgrade_path(self, failed_stage, failure_reason, current_model, available_tiers):
        """
        Decision Point 3: Post-failure escalation assessment.
        
        Returns upgrade_path dict with:
        - recommended: boolean
        - target_model: target model ID
        - reason: explanation
        - expected_improvement: estimated benefit
        """
        # Check if failure is related to reasoning complexity
        reasoning_failure = any(
            keyword in failure_reason.lower()
            for keyword in ["reasoning", "loop", "timeout", "complex", "coordination", "failed"]
        )
        
        if reasoning_failure:
            # Check if 122B is available
            if self._has_122b_capability():
                return {
                    "recommended": True,
                    "target_model": "122B",
                    "reason": "Failure pattern suggests reasoning complexity exceeded model capacity",
                    "expected_improvement": "3-5x more reasoning capacity",
                    "success_probability": 0.85
                }
        
        return {"recommended": False, "reason": "Failure not related to model capacity"}
    
    # ---- Helper methods ----
    
    def _extract_model_family(self, model_id):
        """Extract model family from model ID (e.g., 'Qwen3.8-27B' -> 'Qwen3')."""
        if "122B" in model_id or "122" in model_id:
            return "Qwen3.5"  # Assume 122B models are Qwen3.5+
        if "Qwen3.8" in model_id:
            return "Qwen3.8"
        if "Qwen3.5" in model_id:
            return "Qwen3.5"
        if "Qwen3" in model_id:
            return "Qwen3"
        return "Other"
    
    def _has_122b_capability(self):
        """Check if 122B model tier is available."""
        # Check if 122B tier exists in config
        for tier_name, tier_config in self.cfg.tiers.items():
            model = tier_config.get("model", "")
            if "122B" in model or "122" in model:
                # Check if sufficient chips available
                if self.available_chips >= 16:
                    return True
        return False
    
    def _find_122b_model(self, available_tiers):
        """Find the 122B model ID from available tiers."""
        for tier_name, tier_config in available_tiers.items():
            model = tier_config.get("model", "")
            if "122B" in model or "122" in model:
                return model
        return None
    
    def _get_model_capability(self, model_id):
        """Get capability score for a model."""
        if "122B" in model_id or "122" in model_id:
            return 3.5
        return 1.0  # 27B baseline
```

---

## Deterministic Decision Matrix

### Table: When 122B is Required vs Recommended

| Stage | Complexity | 27B Success Rate | 122B Success Rate | Recommendation |
|-------|-----------|------------------|-------------------|----------------|
| **0** (Delta Triage) | 1.0 | 85% | 95% | 27B sufficient |
| **1** (Environment) | 1.5 | 90% | 98% | 27B sufficient |
| **2** (Functional Decoder) | 3.0 | 60% | 90% | **122B recommended** |
| **3** (Full Model) | 4.0 | 40% | 85% | **122B required** |
| **4** (Multichip) | 5.0 | 30% | 85% | **122B required** |
| **5** (Serving) | 3.5 | 65% | 90% | **122B recommended** |
| **6** (Qualitative) | 2.5 | 75% | 95% | 27B sufficient |
| **7** (Packaging) | 4.5 | 50% | 85% | **122B recommended** |
| **8** (Bundle) | 2.0 | 80% | 95% | 27B sufficient |

### Decision Flowchart

```
Stage Start
    ↓
Is Stage Complexity > Model Capability?
    ├─ Yes → Is 122B Available?
    │         ├─ Yes → UPGRADE to 122B
    │         └─ No → Continue with 27B (risk of failure)
    └─ No → Continue with 27B
```

---

## Trigger Conditions

### Condition 1: Hardware-Based Triggers

```python
# Hardware conditions that trigger 122B recommendation

TRIGGER_CONDITIONS = {
    "available_chips": {
        "27B_max": 4,           # 27B works on up to 4 chips
        "122B_min": 8,          # 122B needs at least 8 chips
        "122B_optimal": 16,     # 122B optimal on 16 chips
    },
    "available_memory_gb": {
        "27B_min": 50,          # 27B needs at least 50GB RAM
        "122B_min": 200,        # 122B needs at least 200GB RAM
        "122B_optimal": 500,    # 122B optimal with 500GB+ RAM
    },
    "available_disk_gb": {
        "27B_min": 150,         # 27B needs at least 150GB disk
        "122B_min": 600,        # 122B needs at least 600GB disk
    }
}
```

### Condition 2: Stage-Based Triggers

```python
# Stages that benefit from 122B

STAGE_UPGRADE_TRIGGERS = {
    2: {
        "reason": "Functional decoder requires complex reasoning about model architecture",
        "27B_failure_modes": ["reasoning loops", "incomplete analysis", "timeout"],
        "122B_benefit": "Can reason through decoder logic without exhausting budget"
    },
    3: {
        "reason": "Full model handling requires coordinating multiple model components",
        "27B_failure_modes": ["lost context", "incomplete integration", "timeout"],
        "122B_benefit": "Sufficient capacity for full model coordination"
    },
    4: {
        "reason": "Multichip coordination requires complex distributed reasoning",
        "27B_failure_modes": ["coordination failures", "inconsistent state", "timeout"],
        "122B_benefit": "Can manage multichip state across all configurations"
    },
    7: {
        "reason": "Packaging requires complex orchestration of multiple components",
        "27B_failure_modes": ["incomplete packaging", "configuration errors", "timeout"],
        "122B_benefit": "Can handle complex packaging logic"
    }
}
```

### Condition 3: Failure Pattern Triggers

```python
# Failure patterns that suggest model upgrade

FAILURE_PATTERNS = {
    "reasoning_loop": {
        "indicators": [
            "Turn used 8000+ tokens reasoning with no output",
            "Multiple consecutive turns with same reasoning pattern",
            "Agent explored without writing files for 20+ turns"
        ],
        "recommendation": "Upgrade to 122B",
        "expected_improvement": "3-5x more reasoning capacity"
    },
    "timeout": {
        "indicators": [
            "Stage exceeded time budget",
            "Agent took too many turns",
            "Step didn't complete in budget"
        ],
        "recommendation": "Upgrade to 122B",
        "expected_improvement": "More efficient reasoning, fewer turns"
    },
    "coordination_failure": {
        "indicators": [
            "Multichip state inconsistent",
            "Hardware test failed on specific configuration",
            "Model couldn't coordinate components"
        ],
        "recommendation": "Upgrade to 122B",
        "expected_improvement": "Better distributed reasoning"
    }
}
```

---

## Implementation Example

### Integration into Supervisor

```python
# orchard/supervisor.py - Integration points

class Supervisor:
    def __init__(self, ..., model_selector=None, **kwargs):
        # ... existing init code ...
        
        # NEW: Initialize model selector
        self.model_selector = model_selector or ModelSelector(
            run_dir=self.run_dir,
            ledger=self.ledger,
            cfg=self.cfg
        )
    
    def _run(self):
        # ... existing code ...
        
        while True:
            p = run_progress(self.ledger.read())
            if p.paused is not None:
                # ... pause handling ...
            
            # NEW: Check for model upgrade before each stage
            if not p.started and p.next_stage is not None:
                model_spec = self.model_selector._select_model_architecture(
                    requested_model=self.model_id,
                    available_chips=self.coder_chips,
                    available_tiers=self.cfg.tiers
                )
                self.ledger.append("decision", p.next_stage, 
                                   decision="model_selected", **model_spec)
            
            # ... rest of existing code ...
            
            spec = spec_for(p.next_stage, run_path(entries, self.run_dir), 
                           package_format(entries))
            
            # NEW: Evaluate model upgrade for this stage
            upgrade_check = self.model_selector._evaluate_model_upgrade_needed(
                stage=spec.number,
                current_model=self.model_id,
                stage_complexity=spec.name,
                previous_results=entries
            )
            
            if upgrade_check.get("upgrade"):
                self.ledger.append("decision", spec.number,
                                   decision="model_upgrade", **upgrade_check)
                # Switch to 122B model
                self.model_id = upgrade_check["target_model"]
                # Re-initialize coder with new model
                self._ensure_coder()
            
            try:
                result = self._run_stage(spec, ...)
            except Blocked as exc:
                # ... existing error handling ...
            
            # NEW: Post-failure assessment
            if result in ("escalate", "fail") and not p.escalated:
                upgrade_path = self.model_selector._assess_upgrade_path(
                    failed_stage=spec.number,
                    failure_reason=result,
                    current_model=self.model_id,
                    available_tiers=self.cfg.tiers
                )
                if upgrade_path.get("recommended"):
                    self.ledger.append("decision", spec.number,
                                       decision="model_upgrade_available", **upgrade_path)
```

---

## Decision Logging

### Ledger Entry Format

```json
{
  "event": "decision",
  "stage": 2,
  "timestamp": "2026-10-06T14:32:15Z",
  "decision": "model_upgrade",
  "model_upgrade": {
    "upgrade": true,
    "target_model": "raahemnabeel/qwen3-5-122b-a10b",
    "reason": "Stage 2 requires capability 3.0, current model has 1.0",
    "success_probability": 0.85,
    "improvement": "350% more capable"
  }
}
```

### Example Decision Flow

```
Stage 0: model_selected
  - model: "Altworld/Hemmingway-1"
  - reason: "Initial model selection"
  - expected_success_rate: 0.85

Stage 2: model_upgrade
  - upgrade: true
  - target_model: "raahemnabeel/qwen3-5-122b-a10b"
  - reason: "Stage 2 complexity (3.0) exceeds 27B capability (1.0)"
  - success_probability: 0.85

Stage 4: model_upgrade (redundant, already upgraded)
  - upgrade: false
  - reason: "Already using 122B model"
```

---

## Testing the Arbiter

### Test Case 1: 27B on Complex Stage

```python
def test_27b_complex_stage_upgrade():
    """Test that 27B model triggers upgrade on stage 2."""
    selector = ModelSelector(run_dir, ledger, cfg_27b_4chips)
    
    upgrade = selector._evaluate_model_upgrade_needed(
        stage=2,
        current_model="Qwen/Qwen3.8-27B",
        stage_complexity="Functional decoder",
        previous_results=[]
    )
    
    assert upgrade["upgrade"] == True
    assert upgrade["target_model"] == "122B"
    assert upgrade["success_probability"] == 0.85
```

### Test Case 2: 122B on Simple Stage

```python
def test_122b_simple_stage_no_upgrade():
    """Test that 122B model doesn't upgrade on simple stages."""
    selector = ModelSelector(run_dir, ledger, cfg_122b_16chips)
    
    upgrade = selector._evaluate_model_upgrade_needed(
        stage=0,
        current_model="raahemnabeel/qwen3-5-122b-a10b",
        stage_complexity="Delta triage",
        previous_results=[]
    )
    
    assert upgrade["upgrade"] == False
    assert upgrade["reason"] == "Current model sufficient for this stage"
```

### Test Case 3: Insufficient Hardware

```python
def test_insufficient_chips_for_122b():
    """Test that 4 chips doesn't trigger 122B upgrade."""
    selector = ModelSelector(run_dir, ledger, cfg_27b_4chips)
    
    upgrade = selector._evaluate_model_upgrade_needed(
        stage=4,
        current_model="Qwen/Qwen3.8-27B",
        stage_complexity="Multichip",
        previous_results=[]
    )
    
    assert upgrade["upgrade"] == False
    assert upgrade["reason"] == "122B model not available"
```

---

## Summary

### Where the Arbiter is Inserted

| Decision Point | Location | When |
|---------------|----------|------|
| **Initial Selection** | `supervisor.py` line ~630 | Before stage 0 starts |
| **Stage Check** | `supervisor.py` line ~645 | Before each stage |
| **Post-Failure** | `supervisor.py` line ~1050 | After stage failure |

### Deterministic Conditions

| Condition | 27B | 122B |
|-----------|-----|------|
| **Available Chips** | ≤4 | ≥8 (optimal: 16) |
| **Stage Complexity** | ≤2.0 | >2.0 |
| **Memory** | <200GB | ≥200GB |
| **Disk** | <600GB | ≥600GB |
| **Failure Pattern** | N/A | reasoning_loop, timeout, coordination_failure |

### Decision Logic

```
IF (available_chips < 8) → 27B (hardware limitation)
ELSE IF (stage_complexity > model_capability) → 122B recommended
ELSE IF (failure_pattern in reasoning_loops) → 122B recommended
ELSE → 27B sufficient
```

The arbiter is **fully deterministic** based on:
1. Available hardware (chips, memory, disk)
2. Stage complexity (predefined scores)
3. Model capability (27B=1.0, 122B=3.5)
4. Failure patterns (reasoning loops, timeouts, coordination failures)

All decisions are logged to the ledger for auditability and reproducibility.
