# Gradient Flow Analysis - Developer Documentation

## Table of Contents

1. [Quick Start](#quick-start)
2. [Core Concepts](#core-concepts)
3. [Analyzer Reference](#analyzer-reference)
4. [Configuration Guide](#configuration-guide)
5. [Network-Specific Recommendations](#network-specific-recommendations)
6. [Advanced Features](#advanced-features)
7. [Performance Tuning](#performance-tuning)
8. [Troubleshooting](#troubleshooting)

---

## Quick Start

```python
from gradient_flow import GradientFlowAnalyzer

# Create analyzer (uses scientifically validated defaults)
analyzer = GradientFlowAnalyzer(model)

# Analyze gradient flow
issues = analyzer.analyze(
    input_fn=lambda: torch.randn(32, 784),
    loss_fn=lambda output: F.cross_entropy(output, targets),
    steps=20
)

# Review results
analyzer.print_summary(issues)
for issue in issues:
    print(issue)
```

**That's it!** The analyzer automatically detects:
- Vanishing gradients
- Exploding gradients
- Dead neurons
- Gradient instability
- Numerical issues

---

## Core Concepts

### What is Gradient Flow Analysis?

Gradient flow analysis monitors how gradients propagate backward through your neural network during training. Poor gradient flow prevents learning and causes:

- **Vanishing gradients**: Gradients become too small, early layers don't learn
- **Exploding gradients**: Gradients become too large, training becomes unstable
- **Dead neurons**: Layers output constant zero, gradients are blocked
- **Saturation**: Activations saturate, killing gradients

### How It Works

The analyzer performs three types of analysis:

1. **Scalar Analysis** (Always enabled, 80% detection rate)
   - Gradient magnitude (L2 norm)
   - Temporal variance
   - Change rate
   - Zero gradient ratio

2. **Vector Analysis**
   - Divergence (expansion/contraction)
   - Curl (rotation/circulation)
   - Flow coherence

3. **Jacobian Analysis**
   - Spectral radius
   - Eigenvalue analysis
   - Conditioning

---

## Analyzer Reference

### 1. GradientFlowAnalyzer

**Use for**: All network types (default choice)

```python
from gradient_flow import GradientFlowAnalyzer

analyzer = GradientFlowAnalyzer(
    model,
    enable_rnn_analyzer=False,  # Default: False (not needed for most cases)
    enable_circular_flow_analyser=False,  # Default: False (no detection benefit)
    vanishing_threshold=1e-6,  # Gradient magnitude threshold
    exploding_threshold=10.0,  # Maximum safe gradient magnitude
    dead_threshold=0.9,  # Ratio of zero gradients
    unstable_cv_threshold=3.0  # Coefficient of variation limit
)

issues = analyzer.analyze(input_fn, loss_fn, steps=20)
```

**Detects**:
- VANISHING: Gradients < vanishing_threshold
- EXPLODING: Gradients > exploding_threshold
- DEAD: >90% of gradients are zero
- UNSTABLE: Coefficient of variation > 3.0
- NUMERICAL: NaN or Inf values

**When to use**:
- ✓ Feedforward networks (MLPs, CNNs)
- ✓ ResNets, DenseNets
- ✓ NLP models (transformers, embeddings)
- ✓ Any network type (general purpose)

**When NOT to use**:
- Use RecurrentAnalyzer for RNNs/LSTMs (adds spectral analysis)
- Use RLPolicyAnalyzer for RL policies (adds entropy analysis)
- Use TransformerAnalyzer for attention (adds head diversity)

---

### 2. RecurrentAnalyzer

**Use for**: RNNs, LSTMs, GRUs, ESNs

```python
from gradient_flow import RecurrentAnalyzer

analyzer = RecurrentAnalyzer(
    model,
    enable_jacobian=True,         # ESSENTIAL for spectral radius
    enable_vector_analysis=False, # Not beneficial for RNNs
    optimal_radius=0.9,           # Target spectral radius
    critical_band=0.15,           # Acceptable deviation
    track_hidden_states=True      # Capture temporal dynamics
)

issues, dynamics = analyzer.analyze(input_fn, loss_fn, steps=20)
analyzer.print_summary(issues, dynamics)
```

**Detects** (in addition to base issues):
- Spectral radius too low (fast forgetting)
- Spectral radius too high (exploding states)
- Low effective rank (insufficient capacity)
- Short effective memory (<10 timesteps)

**Provides**:
```python
dynamics.spectral_radius          # Eigenvalue magnitude
dynamics.criticality              # SUBCRITICAL/CRITICAL/SUPERCRITICAL
dynamics.effective_memory_length  # Timesteps until gradient decays
dynamics.temporal_decay_rate      # Per-step gradient decay
```

**Why Jacobian is NEEDED for RNNs**:
- Detects gradient flow through time
- Measures stability via eigenvalues
- Calculates effective memory length
- NOT detectable with scalar metrics alone

---

### 3. RLPolicyAnalyzer

**Use for**: Actor-Critic, PPO, A2C, SAC

```python
from gradient_flow import RLPolicyAnalyzer

analyzer = RLPolicyAnalyzer(
    model,
    enable_jacobian=False,        # Not needed for policies
    enable_vector_analysis=False, # Not beneficial
    min_entropy_threshold=0.5,    # Minimum exploration level
    min_std_threshold=0.1,        # Continuous action std min
    max_std_threshold=2.0,        # Continuous action std max
    track_distributions=True      # Capture action distributions
)

# Requires action_extractor to analyze distributions
def action_extractor(output):
    # Extract (logits, std, mean, value) from your policy output
    return logits, std, mean, value

issues, stats = analyzer.analyze(
    input_fn, loss_fn, steps=20,
    action_extractor=action_extractor
)
```

**Detects**:
- HIGH entropy collapse (agent stopped exploring)
- MEDIUM entropy collapse (exploration declining)
- Insufficient exploration (too deterministic)
- Excessive exploration (too random)
- Low continuous action std

**Provides**:
```python
stats.discrete_entropy_normalized  # 0-1, higher = more exploration
stats.continuous_std_mean          # Action space coverage
stats.entropy_collapse_risk        # LOW/MEDIUM/HIGH
stats.exploration_adequacy         # TOO_LOW/ADEQUATE/TOO_HIGH
```

---

### 4. TransformerAnalyzer

**Use for**: Transformer models with attention mechanisms

```python
from gradient_flow import TransformerAnalyzer

analyzer = TransformerAnalyzer(
    model,
    enable_jacobian=False,        # Not needed
    enable_vector_analysis=False, # Not beneficial
    min_entropy_threshold=0.5,    # Attention entropy minimum
    min_diversity_threshold=0.3,  # Head diversity minimum
    track_attention_weights=True  # Capture attention patterns
)

issues, attn_stats = analyzer.analyze(input_fn, loss_fn, steps=20)
```

**Detects**:
- Low attention entropy (attending to few positions)
- High rank collapse risk (sparse attention)
- Low head diversity (redundant heads)

**Provides**:
```python
attn_stats.attention_entropy_mean    # Average entropy across heads
attn_stats.attention_sparsity        # % of near-zero attention weights
attn_stats.head_diversity            # Variance across heads
attn_stats.rank_collapse_risk        # LOW/MEDIUM/HIGH
```

---

## Configuration Guide

### Scientifically Validated Defaults

Based on empirical testing (see `validation/VALIDATION_RESULTS.md`):

```python
# ✓ RECOMMENDED (80% detection, 26x faster)
GradientFlowAnalyzer(model, enable_jacobian=False, enable_vector_analysis=False)

# ✗ NOT RECOMMENDED (same detection, much slower)
GradientFlowAnalyzer(model, enable_jacobian=True, enable_vector_analysis=True)
```

### When to Enable Jacobian Analysis

**Enable ONLY for:**
1. **RNNs/LSTMs** - Spectral radius is essential
2. **Research/debugging** - Full diagnostic information
3. **Gradient subspace analysis** - Detecting rank collapse

**Do NOT enable for:**
- Feedforward networks (no benefit)
- Standard pathology detection (scalar metrics sufficient)
- Production monitoring (too expensive)

### When to Enable Vector Analysis

**Enable ONLY for:**
1. **Detecting circular gradients** - Adversarial training, GANs
2. **Research on gradient dynamics** - Flow patterns, vorticity
3. **Diagnosing oscillating training** - Curl indicates rotation

**Do NOT enable for:**
- Standard gradient issues (no benefit)
- Temporal sequences (not applicable)
- Most use cases (adds cost, zero detection value)

---

## Network-Specific Recommendations

### Feedforward Networks (MLPs, CNNs)

```python
analyzer = GradientFlowAnalyzer(
    model,
    enable_jacobian=False,       # Not needed
    enable_vector_analysis=False # Not beneficial
)
```

**Rationale**: Scalar metrics detect all standard pathologies (vanishing, exploding, dead). Validation shows 80% detection with minimal cost.

---

### Recurrent Networks (RNNs, LSTMs, GRUs)

```python
# Option 1: General analysis with Jacobian
analyzer = GradientFlowAnalyzer(
    model,
    enable_jacobian=True,        # ESSENTIAL for temporal analysis
    enable_vector_analysis=False
)

# Option 2: RNN-specific analyzer (RECOMMENDED)
analyzer = RecurrentAnalyzer(
    model,
    enable_jacobian=True,        # Enabled by default
    optimal_radius=0.9
)
```

**Rationale**: Jacobian eigenvalues reveal spectral radius (stability), effective memory length, and temporal decay patterns. RecurrentAnalyzer adds RNN-specific diagnostics.

---

### Transformers

```python
# Option 1: General analysis
analyzer = GradientFlowAnalyzer(model)  # Uses defaults

# Option 2: Attention-specific analyzer (RECOMMENDED)
analyzer = TransformerAnalyzer(
    model,
    min_entropy_threshold=0.5,
    min_diversity_threshold=0.3
)
```

**Rationale**: Transformers benefit from attention pattern analysis (entropy, head diversity, rank collapse). Use TransformerAnalyzer for attention-specific diagnostics.

---

### RL Policy Networks

```python
analyzer = RLPolicyAnalyzer(
    model,
    min_entropy_threshold=0.5,   # Adjust based on task
    min_std_threshold=0.1,
    track_distributions=True
)
```

**Rationale**: RL policies require distribution analysis (entropy, exploration). Use RLPolicyAnalyzer to monitor policy collapse and exploration adequacy.

---

### ResNets / Skip Connections

```python
analyzer = GradientFlowAnalyzer(model)  # Standard defaults work well
```

**Rationale**: Skip connections stabilize gradients. Standard scalar analysis is sufficient. Complex methods add no value.

---

### GANs / Adversarial Training

```python
analyzer = GradientFlowAnalyzer(
    model,
    enable_jacobian=False,
    enable_vector_analysis=True  # Consider for detecting circular gradients
)
```

**Rationale**: Adversarial training can produce oscillating/circular gradient patterns. Vector curl analysis MAY help (not empirically validated).

---

## Advanced Features

### Custom Thresholds

```python
analyzer = GradientFlowAnalyzer(
    model,
    vanishing_threshold=1e-7,    # Stricter (catches more issues)
    exploding_threshold=5.0,     # Stricter (lower tolerance)
    dead_threshold=0.95,         # More lenient (99% zero required)
    unstable_cv_threshold=2.0    # Stricter (lower variance allowed)
)
```

### Analyzing Specific Layers

```python
issues = analyzer.analyze(input_fn, loss_fn, steps=20)

# Filter by severity
critical = [i for i in issues if i.severity == IssueSeverity.CRITICAL]

# Filter by type
vanishing = [i for i in issues if i.issue_type == IssueType.VANISHING]

# Filter by layer name
encoder_issues = [i for i in issues if 'encoder' in i.layer.lower()]
```

### Progressive Analysis

```python
# Quick check (5 steps)
issues = analyzer.analyze(input_fn, loss_fn, steps=5)
if len(issues) == 0:
    return  # Model is healthy

# Detailed analysis (50 steps)
issues = analyzer.analyze(input_fn, loss_fn, steps=50)
```

---

## Performance Tuning

### Analysis Speed

| Configuration | Avg Time | Detection | Use Case |
|--------------|----------|-----------|----------|
| Scalar only  | 0.009s   | 80%       | Production, fast checks |
| Scalar + Jacobian | 0.009s | 80% | RNN analysis |
| Scalar + Vector   | 0.031s | 80% | Research (no benefit) |
| Full analysis     | 0.025s | 80% | Debugging (no benefit) |

### Memory Usage

- **Scalar only**: ~10MB per layer
- **+ Jacobian**: +50MB per layer (stores eigenvectors)
- **+ Vector**: +20MB per layer (stores flow fields)

### Recommended Settings by Environment

**Production monitoring**:
```python
GradientFlowAnalyzer(model)  # Defaults: fast, sufficient
```

**Development debugging**:
```python
RecurrentAnalyzer(model)  # If RNN
TransformerAnalyzer(model)  # If transformer
RLPolicyAnalyzer(model)  # If RL policy
GradientFlowAnalyzer(model)  # Otherwise
```

**Research**:
```python
GradientFlowAnalyzer(
    model,
    enable_jacobian=True,        # Full diagnostics
    enable_vector_analysis=True
)
```

---

## Troubleshooting

### "No issues detected but training fails"

1. **Increase analysis steps**: `steps=50` instead of `steps=20`
2. **Lower thresholds**: Make detection more sensitive
3. **Check later in training**: Gradients may degrade over time
4. **Analyze with optimizer**: Pass `optimizer` parameter

### "Too many false positives"

1. **Raise thresholds**: `vanishing_threshold=1e-5` instead of `1e-6`
2. **Reduce analysis steps**: Early training has noisy gradients
3. **Filter by severity**: Focus on CRITICAL/HIGH only

### "Analysis is too slow"

1. **Disable complex methods**: Use defaults (both False)
2. **Reduce steps**: `steps=5` for quick checks
3. **Analyze subset of batches**: Don't analyze every step

### "Bottleneck not detected"

Known issue: Current thresholds miss some bottlenecks (0% detection in validation). This is a threshold tuning problem, not a method problem. Workaround: Manually check compression ratios.

---

## See Also

- **Validation Results**: `gradient_flow/validation/VALIDATION_RESULTS.md`
- **Scientific Summary**: `gradient_flow/SCIENTIFIC_VALIDATION_SUMMARY.md`
- **Examples**: `gradient_flow/examples/`
- **API Reference**: `docs/API.md`
- **Architecture Guide**: `docs/ARCHITECTURE.md`
