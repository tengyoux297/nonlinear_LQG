# Information Gain: Purpose and Approach Comparison

## What is Information Gain Used For?

**Information gain** is used to **select which sensors to activate** at each time step in the sensor selection algorithm. The goal is to choose sensors that will provide the **most reduction in state estimation uncertainty**.

### How It Works:
1. **At each sensor selection step** (every `update_interval` time steps):
   - The algorithm evaluates **all possible sensor subsets**
   - For each subset, it calculates the **information gain**
   - It selects the subset with the **maximum information gain**
   - Only the selected sensors are used for state estimation

2. **Information gain measures**: How much the **covariance (uncertainty) decreases** when using a particular sensor subset

3. **Higher information gain** = **Better state estimation** = **Lower uncertainty**

---

## Two Approaches Compared

### Approach 1: Heuristic (Previous/Logarithmic)
```python
# Old heuristic approach
info_gain = current_uncertainty * np.log(1 + num_active_sensors) / np.log(1 + total_sensors)
```

**How it works:**
- Only considers the **number of sensors** (not which sensors)
- Uses a **logarithmic scaling** to model diminishing returns
- **Ignores**:
  - Which specific sensors are selected
  - Sensor measurement matrices (C, M)
  - Sensor noise characteristics (V)
  - Actual covariance reduction

**Example:**
- With 10 total sensors:
  - Subset [0, 1, 2] → info_gain = uncertainty × log(4)/log(11) ≈ 0.36 × uncertainty
  - Subset [5, 6, 7] → info_gain = uncertainty × log(4)/log(11) ≈ 0.36 × uncertainty
  - **Same gain** even though sensors might be very different!

**Problems:**
- ❌ Cannot distinguish between good and bad sensors
- ❌ May select sensors that don't actually reduce uncertainty
- ❌ Doesn't account for sensor quality or measurement model
- ❌ May waste resources on redundant sensors

---

### Approach 2: Rigorous (Current/Covariance-Based)
```python
# New rigorous approach
P_post = self._compute_covariance_for_sensors(sensor_subset)  # Actual Kalman update
info_gain = trace(P_pred) - trace(P_post)  # Actual uncertainty reduction
```

**How it works:**
- Computes the **actual covariance matrix** after using the sensor subset
- Uses **Kalman filter equations** to predict uncertainty reduction
- **Accounts for**:
  - Specific sensors selected
  - Measurement matrices (C, M) for each sensor
  - Measurement noise (V) for each sensor
  - Current state and covariance
  - Actual information content

**Example:**
- With 10 total sensors:
  - Subset [0, 1, 2]: 
    - Computes actual P_post using sensors 0, 1, 2
    - info_gain = trace(P_pred) - trace(P_post) = **actual reduction**
  - Subset [5, 6, 7]:
    - Computes actual P_post using sensors 5, 6, 7
    - info_gain = trace(P_pred) - trace(P_post) = **actual reduction**
  - **Different gains** based on which sensors provide more information!

**Benefits:**
- ✅ Selects sensors that actually reduce uncertainty
- ✅ Accounts for sensor-specific properties
- ✅ Avoids redundant sensors
- ✅ Theoretically sound (based on information theory)
- ✅ Better estimation performance expected

---

## Expected Differences in Results

### 1. **Sensor Selection Quality**
- **Heuristic**: May select suboptimal sensors (e.g., redundant or noisy sensors)
- **Rigorous**: Selects sensors that maximize actual information gain

### 2. **Estimation Accuracy**
- **Heuristic**: Lower accuracy due to poor sensor selection
- **Rigorous**: Higher accuracy due to optimal sensor selection

### 3. **Covariance Trace (Uncertainty)**
- **Heuristic**: May not minimize uncertainty effectively
- **Rigorous**: Actively minimizes uncertainty through optimal selection

### 4. **Computational Cost**
- **Heuristic**: Very fast (just a logarithm calculation)
- **Rigorous**: Slower (requires matrix operations for each subset), but more accurate

### 5. **Robustness**
- **Heuristic**: May fail when sensors have different qualities
- **Rigorous**: Adapts to sensor characteristics automatically

---

## Visual Example

### Scenario: 3 sensors available, need to select 2

**Heuristic Approach:**
```
Subset [0, 1]: info_gain = uncertainty × log(3)/log(4) ≈ 0.79 × uncertainty
Subset [0, 2]: info_gain = uncertainty × log(3)/log(4) ≈ 0.79 × uncertainty  
Subset [1, 2]: info_gain = uncertainty × log(3)/log(4) ≈ 0.79 × uncertainty
→ All subsets have SAME gain! Random selection.
```

**Rigorous Approach:**
```
Subset [0, 1]: 
  - P_pred trace = 10.0
  - P_post trace = 3.5 (after using sensors 0, 1)
  - info_gain = 10.0 - 3.5 = 6.5

Subset [0, 2]:
  - P_pred trace = 10.0
  - P_post trace = 2.1 (after using sensors 0, 2)
  - info_gain = 10.0 - 2.1 = 7.9  ← HIGHER!

Subset [1, 2]:
  - P_pred trace = 10.0
  - P_post trace = 4.8 (after using sensors 1, 2)
  - info_gain = 10.0 - 4.8 = 5.2

→ Selects [0, 2] because it provides the most information!
```

---

## When to Use Each Approach

### Use Heuristic When:
- ⚡ Need very fast computation
- 📊 All sensors are similar quality
- 🎯 Rough approximation is acceptable
- 💻 Computational resources are limited

### Use Rigorous When:
- 🎯 Need optimal sensor selection
- 📈 Sensors have different qualities
- 🔬 Research/publication quality results
- ✅ Want theoretically sound approach
- 💰 Computational cost is acceptable

---

## Summary

**Information gain** is the **objective function** for sensor selection. The rigorous approach computes **actual uncertainty reduction** using Kalman filter theory, while the heuristic approach uses a **simple logarithmic approximation**. The rigorous approach should provide **better sensor selections** and **improved estimation performance**, at the cost of **higher computational complexity**.
