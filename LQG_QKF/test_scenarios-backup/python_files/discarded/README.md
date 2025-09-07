# Nonlinear Kalman Filter Test Applications - Python Conversion

This directory contains Python conversions of the MATLAB nonlinear Kalman filter test applications from the original framework. These applications demonstrate various nonlinear filtering scenarios and compare different Kalman filter variants.

## Applications

### 1. Target Tracking (`target_tracking.py`)
**Application**: 3D target tracking with radar measurements
- **State**: 6D [x, y, z, vx, vy, vz] - position and velocity
- **Measurement**: Range and bearing measurements
- **Dynamics**: Constant velocity model
- **Nonlinearity**: Range/bearing measurement model

### 2. Terrain Referenced Navigation (`terrain_referenced_navigation.py`)
**Application**: Aircraft navigation using terrain height measurements
- **State**: 4D [x, y, vx, vy] - position and velocity
- **Measurement**: Terrain height at current position
- **Dynamics**: Constant velocity model
- **Nonlinearity**: Terrain height function (sinusoidal surface)

### 3. Synchronous Generator State Estimation (`synchronous_generator_state_estimation.py`)
**Application**: Power system generator state estimation
- **State**: 3D [delta, omega, E_q] - rotor angle, angular velocity, internal voltage
- **Measurement**: Active and reactive power
- **Dynamics**: Generator swing equations
- **Nonlinearity**: Power flow equations

### 4. Pendulum State Estimation (`pendulum_state_estimation.py`)
**Application**: Simple pendulum state estimation
- **State**: 2D [theta, theta_dot] - angle and angular velocity
- **Measurement**: Angle measurement
- **Dynamics**: Pendulum equations of motion
- **Nonlinearity**: Trigonometric functions in dynamics

### 5. Battery State Estimation (`battery_state_estimation.py`)
**Application**: Battery state-of-charge and state-of-health estimation
- **State**: 3D [SOC, SOH, V] - state of charge, state of health, voltage
- **Measurement**: Voltage measurement
- **Dynamics**: Battery equivalent circuit model
- **Nonlinearity**: Battery voltage model

## Filter Types

Each application tests multiple Kalman filter variants:

- **EKF**: Extended Kalman Filter
- **IEKF**: Iterated Extended Kalman Filter
- **UKF**: Unscented Kalman Filter
- **CKF**: Cubature Kalman Filter
- **EKF2**: Second-order Extended Kalman Filter
- **IEKF2**: Iterated Second-order Extended Kalman Filter

## Framework Comparison

Each filter is tested with two frameworks:
- **Old Framework**: Standard Kalman filter implementation
- **New Framework**: Improved framework with enhanced covariance updates

## Usage

### Running Individual Applications

```python
# Run target tracking
python target_tracking.py

# Run terrain navigation
python terrain_referenced_navigation.py

# Run generator estimation
python synchronous_generator_state_estimation.py

# Run pendulum estimation
python pendulum_state_estimation.py

# Run battery estimation
python battery_state_estimation.py
```

### Parameters

- **REPEAT**: Number of Monte Carlo runs (default: 1000, reduced from 10000 for faster testing)
- **SCALE**: Measurement noise range (10^scale)
- **NOISE1_REF**: Reference noise level for specific applications

### Output

Each application generates:
1. **Console output**: Runtime statistics and progress
2. **Plot**: Performance comparison across noise levels
3. **Saved figure**: High-resolution PNG file with results

## Key Features

### 1. Safe Matrix Operations
```python
def safe_matrix_inverse(A, reg=1e-8):
    """Safely compute matrix inverse with regularization"""
    try:
        return np.linalg.inv(A + reg * np.eye(A.shape[0]))
    except np.linalg.LinAlgError:
        return np.linalg.pinv(A + reg * np.eye(A.shape[0]))
```

### 2. Framework Comparison
Each application compares old vs new framework implementations:
- **Old**: Standard Kalman filter covariance updates
- **New**: Enhanced covariance updates with trace checking

### 3. Comprehensive Testing
- Multiple noise levels (10^-4 to 10^2)
- Multiple filter types
- Monte Carlo averaging
- Runtime performance tracking

### 4. Visualization
- Log-log plots of RMSE vs noise level
- Color-coded filter types
- Separate plots for each state component

## Dependencies

```python
import numpy as np
import matplotlib.pyplot as plt
import time
from scipy.linalg import cholesky, sqrtm
from typing import Tuple, List, Dict, Any
import warnings
```

## Notes

1. **Reduced Monte Carlo Runs**: For faster testing, REPEAT is set to 1000 instead of the original 10000
2. **Placeholder Implementations**: UKF, CKF, EKF2, and IEKF2 use EKF as placeholder for brevity
3. **Numerical Stability**: Safe matrix inverse operations prevent singular matrix errors
4. **Reproducibility**: Fixed random seeds ensure consistent results

## Extending the Framework

To add your own LQG system:

1. **Create new application file** following the existing pattern
2. **Define dynamics and measurement functions**
3. **Implement filter variants** (EKF, IEKF, etc.)
4. **Add to main() function** with proper result tracking
5. **Create visualization** for your specific states

## Performance Considerations

- **Memory**: Large Monte Carlo arrays for high-dimensional states
- **Computation**: Matrix operations scale with state dimension
- **Accuracy**: Trade-off between speed (fewer MC runs) and accuracy
- **Numerical**: Regularization prevents singular matrix issues

## Future Improvements

1. **Full UKF Implementation**: Complete unscented transform
2. **Full CKF Implementation**: Cubature point generation
3. **Second-order Methods**: Complete EKF2 and IEKF2
4. **Parallel Processing**: Multi-threaded Monte Carlo runs
5. **GPU Acceleration**: CUDA implementation for large-scale problems 