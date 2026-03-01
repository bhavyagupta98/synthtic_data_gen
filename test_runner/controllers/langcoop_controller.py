"""
PID Controller for LangCoop (CARLA 0.9.16 compatible)
Based on LangCoop's VLMControllerSpeedCurvature.
"""

import numpy as np
from typing import Dict


class LangCoopController:
    """
    PID Controller for speed-curvature based control.
    Compatible with CARLA 0.9.16.
    """
    
    def __init__(self, **kwargs):
        """Initialize controller with PID gains."""
        # Speed control PID gains
        self.speed_kp = kwargs.get('speed_kp', 0.8)
        self.speed_ki = kwargs.get('speed_ki', 0.1)
        self.speed_kd = kwargs.get('speed_kd', 0.2)
        
        # Steering control gains
        self.steer_kp = kwargs.get('steer_kp', 1.0)
        
        # Integral error tracking
        self.speed_error_integral = 0.0
        self.prev_speed_error = 0.0
        
        # Limits
        self.max_throttle = 1.0
        self.max_brake = 1.0
        self.max_steer = 1.0
        
    def run_step(self, route_info: Dict, curr_speed: float, buffer_idx: int = 0) -> Dict:
        """
        Compute control commands from route information.
        
        Args:
            route_info: Dict with 'target_speed' and 'curvature'
            curr_speed: Current vehicle speed (m/s)
            buffer_idx: Index into prediction buffer (0 = current)
            
        Returns:
            Dict with 'throttle', 'brake', 'steer'
        """
        # Extract target speed and curvature
        target_speeds = route_info.get('target_speed', [8.0])
        curvatures = route_info.get('curvature', [0.0])
        
        if isinstance(target_speeds, list):
            target_speed = target_speeds[min(buffer_idx, len(target_speeds)-1)]
        else:
            target_speed = float(target_speeds)
        
        if isinstance(curvatures, list):
            curvature = curvatures[min(buffer_idx, len(curvatures)-1)]
        else:
            curvature = float(curvatures)
        
        # Speed control (PID)
        speed_error = target_speed - curr_speed
        
        # Proportional
        throttle = self.speed_kp * speed_error
        
        # Integral
        self.speed_error_integral += speed_error * 0.05  # dt = 0.05s
        throttle += self.speed_ki * self.speed_error_integral
        
        # Derivative
        speed_error_derivative = (speed_error - self.prev_speed_error) / 0.05
        throttle += self.speed_kd * speed_error_derivative
        
        self.prev_speed_error = speed_error
        
        # Split throttle/brake
        brake = 0.0
        if throttle < 0:
            brake = min(-throttle, self.max_brake)
            throttle = 0.0
        else:
            throttle = min(throttle, self.max_throttle)
        
        # Steering from curvature
        # curvature = 1/radius, negative = turn right, positive = turn left
        steer = self.steer_kp * curvature
        steer = max(-self.max_steer, min(self.max_steer, steer))
        
        return {
            'throttle': float(throttle),
            'brake': float(brake),
            'steer': float(steer)
        }
    
    def reset(self):
        """Reset integral error."""
        self.speed_error_integral = 0.0
        self.prev_speed_error = 0.0
