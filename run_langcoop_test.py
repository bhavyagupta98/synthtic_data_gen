#!/usr/bin/env python3
"""
LangCoop Test Runner - Full Integration
Complete independent module following LangCoop architecture.

Features:
- Connects to CARLA at carla-rpc
- Uses VLMPlannerSpeedCurvature with local vLLM
- Chain-of-Thought (CoT) reasoning
- Full metrics calculation (RC%, DS, infractions)
- Visualization and reporting
"""

import argparse
import logging
import sys
from pathlib import Path
import carla
import time
import numpy as np

# Add paths
sys.path.insert(0, str(Path(__file__).parent))

from test_runner.agents import LangCoopAgent
from test_runner.evaluator.metrics import MetricsCalculator
from test_runner.evaluator.scenario_manager import ScenarioManager, Scenario, Route
from test_runner.visualization import MetricsVisualizer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class LangCoopTestRunner:
    """
    Full independent test module following LangCoop architecture.
    """
    
    def __init__(
        self,
        carla_host: str = 'carla-rpc',
        carla_port: int = 2000,
        agent_config: str = 'configs/langcoop_agent_config.yaml',
        results_dir: str = 'test_results_langcoop'
    ):
        """
        Initialize LangCoop test runner.
        
        Args:
            carla_host: CARLA server hostname (e.g., 'carla-rpc')
            carla_port: CARLA server port
            agent_config: Path to agent configuration
            results_dir: Directory for results
        """
        self.carla_host = carla_host
        self.carla_port = carla_port
        self.agent_config = agent_config
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Create directories for images
        self.images_dir = self.results_dir / 'images'
        self.images_dir.mkdir(parents=True, exist_ok=True)
        
        self.client = None
        self.world = None
        self.agent = None
        self.vehicle = None
        
        self.metrics_calculator = MetricsCalculator()
        # ScenarioManager will use test_runner/scenarios/ by default
        self.scenario_manager = ScenarioManager()
        self.visualizer = MetricsVisualizer()
        
        logger.info("LangCoop Test Runner initialized")
    
    def connect_to_carla(self):
        """Connect to CARLA server."""
        try:
            logger.info(f"Connecting to CARLA at {self.carla_host}:{self.carla_port}...")
            self.client = carla.Client(self.carla_host, self.carla_port)
            self.client.set_timeout(20.0)
            self.world = self.client.get_world()
            logger.info(f"✓ Connected to CARLA successfully")
            
            # Log server info
            server_version = self.client.get_server_version()
            logger.info(f"CARLA Server version: {server_version}")
            
            return True
        except Exception as e:
            logger.error(f"✗ Failed to connect to CARLA: {e}")
            logger.error(f"Make sure CARLA server is running on {self.carla_host}:{self.carla_port}")
            return False
    
    def load_map(self, map_name: str):
        """Load CARLA map."""
        try:
            logger.info(f"Loading map: {map_name}")
            self.world = self.client.load_world(map_name)
            time.sleep(2)  # Wait for map to load
            logger.info(f"✓ Map loaded: {map_name}")
        except Exception as e:
            logger.error(f"✗ Failed to load map {map_name}: {e}")
            raise
    
    def setup_environment(self, scenario: Scenario):
        """
        Setup simulation environment (weather, traffic, etc).
        
        Args:
            scenario: Scenario object with environmental parameters
        """
        logger.info(f"Setting up scenario: {scenario.scenario_id}")
        
        # Set weather
        weather = carla.WeatherParameters()
        if scenario.weather:
            weather.cloudiness = scenario.weather.get('cloudiness', 30)
            weather.precipitation = scenario.weather.get('precipitation', 0)
            weather.wind_intensity = scenario.weather.get('wind_intensity', 0)
            weather.sun_altitude_angle = scenario.weather.get('sun_altitude_angle', 45)
            weather.fog_density = scenario.weather.get('fog_density', 0)
        
        self.world.set_weather(weather)
        
        # Set simulation settings
        settings = self.world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 0.05  # 20 Hz
        self.world.apply_settings(settings)
        
        logger.info(f"✓ Environment setup complete")
    
    def spawn_vehicle(self, spawn_point: carla.Transform):
        """
        Spawn ego vehicle at specified point.
        
        Args:
            spawn_point: CARLA Transform for spawn location
            
        Returns:
            Vehicle actor
        """
        blueprint_library = self.world.get_blueprint_library()
        vehicle_bp = blueprint_library.filter('vehicle.tesla.model3')[0]

        # 1) Try requested spawn point with small z-offset retries.
        z_offsets = [0.0, 0.3, 0.6, 1.0]
        for z_offset in z_offsets:
            candidate = carla.Transform(
                location=carla.Location(
                    x=spawn_point.location.x,
                    y=spawn_point.location.y,
                    z=spawn_point.location.z + z_offset,
                ),
                rotation=spawn_point.rotation,
            )
            vehicle = self.world.try_spawn_actor(vehicle_bp, candidate)
            if vehicle is not None:
                logger.info(
                    f"✓ Vehicle spawned at ({candidate.location.x:.1f}, {candidate.location.y:.1f}, {candidate.location.z:.1f})"
                )
                return vehicle

        # 2) Fallback to map spawn points if requested point is occupied.
        fallback_points = self.world.get_map().get_spawn_points()
        for fallback in fallback_points:
            vehicle = self.world.try_spawn_actor(vehicle_bp, fallback)
            if vehicle is not None:
                logger.warning(
                    "Requested spawn point occupied; spawned at fallback "
                    f"({fallback.location.x:.1f}, {fallback.location.y:.1f}, {fallback.location.z:.1f})"
                )
                return vehicle

        raise RuntimeError("Failed to spawn vehicle: all candidate spawn points are occupied")
    
    def setup_agent(self):
        """Setup LangCoop VLM agent."""
        try:
            logger.info("Initializing LangCoop VLM Agent...")
            self.agent = LangCoopAgent(agent_config_path=self.agent_config)
            self.agent.setup(self.vehicle)
            logger.info("✓ LangCoop VLM Agent initialized with:")
            logger.info(f"  - VLM Planner: VLMPlannerSpeedCurvature")
            logger.info(f"  - Chain-of-Thought (CoT) prompting")
            logger.info(f"  - Local vLLM endpoint")
            return True
        except Exception as e:
            logger.error(f"✗ Failed to setup agent: {e}")
            logger.error("Make sure vLLM server is running and accessible")
            return False
    
    def run_scenario(
        self,
        scenario: Scenario,
        route: Route,
        max_steps: int = 1000,
        skip_frames: int = 4
    ):
        """
        Run test scenario with LangCoop agent.
        
        Args:
            scenario: Scenario configuration
            route: Route to follow
            max_steps: Maximum simulation steps
            skip_frames: Frame skip for control (LangCoop uses 4)
            
        Returns:
            Metrics dictionary
        """
        logger.info(f"Starting scenario: {scenario.scenario_id}")
        route_distance = route.compute_distance()
        route_metric_id = f"{scenario.scenario_id}_agent_0"
        logger.info(f"Route: {len(route.waypoints)} waypoints, {route_distance:.1f}m")
        
        # Setup environment
        self.setup_environment(scenario)
        
        # Spawn vehicle at first waypoint
        spawn_point = route.waypoints[0].to_transform()
        self.vehicle = self.spawn_vehicle(spawn_point)
        
        # Setup agent
        if not self.setup_agent():
            return None
        
        # Set waypoints for agent
        world_map = self.world.get_map()
        for wp in route.waypoints[1:]:  # Skip first (spawn point)
            wp_location = carla.Location(
                x=wp.location['x'],
                y=wp.location['y'],
                z=wp.location['z']
            )
            waypoint = world_map.get_waypoint(wp_location)
            if waypoint:
                self.agent.set_target_waypoint(waypoint)
        
        # Simulation loop
        logger.info("Starting simulation loop...")
        start_time = time.time()
        completed_distance = 0.0
        prev_location = self.vehicle.get_location()
        speed_samples = []
        
        collision_sensor = self._setup_collision_sensor(route_metric_id)
        
        for step in range(max_steps):
            self.world.tick()
            
            # Agent step (follows LangCoop skip_frames=4 pattern)
            if step % skip_frames == 0:
                try:
                    control = self.agent.step()
                    self.vehicle.apply_control(control)
                except Exception as e:
                    logger.error(f"Agent step failed at step {step}: {e}")
                    self.metrics_calculator.record_event(
                        route_metric_id,
                        step,
                        'timeout',
                        description=str(e)
                    )
                    break
            
            # Update metrics
            current_location = self.vehicle.get_location()
            distance_delta = current_location.distance(prev_location)
            completed_distance += distance_delta
            prev_location = current_location
            
            # Save camera image every 10 steps (every ~0.5s at 20 Hz)
            if step % 10 == 0 and 'camera' in self.agent.sensor_data:
                self._save_camera_image(
                    self.agent.sensor_data['camera'],
                    scenario.scenario_id,
                    step
                )
            
            # Check infractions
            velocity = self.vehicle.get_velocity()
            speed = np.linalg.norm([velocity.x, velocity.y, velocity.z])
            speed_samples.append(speed)
            
            if speed > 20.5:  # Speed limit from LangCoop
                self.metrics_calculator.record_event(
                    route_metric_id,
                    step,
                    'speed_violation',
                    description=f"speed={speed:.2f}"
                )
            
            # Progress logging
            if step % 100 == 0:
                progress = (completed_distance / route_distance) * 100 if route_distance > 0 else 0.0
                logger.info(f"Step {step}/{max_steps} | Progress: {progress:.1f}% | Speed: {speed:.1f} m/s")
        
        # Cleanup
        if collision_sensor:
            collision_sensor.destroy()
        if self.agent:
            self.agent.destroy()
        if self.vehicle and self.vehicle.is_alive:
            self.vehicle.destroy()
        
        # Calculate final metrics
        elapsed_time = time.time() - start_time
        self.metrics_calculator.set_route_completion(
            route_metric_id,
            completed_distance,
            route_distance,
            elapsed_time
        )
        if speed_samples:
            self.metrics_calculator.set_speed_metrics(
                route_metric_id,
                float(np.mean(speed_samples)),
                float(np.max(speed_samples))
            )
        self.metrics_calculator.calculate_driving_score(route_metric_id)

        metrics = self.metrics_calculator.get_summary()
        route_metrics = metrics.get('routes', {}).get(route_metric_id, {})
        
        logger.info(f"Scenario complete:")
        logger.info(f"  Route Completion: {route_metrics.get('rc', 0.0):.1f}%")
        logger.info(f"  Driving Score: {route_metrics.get('ds', 0.0):.1f}/100")
        logger.info(f"  Collisions: {metrics['total_collisions']}")
        logger.info(f"  Total Violations: {metrics['total_violations']}")
        
        return metrics
    
    def _setup_collision_sensor(self, route_metric_id: str):
        """Setup collision detection sensor."""
        blueprint_library = self.world.get_blueprint_library()
        collision_bp = blueprint_library.find('sensor.other.collision')
        collision_sensor = self.world.spawn_actor(
            collision_bp,
            carla.Transform(),
            attach_to=self.vehicle
        )
        
        def on_collision(event):
            logger.warning(f"Collision detected with {event.other_actor.type_id}")
            self.metrics_calculator.record_event(
                route_metric_id,
                int(event.frame),
                'collision',
                description=event.other_actor.type_id
            )
        
        collision_sensor.listen(on_collision)
        return collision_sensor
    
    def _save_camera_image(self, image: np.ndarray, scenario_id: str, step: int):
        """Save camera image to disk for visualization."""
        try:
            from PIL import Image
            scenario_img_dir = self.images_dir / scenario_id
            scenario_img_dir.mkdir(parents=True, exist_ok=True)
            
            img_path = scenario_img_dir / f"frame_{step:06d}.jpg"
            pil_image = Image.fromarray(image.astype(np.uint8))
            pil_image.save(img_path, quality=85)
        except Exception as e:
            logger.debug(f"Failed to save image at step {step}: {e}")
    
    def run_tests(
        self,
        scenario_ids: list = None,
        max_steps: int = 1000
    ):
        """
        Run multiple test scenarios.
        
        Args:
            scenario_ids: List of scenario IDs to run (None = all)
            max_steps: Maximum steps per scenario
        """
        if not self.connect_to_carla():
            return
        
        if scenario_ids:
            selected_ids = scenario_ids
        else:
            selected_ids = self.scenario_manager.list_scenarios()

        scenarios = []
        for sid in selected_ids:
            scenario = self.scenario_manager.get_scenario(sid)
            if scenario is not None:
                scenarios.append(scenario)
        
        logger.info(f"Running {len(scenarios)} scenarios")
        if not scenarios:
            logger.warning("No scenarios found. Add scenario JSON files under test_runner/scenarios")
            return
        
        for scenario in scenarios:
            route = self.scenario_manager.get_route(scenario.route_id)
            if route is None:
                logger.warning(f"Skipping scenario {scenario.scenario_id}: missing route {scenario.route_id}")
                continue
            
            # Load appropriate map
            map_name = route.map_name
            self.load_map(map_name)
            
            # Run scenario
            self.run_scenario(scenario, route, max_steps=max_steps)
        
        # Generate report
        self._generate_report()
    
    def _generate_report(self):
        """Generate evaluation report with plots."""
        logger.info("Generating evaluation report...")
        summary = self.metrics_calculator.get_summary()
        self.visualizer.add_metrics(summary)
        
        self.visualizer.export_json(output_dir=str(self.results_dir), filename='metrics.json')
        self.visualizer.save_detailed_report(output_dir=str(self.results_dir), filename='detailed_report.txt')
        self.visualizer.plot_summary(output_dir=str(self.results_dir))

        logger.info(f"✓ Results saved in: {self.results_dir}")


def main():
    parser = argparse.ArgumentParser(description='LangCoop Test Runner - Full Integration')
    parser.add_argument('--host', type=str, default='carla-rpc',
                        help='CARLA server hostname (default: carla-rpc)')
    parser.add_argument('--port', type=int, default=2000,
                        help='CARLA server port (default: 2000)')
    parser.add_argument('--agent-config', type=str,
                        default='configs/langcoop_agent_config.yaml',
                        help='Agent configuration file')
    parser.add_argument('--scenario-ids', type=str, nargs='+',
                        default=['town05_clear_easy'],
                        help='Scenario IDs to run (default: town05_clear_easy)')
    parser.add_argument('--max-steps', type=int, default=500,
                        help='Maximum steps per scenario (default: 500 = 25 seconds simulation)')
    parser.add_argument('--results-dir', type=str, default='test_results_langcoop',
                        help='Results directory')
    
    args = parser.parse_args()
    
    logger.info("=" * 70)
    logger.info("LangCoop Test Runner - Full Integration")
    logger.info("=" * 70)
    logger.info(f"CARLA Server: {args.host}:{args.port}")
    logger.info(f"Agent Config: {args.agent_config}")
    logger.info("=" * 70)
    
    runner = LangCoopTestRunner(
        carla_host=args.host,
        carla_port=args.port,
        agent_config=args.agent_config,
        results_dir=args.results_dir
    )
    
    runner.run_tests(
        scenario_ids=args.scenario_ids,
        max_steps=args.max_steps
    )


if __name__ == '__main__':
    main()
