"""
VLM Planner - Speed Curvature (CARLA 0.9.16 + Python 3.12)
Ported from LangCoop's VLMPlannerSpeedCurvature to work with modern CARLA/Python.
"""

import re
import json
import logging
import base64
from io import BytesIO
from PIL import Image
import numpy as np
import torch
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


class VLMPlannerSpeedCurvature:
    """
    VLM Planner using speed-curvature prediction.
    Follows LangCoop architecture with Chain-of-Thought reasoning.
    
    Compatible with CARLA 0.9.16 and Python 3.12.
    """
    
    def __init__(self, api_model_name: str, api_base_url: str, api_key: str, **kwargs):
        """
        Initialize VLM planner.
        
        Args:
            api_model_name: Model name (e.g., 'qwen2-vl-7b')
            api_base_url: API endpoint (e.g., 'http://localhost:8000/v1')
            api_key: API key (use 'EMPTY' for local vLLM)
        """
        self.api_model_name = api_model_name
        self.api_base_url = api_base_url
        self.api_key = api_key
        
        # Initialize OpenAI-compatible client
        try:
            from openai import OpenAI
            self.client = OpenAI(api_key=api_key, base_url=api_base_url)
            logger.info(f"VLM Planner initialized: {api_model_name} @ {api_base_url}")
        except ImportError:
            logger.error("OpenAI package not installed. Install with: pip install openai")
            raise
        
        self.IMAGE_PLACEHOLDER = "<IMAGE_PLACEHOLDER>"
        self._zero_deadlock_streak = 0
    
    def forward(self, perception_memory_bank: List[Dict], model_config: Dict) -> List[Dict]:
        """
        Main forward pass: predict speed and curvature from perception data.
        
        Args:
            perception_memory_bank: List of historical perception frames
            model_config: Configuration with prompts and settings
            
        Returns:
            List of predicted results (one per agent)
        """
        if len(perception_memory_bank) < 2:
            # Need at least 2 frames for history
            logger.warning("Not enough frames in perception memory bank")
            return [self._get_default_prediction()]
        
        # For single agent, agent_idx = 0
        agent_idx = 0
        
        # Step 1: Get ego vehicle history
        ego_history_json = self._get_ego_history(perception_memory_bank, agent_idx)
        
        # Step 2: Chain-of-Thought reasoning
        front_image = perception_memory_bank[-1]['front_image']
        
        prompt_template = model_config.get('planning', {}).get('prompt_template', {})

        scene_description = self._get_scene_description(
            front_image, 
            prompt_template
        )
        
        object_description = self._get_objects_description(
            front_image,
            prompt_template
        )
        
        intent_description = self._get_intent_description(
            front_image,
            perception_memory_bank[-1]['target'][agent_idx],
            prompt_template
        )
        
        # Step 3: Combined prediction
        target_waypoint = perception_memory_bank[-1]['target'][agent_idx]
        target_description = self._get_target_description(target_waypoint)
        
        result = self._predict_speed_curvature(
            front_image,
            scene_description,
            object_description,
            intent_description,
            ego_history_json,
            target_description,
            prompt_template
        )
        
        return [result]

    def _resolve_prompt(self, prompt_template: Dict, candidates: List[str], fallback: str) -> str:
        """Resolve prompt text from multiple possible keys and value shapes."""
        if not isinstance(prompt_template, dict):
            return fallback

        for key in candidates:
            if key not in prompt_template:
                continue

            value = prompt_template[key]

            if isinstance(value, str) and value.strip():
                return value

            if isinstance(value, dict):
                for subkey in ("concise", "default", "text"):
                    subval = value.get(subkey)
                    if isinstance(subval, str) and subval.strip():
                        return subval

                for subval in value.values():
                    if isinstance(subval, str) and subval.strip():
                        return subval

        return fallback
    
    def _get_ego_history(self, perception_memory_bank: List[Dict], agent_idx: int) -> str:
        """
        Build JSON string with ego vehicle history.
        
        Args:
            perception_memory_bank: List of perception frames
            agent_idx: Agent index
            
        Returns:
            JSON string with ego history
        """
        ego_history_list = []
        prev_position = None
        prev_yaw = None
        dt = 0.5  # 0.5 seconds between frames
        
        # Current position as reference
        curr_pose = perception_memory_bank[-1]['detmap_pose'][agent_idx]
        curr_x, curr_y = float(curr_pose[0]), float(curr_pose[1])
        
        # Process historical frames
        for frame_data in perception_memory_bank[:-1]:
            timestamp = frame_data['timestamp']
            ego_pose = frame_data['detmap_pose'][agent_idx]
            x, y = float(ego_pose[0]), float(ego_pose[1])
            yaw = float(frame_data['ego_yaw'][agent_idx])
            
            # Calculate speed
            if prev_position is not None:
                dx = x - prev_position[0]
                dy = y - prev_position[1]
                speed = np.sqrt(dx * dx + dy * dy) / dt
            else:
                speed = 0.0
            
            # Calculate curvature
            if prev_yaw is not None:
                d_yaw = yaw - prev_yaw
                distance = max(1e-6, speed * dt)
                curvature = d_yaw / distance
            else:
                curvature = 0.0
            
            record = {
                "timestamp": timestamp,
                "speed": round(speed, 3),
                "curvature": round(curvature, 3),
                "waypoints": [round(x - curr_x, 2), round(y - curr_y, 2)]
            }
            ego_history_list.append(record)
            prev_position = (x, y)
            prev_yaw = yaw
        
        ego_history_json = {"ego_history": ego_history_list}
        return json.dumps(ego_history_json, indent=2)
    
    def _encode_image(self, image: np.ndarray) -> str:
        """Encode numpy image to base64."""
        if image is None:
            raise ValueError("Image is None")
        
        # Convert to PIL Image
        pil_image = Image.fromarray(image.astype(np.uint8))
        
        # Encode to base64
        buffer = BytesIO()
        pil_image.save(buffer, format='PNG')
        img_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        
        return img_base64
    
    def _get_scene_description(self, image: np.ndarray, prompt_template: Dict) -> str:
        """Get scene description using CoT."""
        try:
            img_base64 = self._encode_image(image)

            scene_prompt = self._resolve_prompt(
                prompt_template,
                ["scene"],
                "Describe the driving scenario, including weather, traffic, and road conditions."
            )
            scene_prompt = scene_prompt.replace(self.IMAGE_PLACEHOLDER, "")
            
            response = self.client.chat.completions.create(
                model=self.api_model_name,
                messages=[{
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{img_base64}"
                            }
                        },
                        {
                            "type": "text",
                            "text": scene_prompt
                        }
                    ]
                }],
                max_tokens=200,
                temperature=0.3
            )
            
            description = response.choices[0].message.content.strip()
            logger.debug(f"Scene: {description[:100]}...")
            return description
            
        except Exception as e:
            logger.warning(f"Scene description failed: {e}")
            return "Clear conditions, standard road layout."
    
    def _get_objects_description(self, image: np.ndarray, prompt_template: Dict) -> str:
        """Get object detection using CoT."""
        try:
            img_base64 = self._encode_image(image)

            object_prompt = self._resolve_prompt(
                prompt_template,
                ["objects", "default"],
                "Identify important road users in the driving scene. List two or three of them with their location and a short description of their status and intent."
            )
            object_prompt = object_prompt.replace(self.IMAGE_PLACEHOLDER, "")
            
            response = self.client.chat.completions.create(
                model=self.api_model_name,
                messages=[{
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{img_base64}"
                            }
                        },
                        {
                            "type": "text",
                            "text": object_prompt
                        }
                    ]
                }],
                max_tokens=200,
                temperature=0.3
            )
            
            description = response.choices[0].message.content.strip()
            logger.debug(f"Objects: {description[:100]}...")
            return description
            
        except Exception as e:
            logger.warning(f"Object description failed: {e}")
            return "No significant objects detected."
    
    def _get_intent_description(self, image: np.ndarray, target: List[float], 
                                prompt_template: Dict) -> str:
        """Get driving intent using CoT."""
        try:
            img_base64 = self._encode_image(image)
            
            # Format target description
            target_desc = self._get_target_description(target)

            intent_prompt = self._resolve_prompt(
                prompt_template,
                ["intent", "default"],
                "Should you turn left, turn right, go straight, slightly adjust direction, accelerate, or decelerate? Describe how you would navigate the vehicle to reach the target."
            )
            intent_prompt = intent_prompt.replace(self.IMAGE_PLACEHOLDER, "")
            intent_prompt = intent_prompt.replace("{target_description}", target_desc)
            
            response = self.client.chat.completions.create(
                model=self.api_model_name,
                messages=[{
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{img_base64}"
                            }
                        },
                        {
                            "type": "text",
                            "text": intent_prompt
                        }
                    ]
                }],
                max_tokens=200,
                temperature=0.3
            )
            
            description = response.choices[0].message.content.strip()
            logger.debug(f"Intent: {description[:100]}...")
            return description
            
        except Exception as e:
            logger.warning(f"Intent description failed: {e}")
            return "Maintain current course and speed."
    
    def _get_target_description(self, target: List[float]) -> str:
        """Format target waypoint description."""
        x_distance = abs(target[0])
        y_distance = abs(target[1])
        
        x_direction = "left" if target[0] < 0 else "right"
        y_direction = "front" if target[1] > 0 else "rear"
        
        return f"The target is {x_distance:.1f} meters to your {x_direction} and {y_distance:.1f} meters to your {y_direction}."
    
    def _predict_speed_curvature(self, image: np.ndarray, scene_desc: str,
                                 object_desc: str, intent_desc: str,
                                 ego_history: str, target_desc: str,
                                 prompt_template: Dict) -> Dict:
        """
        Final prediction step: combine all context to predict speed and curvature.
        
        Returns:
            Dict with 'target_speed' and 'curvature' arrays
        """
        try:
            img_base64 = self._encode_image(image)

            # Combined prompt - use ORIGINAL placeholder names
            comb_prompt = self._resolve_prompt(
                prompt_template,
                ["prediction"],
                (
                    "You are an autonomous driving vehicle controller. "
                    "You have access to a front-view camera image. <IMAGE_PLACEHOLDER>\n"
                    "Here is the environment description and detected objects:\n"
                    "- Scene: {scene_description}\n"
                    "- Objects and intents: {object_description}\n"
                    "- Ego vehicle history: {ego_history_prompt}\n"
                    "- Intent of the ego vehicle: {intent_description}\n"
                    "- Collaborative agents' information are described as follows: {collab_agent_description}\n"
                    "{target_description}\n"
                    "Generate the vehicle's desired speed and curvature for the next 5 timestamps, ensuring safe and efficient movement towards the target.\n"
                    "- Speed (m/s): Range [0, 20]\n"
                    "- Curvature (degree/m): Range [-180, 180]\n"
                    "- Negative curvature = turning left\n"
                    "- Positive curvature = turning right\n"
                    "- Ensure traffic rule compliance\n"
                    "- Numerical values must be integers.\n"
                    "- Avoid collisions by slowing down or changing lanes.\n"
                    'Output MUST be a valid JSON structure with the key "predicted_speeds_curvatures" containing a list of 5 [speed, curvature] pairs.\n'
                    "```json\n"
                    "{\n"
                    '    "predicted_speeds_curvatures": [\n'
                    "        [speed_1, curvature_1],\n"
                    "        [speed_2, curvature_2],\n"
                    "        [speed_3, curvature_3],\n"
                    "        [speed_4, curvature_4],\n"
                    "        [speed_5, curvature_5]\n"
                    "    ]\n"
                    "}\n"
                    "```\n"
                    "No additional text outside of this JSON format."
                )
            )
            comb_prompt = comb_prompt.replace(self.IMAGE_PLACEHOLDER, "")
            
            # Log the values being replaced
            logger.info(f"[REPLACEMENT DEBUG]")
            logger.info(f"  scene_desc exists: {bool(scene_desc)}, len={len(scene_desc) if scene_desc else 0}")
            logger.info(f"  object_desc exists: {bool(object_desc)}, len={len(object_desc) if object_desc else 0}")
            logger.info(f"  ego_history exists: {bool(ego_history)}, len={len(ego_history) if ego_history else 0}")
            logger.info(f"  intent_desc exists: {bool(intent_desc)}, len={len(intent_desc) if intent_desc else 0}")
            logger.info(f"  target_desc exists: {bool(target_desc)}, len={len(target_desc) if target_desc else 0}")
            
            comb_prompt = comb_prompt.replace("{scene_description}", scene_desc)
            comb_prompt = comb_prompt.replace("{object_description}", object_desc)  # SINGULAR
            comb_prompt = comb_prompt.replace("{ego_history_prompt}", ego_history)    # WITH _prompt
            comb_prompt = comb_prompt.replace("{intent_description}", intent_desc)
            comb_prompt = comb_prompt.replace("{target_description}", target_desc)
            comb_prompt = comb_prompt.replace("{collab_agent_description}", "")  # Single agent
            
            # Log the prepared prompt for debugging
            logger.debug(f"[PROMPT PREPARED]\n{comb_prompt}\n[END PROMPT]")
            
            response = self.client.chat.completions.create(
                model=self.api_model_name,
                messages=[{
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{img_base64}"
                            }
                        },
                        {
                            "type": "text",
                            "text": comb_prompt
                        }
                    ]
                }],
                max_tokens=512,
                temperature=0.3
            )
            
            result_text = response.choices[0].message.content.strip()
            logger.debug(f"VLM Response (full): {result_text}")
            
            # Parse JSON result
            parsed_result = self._parse_speed_curvature_response(result_text)

            # Track consecutive zero predictions
            speeds = parsed_result.get('target_speed', [])
            all_zero = speeds and all(abs(float(s)) < 1e-3 for s in speeds)
            
            if all_zero:
                self._zero_deadlock_streak += 1
            else:
                self._zero_deadlock_streak = 0
            
            # Guard against deadlock: if model outputs all zeros while intent is to accelerate
            # and scene appears clear, apply an escalating forward rollout.
            # Never override if intent explicitly requires stopping.
            intent_text = (intent_desc or '').lower()
            explicit_stop_intent = any(token in intent_text for token in 
                                      ("stop", "decelerate", "brake", "slow down", "halt", "wait"))
            
            should_override = self._should_override_zero_prediction(
                parsed_result, scene_desc, object_desc, intent_desc
            )
            
            # Apply override only if:
            # 1. Clear-road + accelerate conditions met, OR
            # 2. Streak >= 3 BUT no explicit stop intent
            if should_override and self._zero_deadlock_streak > 0:
                parsed_result = self._get_zero_deadlock_override(self._zero_deadlock_streak)
                logger.warning(
                    "Applied zero-speed deadlock override | streak=%d | applied_speed=%.2f | reason=clear-road",
                    self._zero_deadlock_streak,
                    float(parsed_result['target_speed'][0])
                )
            elif self._zero_deadlock_streak >= 3 and not explicit_stop_intent:
                parsed_result = self._get_zero_deadlock_override(self._zero_deadlock_streak)
                logger.warning(
                    "Applied zero-speed deadlock override | streak=%d | applied_speed=%.2f | reason=persistent-deadlock",
                    self._zero_deadlock_streak,
                    float(parsed_result['target_speed'][0])
                )

            logger.info(f"Predicted: speed={parsed_result['target_speed'][0]:.2f} m/s, "
                       f"curvature={parsed_result['curvature'][0]:.3f} rad/m")
            
            return parsed_result
            
        except Exception as e:
            logger.error(f"Speed-curvature prediction failed: {e}")
            return self._get_default_prediction()
    
    def _parse_speed_curvature_response(self, response_text: str) -> Dict:
        """Parse VLM response to extract speed-curvature pairs."""
        try:
            # Log FULL response for debugging
            logger.debug(f"[VLM FULL RESPONSE]\n{response_text}\n[END RESPONSE]")
            
            # Extract JSON from response - try multiple patterns
            json_patterns = [
                (re.compile(r"```json\s*([\s\S]*?)\s*```", re.MULTILINE), "json_block"),
                (re.compile(r"\{[\s\S]*\}", re.MULTILINE), "json_object"),
            ]
            
            json_str = None
            for pattern, pname in json_patterns:
                json_match = pattern.search(response_text)
                if json_match:
                    if json_match.lastindex and json_match.group(1):
                        json_str = json_match.group(1)
                    else:
                        json_str = json_match.group(0)
                    logger.info(f"[JSON EXTRACTION] Method: {pname}")
                    logger.debug(f"Extracted JSON: {json_str[:200]}...")
                    break
            
            if not json_str:
                logger.warning("[JSON EXTRACTION] No JSON pattern matched!")
            
            if json_str:
                # Strip C-style comments (// ...) from JSON string
                json_str = re.sub(r'//.*?(?=[\n,\]])', '', json_str)
                
                json_result = json.loads(json_str)
                logger.info(f"[JSON PARSED] Keys: {list(json_result.keys()) if isinstance(json_result, dict) else 'list'}")
                
                # Handle various JSON response structures
                pairs = None
                if isinstance(json_result, dict):
                    if 'predicted_speeds_curvatures' in json_result:
                        pairs = json_result['predicted_speeds_curvatures']
                        logger.info(f"[PAIRS SOURCE] predicted_speeds_curvatures")
                    elif 'predictions' in json_result:
                        pairs = json_result['predictions']
                        logger.info(f"[PAIRS SOURCE] predictions")
                    elif 'speeds_curvatures' in json_result:
                        pairs = json_result['speeds_curvatures']
                        logger.info(f"[PAIRS SOURCE] speeds_curvatures")
                    else:
                        logger.warning(f"[PAIRS SOURCE] No recognized key. Available: {list(json_result.keys())}")
                elif isinstance(json_result, list):
                    pairs = json_result
                    logger.info(f"[PAIRS SOURCE] Direct list")
                
                if pairs:
                    try:
                        logger.info(f"[EXTRACTION] Extracting from {len(pairs)} pairs: {pairs}")
                        speeds = [float(pair[0]) for pair in pairs]
                        curvatures = [float(pair[1]) for pair in pairs]
                        
                        logger.info(f"[RAW VALUES] speeds={speeds}, curvatures={curvatures}")
                        
                        # Clamp to reasonable ranges
                        speeds = [max(0.0, min(20.0, s)) for s in speeds]
                        curvatures = [max(-0.5, min(0.5, c)) for c in curvatures]
                        
                        logger.info(f"Successfully parsed: speeds={speeds}, curvatures={curvatures}")
                        
                        return {
                            'target_speed': speeds,
                            'curvature': curvatures,
                            'dt': 0.5
                        }
                    except (ValueError, IndexError, TypeError) as e:
                        logger.warning(f"Failed to extract speed/curvature from pairs: {e}")
            
            # Fallback: try to extract numbers as [s0, c0, s1, c1, ...]
            logger.debug("JSON extraction failed, trying regex number extraction...")
            numbers = re.findall(r'[-+]?\d*\.?\d+', response_text)
            logger.debug(f"Extracted numbers: {numbers}")
            
            if len(numbers) >= 10:  # At least 5 pairs
                try:
                    speeds = [float(numbers[i*2]) for i in range(5)]
                    curvatures = [float(numbers[i*2+1]) for i in range(5)]
                    
                    speeds = [max(0.0, min(20.0, s)) for s in speeds]
                    curvatures = [max(-0.5, min(0.5, c)) for c in curvatures]
                    
                    logger.info(f"Extracted via regex: speeds={speeds}, curvatures={curvatures}")
                    
                    return {
                        'target_speed': speeds,
                        'curvature': curvatures,
                        'dt': 0.5
                    }
                except (ValueError, IndexError) as e:
                    logger.warning(f"Regex extraction failed: {e}")
            
        except json.JSONDecodeError as e:
            logger.warning(f"JSON decode failed: {e}")
        except Exception as e:
            logger.error(f"Unexpected error parsing VLM response: {e}", exc_info=True)
        
        logger.warning("Returning default prediction due to parse failure")
        return self._get_default_prediction()

    def _should_override_zero_prediction(self, prediction: Dict, scene_desc: str, object_desc: str, intent_desc: str) -> bool:
        """Return True when all-zero prediction likely indicates VLM deadlock on a clear road."""
        speeds = prediction.get('target_speed', [])
        curvatures = prediction.get('curvature', [])
        if not speeds or not curvatures:
            return False

        all_zero = all(abs(float(s)) < 1e-3 for s in speeds) and all(abs(float(c)) < 1e-3 for c in curvatures)
        if not all_zero:
            return False

        intent_text = (intent_desc or '').lower()
        accelerate_intent = any(token in intent_text for token in ("accelerate", "speed up", "go straight"))
        if not accelerate_intent:
            return False

        scene_text = (scene_desc or '').lower()
        object_text = (object_desc or '').lower()

        clear_scene_markers = [
            "no visible traffic",
            "no vehicles",
            "empty road",
            "clear road",
            "no pedestrians"
        ]
        stop_markers = ["stop sign", "red light", "traffic light"]

        scene_looks_clear = any(marker in scene_text for marker in clear_scene_markers)
        hard_stop_cue = any(marker in scene_text for marker in stop_markers)

        # If object description only talks about static/non-traffic entities,
        # treat as likely hallucinated hazard content for control purposes.
        non_traffic_only = any(token in object_text for token in ("birds", "trees", "streetlights")) and not any(
            token in object_text for token in ("vehicle", "car", "truck", "pedestrian", "cyclist")
        )

        return accelerate_intent and scene_looks_clear and (non_traffic_only or not hard_stop_cue)

    def _get_zero_deadlock_override(self, streak: int) -> Dict:
        """Escalating fallback to break persistent zero-speed deadlocks."""
        if streak <= 1:
            speeds = [2.0, 3.5, 5.0, 5.0, 5.0]
        elif streak == 2:
            speeds = [3.5, 5.0, 6.5, 7.0, 7.0]
        else:
            speeds = [5.0, 6.5, 8.0, 8.0, 8.0]

        return {
            'target_speed': speeds,
            'curvature': [0.0, 0.0, 0.0, 0.0, 0.0],
            'dt': 0.5
        }
    
    def _get_default_prediction(self) -> Dict:
        """Default prediction when VLM fails."""
        return {
            'target_speed': [8.0, 8.5, 9.0, 9.5, 10.0],
            'curvature': [0.0, 0.0, 0.0, 0.0, 0.0],
            'dt': 0.5
        }
    
    def to(self, device):
        """Compatibility method for device placement."""
        return self
    
    def eval(self):
        """Compatibility method for eval mode."""
        return self
