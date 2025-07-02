import numpy as np
import json
import os
import sys
import pickle
from tqdm import tqdm
import h5py
from datetime import datetime
import hashlib
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import multiprocessing as mp
from functools import partial
import warnings
warnings.filterwarnings('ignore')

# Import your existing radar simulation
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from main_radar_script import run_enhanced_prf_simulation

class EliteRadarDataGenerator:
    """
    Elite training data generator for radar ML models
    
    Features:
    - Systematic coverage of parameter space
    - Physics-informed data generation
    - Multi-scale target scenarios
    - Balanced dataset creation
    - Advanced augmentation techniques
    - Quality validation
    - Parallel processing
    """
    
    def __init__(self, output_dir="data/training_data", 
                 validation_split=0.15, test_split=0.15):
        self.output_dir = Path(output_dir)
        self.validation_split = validation_split
        self.test_split = test_split
        
        # Setup logging
        self._setup_logging()
        
        # Create directory structure
        self._create_directories()
        
        # Define target type specifications
        self.target_specs = self._define_target_specifications()
        
        # Define scenario categories for balanced sampling
        self.scenario_categories = self._define_scenario_categories()
        
        # Quality thresholds
        self.quality_thresholds = {
            'min_snr': -5,  # Minimum detectable SNR
            'max_range_error': 0.05,  # 5% range error tolerance
            'min_detection_rate': 0.7  # Minimum detection success rate
        }
        
        self.logger.info(f"🚀 Elite Radar Data Generator initialized")
        self.logger.info(f"📁 Output directory: {self.output_dir}")
        
    def _setup_logging(self):
        """Setup comprehensive logging"""
        log_dir = self.output_dir / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_dir / f'data_generation_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
    def _create_directories(self):
        """Create optimized directory structure"""
        dirs = [
            "range_doppler_maps",
            "labels", 
            "metadata",
            "splits",
            "augmented",
            "validation",
            "quality_reports",
            "logs"
        ]
        
        for dir_name in dirs:
            (self.output_dir / dir_name).mkdir(parents=True, exist_ok=True)
    
    def _define_target_specifications(self):
        """Define comprehensive target specifications based on real-world data"""
        return {
            'fighter_jet': {
                'velocity_range': (150, 600),  # m/s
                'range_range': (5000, 80000),  # m
                'rcs_range': (1, 50),          # m²
                'altitude_range': (1000, 15000),  # m
                'maneuverability': 'high',
                'signature_stability': 'stable',
                'micro_doppler': 'minimal',
                'probability': 0.25
            },
            'commercial_aircraft': {
                'velocity_range': (80, 300),
                'range_range': (10000, 100000),
                'rcs_range': (50, 300),
                'altitude_range': (3000, 12000),
                'maneuverability': 'low',
                'signature_stability': 'very_stable',
                'micro_doppler': 'minimal',
                'probability': 0.20
            },
            'helicopter': {
                'velocity_range': (10, 120),
                'range_range': (500, 25000),
                'rcs_range': (5, 80),
                'altitude_range': (0, 3000),
                'maneuverability': 'very_high',
                'signature_stability': 'variable',
                'micro_doppler': 'strong_rotor',
                'probability': 0.15
            },
            'cruise_missile': {
                'velocity_range': (200, 800),
                'range_range': (1000, 50000),
                'rcs_range': (0.1, 2),
                'altitude_range': (10, 1000),
                'maneuverability': 'medium',
                'signature_stability': 'stable',
                'micro_doppler': 'minimal',
                'probability': 0.08
            },
            'drone_small': {
                'velocity_range': (5, 50),
                'range_range': (100, 5000),
                'rcs_range': (0.001, 0.5),
                'altitude_range': (0, 500),
                'maneuverability': 'high',
                'signature_stability': 'variable',
                'micro_doppler': 'propeller',
                'probability': 0.12
            },
            'drone_large': {
                'velocity_range': (20, 100),
                'range_range': (500, 15000),
                'rcs_range': (0.5, 10),
                'altitude_range': (0, 8000),
                'maneuverability': 'medium',
                'signature_stability': 'stable',
                'micro_doppler': 'minimal',
                'probability': 0.08
            },
            'bird_large': {
                'velocity_range': (10, 30),
                'range_range': (50, 2000),
                'rcs_range': (0.01, 0.1),
                'altitude_range': (0, 1000),
                'maneuverability': 'very_high',
                'signature_stability': 'chaotic',
                'micro_doppler': 'wing_flapping',
                'probability': 0.08
            },
            'bird_small': {
                'velocity_range': (5, 25),
                'range_range': (10, 1000),
                'rcs_range': (0.001, 0.01),
                'altitude_range': (0, 500),
                'maneuverability': 'very_high',
                'signature_stability': 'chaotic',
                'micro_doppler': 'wing_flapping',
                'probability': 0.04
            }
        }
    
    def _define_scenario_categories(self):
        """Define scenario categories for systematic coverage"""
        return {
            'single_target_clean': {
                'num_targets': 1,
                'environment': 'clean',
                'snr_range': (15, 35),
                'difficulty': 'easy',
                'weight': 0.25
            },
            'single_target_challenging': {
                'num_targets': 1,
                'environment': ['light_clutter', 'weather'],
                'snr_range': (5, 20),
                'difficulty': 'medium',
                'weight': 0.20
            },
            'multi_target_clean': {
                'num_targets': (2, 4),
                'environment': 'clean',
                'snr_range': (10, 30),
                'difficulty': 'medium',
                'weight': 0.20
            },
            'multi_target_challenging': {
                'num_targets': (2, 5),
                'environment': ['heavy_clutter', 'interference'],
                'snr_range': (3, 18),
                'difficulty': 'hard',
                'weight': 0.15
            },
            'edge_cases': {
                'num_targets': (1, 6),
                'environment': ['extreme_weather', 'jamming'],
                'snr_range': (-5, 12),
                'difficulty': 'extreme',
                'weight': 0.10
            },
            'empty_scenes': {
                'num_targets': 0,
                'environment': ['clean', 'light_clutter'],
                'snr_range': (10, 25),
                'difficulty': 'easy',
                'weight': 0.10
            }
        }
    
    def generate_systematic_scenarios(self, num_scenarios: int) -> List[Dict]:
        """Generate scenarios with systematic parameter coverage"""
        scenarios = []
        
        self.logger.info(f"🎯 Generating {num_scenarios} systematic scenarios...")
        
        # Calculate scenarios per category
        scenarios_per_category = {}
        for category, config in self.scenario_categories.items():
            count = int(num_scenarios * config['weight'])
            scenarios_per_category[category] = count
        
        # Generate scenarios for each category
        scenario_id = 0
        for category, count in scenarios_per_category.items():
            category_config = self.scenario_categories[category]
            
            self.logger.info(f"📂 Generating {count} scenarios for category: {category}")
            
            for _ in tqdm(range(count), desc=f"Category: {category}"):
                scenario = self._create_systematic_scenario(
                    scenario_id, category, category_config
                )
                scenarios.append(scenario)
                scenario_id += 1
        
        # Shuffle scenarios to avoid category clustering
        np.random.shuffle(scenarios)
        
        self.logger.info(f"✅ Generated {len(scenarios)} systematic scenarios")
        return scenarios
    
    def _create_systematic_scenario(self, scenario_id: int, 
                                  category: str, config: Dict) -> Dict:
        """Create one systematic scenario"""
        
        # 1. RADAR PARAMETERS - Systematic grid coverage
        radar_params = self._generate_systematic_radar_params()
        
        # 2. ENVIRONMENT - Based on category
        environment = self._generate_environment(config['environment'])
        
        # 3. SNR - From category range with physics-based variation
        base_snr = np.random.uniform(*config['snr_range'])
        snr_variation = self._calculate_snr_variation(environment, radar_params)
        final_snr = base_snr + snr_variation
        radar_params['snr_db'] = np.clip(final_snr, -10, 40)
        
        # 4. TARGETS - Balanced target type distribution
        if isinstance(config['num_targets'], tuple):
            num_targets = np.random.randint(*config['num_targets'])
        else:
            num_targets = config['num_targets']
        
        targets = self._generate_balanced_targets(num_targets, category)
        
        # 5. SCENARIO METADATA
        scenario = {
            'scenario_id': scenario_id,
            'category': category,
            'difficulty': config['difficulty'],
            'radar_params': radar_params,
            'targets': targets,
            'environment': environment,
            'generation_strategy': 'systematic',
            'quality_target': self._define_quality_target(config['difficulty'])
        }
        
        return scenario
    
    def _generate_systematic_radar_params(self) -> Dict:
        """Generate radar parameters with systematic coverage"""
        
        # Define parameter grids for systematic coverage
        freq_bands = {
            'L': 1.5e9,
            'S': 3.0e9, 
            'C': 5.5e9,
            'X': 10e9,
           
        }
        
        # Select band and add realistic variation
        band_name = np.random.choice(list(freq_bands.keys()))
        base_freq = freq_bands[band_name]

        # First define fc
        fc = base_freq * np.random.uniform(0.9, 1.1)

        # Then use fc to calculate bandwidth
        bandwidth_ratio = np.random.uniform(0.05, 0.08)  # Was 0.15, now 0.08
        B = fc * bandwidth_ratio
    
        
        # Sampling frequency - Nyquist + margin
        fs = B * np.random.uniform(2.2, 2.8)

         # 🔧 FIX: Hard safety limits
        if fs > 800e6:  # 800 MHz hard limit (well below simulation's breaking point)
            fs = 800e6

        if B > fc * 0.1:  # No more than 10% bandwidth
            B = fc * 0.1
            fs = B * 2.5  # Recalculate
        
        # Pulse parameters
        T_pulse = np.random.uniform(1e-6, 50e-6)
        
        # CPI length based on application
        cpi_options = {
            'air_search': [32, 64],
            'track': [16, 32], 
            'weather': [64, 128],
            'missile_defense': [8, 16]
        }
        application = np.random.choice(list(cpi_options.keys()))
        num_pulses = np.random.choice(cpi_options[application])
        
        # Platform type affects other parameters
        platform_type = np.random.choice(['Ground', 'Aerial'], p=[0.7, 0.3])
        
        radar_params = {
            'fc': fc,
            'fs': fs,
            'T_pulse': T_pulse,
            'B': B,
            'num_pulses': num_pulses,
            'pfa': np.random.choice([1e-3, 1e-4, 1e-5, 1e-6]),
            'platform_type': platform_type,
            'band': band_name,
            'application': application
        }
        
        return radar_params
    
    def _generate_environment(self, env_config) -> Dict:
        """Generate realistic environment conditions"""
        
        if isinstance(env_config, list):
            env_type = np.random.choice(env_config)
        else:
            env_type = env_config
        
        # Environment-specific parameters
        env_params = {
            'clean': {
                'clutter_strength': np.random.uniform(0, 0.1),
                'weather_attenuation': 0,
                'interference_level': 0,
                'multipath_factor': np.random.uniform(0, 0.05)
            },
            'light_clutter': {
                'clutter_strength': np.random.uniform(0.1, 0.3),
                'weather_attenuation': np.random.uniform(0, 0.1),
                'interference_level': np.random.uniform(0, 0.05),
                'multipath_factor': np.random.uniform(0.05, 0.15)
            },
            'heavy_clutter': {
                'clutter_strength': np.random.uniform(0.3, 0.7),
                'weather_attenuation': np.random.uniform(0.1, 0.2),
                'interference_level': np.random.uniform(0.05, 0.1),
                'multipath_factor': np.random.uniform(0.15, 0.3)
            },
            'weather': {
                'clutter_strength': np.random.uniform(0.2, 0.5),
                'weather_attenuation': np.random.uniform(0.2, 0.5),
                'interference_level': np.random.uniform(0, 0.05),
                'multipath_factor': np.random.uniform(0.1, 0.25),
                'precipitation_rate': np.random.uniform(5, 50)  # mm/hr
            },
            'interference': {
                'clutter_strength': np.random.uniform(0.1, 0.3),
                'weather_attenuation': np.random.uniform(0, 0.1),
                'interference_level': np.random.uniform(0.1, 0.4),
                'multipath_factor': np.random.uniform(0.05, 0.2),
                'interference_type': np.random.choice(['jamming', 'comms', 'radar'])
            },
            'extreme_weather': {
                'clutter_strength': np.random.uniform(0.5, 0.8),
                'weather_attenuation': np.random.uniform(0.3, 0.8),
                'interference_level': np.random.uniform(0, 0.1),
                'multipath_factor': np.random.uniform(0.2, 0.4),
                'precipitation_rate': np.random.uniform(25, 100)
            },
            'jamming': {
                'clutter_strength': np.random.uniform(0.1, 0.4),
                'weather_attenuation': np.random.uniform(0, 0.2),
                'interference_level': np.random.uniform(0.3, 0.8),
                'multipath_factor': np.random.uniform(0.1, 0.3),
                'jammer_power': np.random.uniform(100, 1000),  # dBm
                'jammer_type': np.random.choice(['barrage', 'spot', 'sweep'])
            }
        }
        
        environment = {
            'type': env_type,
            **env_params.get(env_type, env_params['clean'])
        }
        
        return environment
    
    def _calculate_snr_variation(self, environment: Dict, radar_params: Dict) -> float:
        """Calculate physics-based SNR variation"""
        
        snr_variation = 0
        
        # Weather attenuation (frequency dependent)
        if environment.get('weather_attenuation', 0) > 0:
            freq_ghz = radar_params['fc'] / 1e9
            # Higher frequencies more affected by weather
            freq_factor = (freq_ghz / 10) ** 1.5
            weather_loss = environment['weather_attenuation'] * freq_factor * 10
            snr_variation -= weather_loss
        
        # Clutter effects
        clutter_strength = environment.get('clutter_strength', 0)
        if clutter_strength > 0.3:
            snr_variation -= clutter_strength * 8
        
        # Interference
        interference = environment.get('interference_level', 0)
        if interference > 0.1:
            snr_variation -= interference * 15
        
        return snr_variation
    
    def _generate_balanced_targets(self, num_targets: int, category: str) -> List[Dict]:
        """Generate balanced set of targets"""
        
        if num_targets == 0:
            return []
        
        targets = []
        
        # Select target types based on probability distribution
        target_types = list(self.target_specs.keys())
        probabilities = [spec['probability'] for spec in self.target_specs.values()]
        
        selected_types = np.random.choice(
            target_types, 
            size=num_targets, 
            p=probabilities,
            replace=True
        )
        
        # Ensure minimum separation between targets
        used_positions = []
        
        for i, target_type in enumerate(selected_types):
            target = self._generate_single_target(target_type, used_positions)
            target['target_id'] = i
            targets.append(target)
            used_positions.append((target['range'], target['velocity']))
        
        return targets
    
    def _generate_single_target(self, target_type: str, used_positions: List[Tuple]) -> Dict:
        """Generate single target with realistic parameters"""
        
        spec = self.target_specs[target_type]
        max_attempts = 100
        
        for attempt in range(max_attempts):
            # Generate parameters within spec ranges
            target_range = np.random.uniform(*spec['range_range'])
            velocity = np.random.uniform(*spec['velocity_range'])
            rcs = np.random.uniform(*spec['rcs_range'])
            
            # Check minimum separation from existing targets
            min_range_sep = 500  # meters
            min_velocity_sep = 10  # m/s
            
            too_close = False
            for used_range, used_velocity in used_positions:
                if (abs(target_range - used_range) < min_range_sep and 
                    abs(velocity - used_velocity) < min_velocity_sep):
                    too_close = True
                    break
            
            if not too_close:
                break
        
        # Direction and azimuth
        direction = np.random.choice(['approaching', 'receding'])
        azimuth = np.random.uniform(0, 360)
        
        # Add realistic parameter correlations
        if target_type.startswith('bird'):
            # Birds: velocity correlates with size/RCS
            velocity *= (1 + 0.3 * np.log10(rcs / spec['rcs_range'][0]))
        elif 'missile' in target_type:
            # Missiles: higher velocity at longer ranges
            velocity *= (1 + 0.2 * (target_range - spec['range_range'][0]) / 
                        (spec['range_range'][1] - spec['range_range'][0]))
        
        target = {
            'type': 'air',  # For simulation compatibility
            'category': target_type,
            'range': float(target_range),
            'velocity': float(velocity),
            'direction': direction,
            'azimuth': float(azimuth),
            'rcs': float(rcs),
            'altitude': np.random.uniform(*spec['altitude_range']),
            'maneuverability': spec['maneuverability'],
            'signature_stability': spec['signature_stability'],
            'micro_doppler': spec['micro_doppler']
        }
        
        return target
    
    def _define_quality_target(self, difficulty: str) -> Dict:
        """Define quality targets based on difficulty"""
        
        quality_targets = {
            'easy': {
                'min_detection_rate': 0.95,
                'max_false_alarm_rate': 0.02,
                'min_classification_accuracy': 0.90
            },
            'medium': {
                'min_detection_rate': 0.85,
                'max_false_alarm_rate': 0.05,
                'min_classification_accuracy': 0.80
            },
            'hard': {
                'min_detection_rate': 0.70,
                'max_false_alarm_rate': 0.08,
                'min_classification_accuracy': 0.70
            },
            'extreme': {
                'min_detection_rate': 0.50,
                'max_false_alarm_rate': 0.12,
                'min_classification_accuracy': 0.60
            }
        }
        
        return quality_targets.get(difficulty, quality_targets['medium'])
    
    def run_simulation_and_validate(self, scenario: Dict) -> Optional[Dict]:
        """Run simulation with comprehensive validation - FIXED VERSION"""
        
        try:
            # Convert to simulation format with CORRECT parameter mapping
            sim_params = self._scenario_to_sim_params(scenario)
            
            # 🔧 DEBUG: Print the parameters being sent to simulation
            print(f"🔧 DEBUG: Sending parameters to simulation:")
            for key, value in sim_params.items():
                if key != 'targets':  # Don't print large target arrays
                    print(f"   {key}: {value}")
                else:
                    print(f"   targets: {len(value)} targets")
            
            # Run simulation - this should now work!
            results = run_enhanced_prf_simulation(sim_params)
            
            # Quality validation
            if not self._validate_simulation_quality(scenario, results):
                self.logger.warning(f"Scenario {scenario['scenario_id']} failed quality check")
                return None
            
            # Extract training data
            training_sample = self._extract_enhanced_training_data(scenario, results)
            
            # Apply augmentation
            augmented_samples = self._apply_data_augmentation(training_sample)
            
            return {
                'original': training_sample,
                'augmented': augmented_samples
            }
            
        except Exception as e:
            self.logger.error(f"Scenario {scenario['scenario_id']} failed: {str(e)}")
            # 🔧 DEBUG: Print the full error traceback
            import traceback
            self.logger.error(f"Full traceback: {traceback.format_exc()}")
            return None
    
    def _validate_simulation_quality(self, scenario: Dict, results: Dict) -> bool:
        """Comprehensive quality validation"""
        
        quality_target = scenario['quality_target']
        
        # Check detection rate
        expected_detections = len(scenario['targets'])
        actual_detections = results.get('matched_targets_count', 0)
        
        if expected_detections > 0:
            detection_rate = actual_detections / expected_detections
            if detection_rate < quality_target['min_detection_rate']:
                return False
        
        # Check false alarm rate
        total_detections = results.get('detected_targets_count', 0)
        false_alarms = results.get('false_alarm_count', 0)
        
        if total_detections > 0:
            false_alarm_rate = false_alarms / total_detections
            if false_alarm_rate > quality_target['max_false_alarm_rate']:
                return False
        
        # Check for reasonable SNR distribution
        if results.get('prf_results'):
            for prf_type, prf_data in results['prf_results'].items():
                if prf_data.get('range_doppler_map') is not None:
                    rd_map = prf_data['range_doppler_map']
                    signal_power = np.mean(np.abs(rd_map)**2)
                    if signal_power < 1e-20:  # Unreasonably low
                        return False
        
        return True
    
    def _apply_data_augmentation(self, training_sample: Dict) -> List[Dict]:
        """Apply sophisticated data augmentation"""
        
        augmented_samples = []
        
        # 1. SNR variation (simulate different weather/range conditions)
        for snr_offset in [-5, -2, 2, 5]:
            aug_sample = self._augment_snr(training_sample, snr_offset)
            if aug_sample:
                augmented_samples.append(aug_sample)
        
        # 2. Slight frequency shifts (simulate Doppler uncertainty)
        for freq_shift in [-0.02, 0.02]:  # ±2%
            aug_sample = self._augment_frequency(training_sample, freq_shift)
            if aug_sample:
                augmented_samples.append(aug_sample)
        
        # 3. Range bin shifts (simulate timing uncertainty)
        for range_shift in [-2, -1, 1, 2]:  # bins
            aug_sample = self._augment_range_shift(training_sample, range_shift)
            if aug_sample:
                augmented_samples.append(aug_sample)
        
        return augmented_samples[:6]  # Limit augmentation
    
    def _augment_snr(self, sample: Dict, snr_offset: float) -> Optional[Dict]:
        """Augment by varying SNR"""
        # Implementation would modify the range-doppler maps by adding/scaling noise
        # This is a simplified placeholder
        aug_sample = sample.copy()
        aug_sample['augmentation'] = {'type': 'snr', 'offset': snr_offset}
        return aug_sample
    
    def _augment_frequency(self, sample: Dict, freq_shift: float) -> Optional[Dict]:
        """Augment by shifting Doppler frequencies"""
        # Implementation would shift the Doppler axis
        aug_sample = sample.copy()
        aug_sample['augmentation'] = {'type': 'frequency', 'shift': freq_shift}
        return aug_sample
    
    def _augment_range_shift(self, sample: Dict, range_shift: int) -> Optional[Dict]:
        """Augment by shifting range bins"""
        # Implementation would shift the range axis
        aug_sample = sample.copy()
        aug_sample['augmentation'] = {'type': 'range', 'shift': range_shift}
        return aug_sample
    
    def generate_elite_dataset(self, num_scenarios: int = 10000, 
                              use_parallel: bool = True) -> Tuple[int, int]:
        """Generate elite training dataset with all optimizations"""
        
        self.logger.info(f"🚀 Starting ELITE dataset generation: {num_scenarios} scenarios")
        
        # Generate systematic scenarios
        scenarios = self.generate_systematic_scenarios(num_scenarios)
        
        # Create train/val/test splits BEFORE processing
        splits = self._create_balanced_splits(scenarios)
        self._save_splits(splits)
        
        # Process scenarios
        if use_parallel:
            successful, failed = self._process_scenarios_parallel(scenarios)
        else:
            successful, failed = self._process_scenarios_sequential(scenarios)
        
        # Generate comprehensive reports
        self._generate_quality_report(scenarios, successful, failed)
        self._generate_dataset_statistics()
        
        self.logger.info(f"✅ ELITE dataset generation complete!")
        self.logger.info(f"📊 Final: {successful}/{num_scenarios} high-quality samples")
        
        return successful, failed
    
    def _create_balanced_splits(self, scenarios: List[Dict]) -> Dict:
        """Create balanced train/validation/test splits"""
        
        # Group by category and difficulty
        grouped_scenarios = {}
        for scenario in scenarios:
            key = (scenario['category'], scenario['difficulty'])
            if key not in grouped_scenarios:
                grouped_scenarios[key] = []
            grouped_scenarios[key].append(scenario)
        
        # Create splits maintaining balance
        train_ids, val_ids, test_ids = [], [], []
        
        for group_scenarios in grouped_scenarios.values():
            np.random.shuffle(group_scenarios)
            n = len(group_scenarios)
            
            n_test = int(n * self.test_split)
            n_val = int(n * self.validation_split)
            n_train = n - n_test - n_val
            
            train_ids.extend([s['scenario_id'] for s in group_scenarios[:n_train]])
            val_ids.extend([s['scenario_id'] for s in group_scenarios[n_train:n_train+n_val]])
            test_ids.extend([s['scenario_id'] for s in group_scenarios[n_train+n_val:]])
        
        return {
            'train': sorted(train_ids),
            'validation': sorted(val_ids),
            'test': sorted(test_ids)
        }
    
    def _save_splits(self, splits: Dict):
        """Save dataset splits"""
        for split_name, scenario_ids in splits.items():
            split_file = self.output_dir / "splits" / f"{split_name}_scenarios.txt"
            with open(split_file, 'w') as f:
                for scenario_id in scenario_ids:
                    f.write(f"{scenario_id}\n")
        
        self.logger.info(f"📋 Saved splits: Train={len(splits['train'])}, "
                        f"Val={len(splits['validation'])}, Test={len(splits['test'])}")
    
    def _process_scenarios_sequential(self, scenarios: List[Dict]) -> Tuple[int, int]:
        """Process scenarios sequentially with progress tracking"""
        
        successful = 0
        failed = 0
        
        for scenario in tqdm(scenarios, desc="Processing scenarios"):
            result = self.run_simulation_and_validate(scenario)
            
            if result:
                self._save_training_sample(result['original'])
                for i, aug_sample in enumerate(result['augmented']):
                    self._save_augmented_sample(aug_sample, i)
                successful += 1
            else:
                failed += 1
        
        return successful, failed
    
    def _process_scenarios_parallel(self, scenarios: List[Dict]) -> Tuple[int, int]:
        """Process scenarios in parallel for speed"""
        
        self.logger.info(f"🔄 Using parallel processing with {mp.cpu_count()} cores")
        
        # Split scenarios into chunks for parallel processing
        chunk_size = max(1, len(scenarios) // mp.cpu_count())
        scenario_chunks = [scenarios[i:i+chunk_size] for i in range(0, len(scenarios), chunk_size)]
        
        # Process chunks in parallel
        with mp.Pool(processes=mp.cpu_count()) as pool:
            results = list(tqdm(
                pool.imap(self._process_scenario_chunk, scenario_chunks),
                total=len(scenario_chunks),
                desc="Processing chunks"
            ))
        
        # Aggregate results
        successful = sum(r[0] for r in results)
        failed = sum(r[1] for r in results)
        
        return successful, failed
    
    def _process_scenario_chunk(self, scenarios: List[Dict]) -> Tuple[int, int]:
        """Process a chunk of scenarios"""
        successful = 0
        failed = 0
        
        for scenario in scenarios:
            result = self.run_simulation_and_validate(scenario)
            
            if result:
                self._save_training_sample(result['original'])
                for i, aug_sample in enumerate(result['augmented']):
                    self._save_augmented_sample(aug_sample, i)
                successful += 1
            else:
                failed += 1
        
        return successful, failed
    
    def _scenario_to_sim_params(self, scenario: Dict) -> Dict:
        """
        Convert scenario to simulation parameters format
        🔧 FIXED: Proper parameter mapping to match simulation expectations
        """
        
        # 🚨 THE FIX: Your simulation expects these EXACT parameter names
        return {
            # ✅ CORRECT MAPPING - simulation expects these names from Flask route
            'centerFreq': scenario['radar_params']['fc'],        # simulation uses params['fc'] 
            'samplingFreq': scenario['radar_params']['fs'],      # simulation uses params['fs']
            'pulseDuration': scenario['radar_params']['T_pulse'], # simulation uses params['T_pulse']
            'chirpBandwidth': scenario['radar_params']['B'],     # simulation uses params['B']
            'numPulses': scenario['radar_params']['num_pulses'], # simulation uses params['num_pulses']
            'snr': scenario['radar_params']['snr_db'],           # simulation uses params['snr_db']
            'pfa': scenario['radar_params']['pfa'],              # simulation uses params['pfa']
            
            # ✅ CFAR parameters - these are correctly mapped
            'guardCellsRange': 3,
            'guardCellsDoppler': 3,
            'trainingCellsRange': 8,
            'trainingCellsDoppler': 8,
            
            # ✅ Target and platform data
            'targets': scenario['targets'],
            'platformType': scenario['radar_params']['platform_type']
        }
    
    def _extract_enhanced_training_data(self, scenario: Dict, results: Dict) -> Dict:
        """Extract comprehensive training data with enhanced labels"""
        
        # Get range-doppler maps from all PRFs
        prf_data = {}
        for prf_type in ['low', 'medium', 'high']:
            if prf_type in results.get('prf_results', {}):
                prf_info = results['prf_results'][prf_type]
                if prf_info.get('range_doppler_map') is not None:
                    prf_data[prf_type] = {
                        'range_doppler_map': prf_info['range_doppler_map'],
                        'range_bins': prf_info['range_bins'],
                        'doppler_velocities': prf_info['doppler_velocities'],
                        'cfar_detections': prf_info.get('cfar_detections'),
                        'threshold_map': prf_info.get('threshold_map')
                    }
        
        # Enhanced detection labels with confidence maps
        detection_labels = self._create_enhanced_detection_labels(
            scenario['targets'], prf_data, results
        )
        
        # Multi-class classification labels
        classification_labels = self._create_enhanced_classification_labels(
            scenario['targets'], results.get('target_matches', [])
        )
        
        # Regression labels for parameter estimation
        parameter_labels = self._create_enhanced_parameter_labels(
            scenario['targets'], results
        )
        
        # Physics-informed auxiliary labels
        auxiliary_labels = self._create_auxiliary_labels(scenario, results)
        
        training_sample = {
            'scenario_id': scenario['scenario_id'],
            'category': scenario['category'],
            'difficulty': scenario['difficulty'],
            'inputs': {
                'range_doppler_maps': prf_data,
                'radar_params': scenario['radar_params'],
                'environment': scenario['environment']
            },
            'labels': {
                'detections': detection_labels,
                'classifications': classification_labels,
                'parameters': parameter_labels,
                'auxiliary': auxiliary_labels
            },
            'metadata': {
                'num_targets': len(scenario['targets']),
                'target_types': [t['category'] for t in scenario['targets']],
                'simulation_results': {
                    'detected_targets_count': results.get('detected_targets_count', 0),
                    'matched_targets_count': results.get('matched_targets_count', 0),
                    'false_alarm_count': results.get('false_alarm_count', 0),
                    'detection_rate': results.get('detection_rate_percent', 0) / 100,
                    'multi_prf_detections': results.get('multi_prf_detections_count', 0)
                },
                'quality_metrics': self._calculate_quality_metrics(scenario, results)
            }
        }
        
        return training_sample
    
    def _create_enhanced_detection_labels(self, true_targets: List[Dict], 
                                        prf_data: Dict, results: Dict) -> Dict:
        """Create enhanced detection labels with confidence and multi-scale information"""
        
        detection_labels = {}
        
        for prf_type, data in prf_data.items():
            if data['range_doppler_map'] is None:
                continue
                
            range_bins = data['range_bins']
            doppler_velocities = data['doppler_velocities']
            map_shape = data['range_doppler_map'].shape
            
            # Multi-scale detection maps
            detection_maps = {
                'point_targets': np.zeros(map_shape, dtype=np.float32),  # Precise locations
                'blob_targets': np.zeros(map_shape, dtype=np.float32),   # Realistic spread
                'confidence_map': np.zeros(map_shape, dtype=np.float32), # Detection confidence
                'target_class_map': np.zeros(map_shape, dtype=np.int32)  # Per-pixel class
            }
            
            for target_idx, target in enumerate(true_targets):
                # Convert target parameters
                target_range = target['range']
                target_velocity = target['velocity']
                if target['direction'] == 'approaching':
                    target_velocity = -abs(target_velocity)
                else:
                    target_velocity = abs(target_velocity)
                
                # Find bin indices
                range_idx = np.argmin(np.abs(range_bins - target_range))
                velocity_idx = np.argmin(np.abs(doppler_velocities - target_velocity))
                
                # Point target (single pixel)
                if (0 <= velocity_idx < map_shape[0] and 0 <= range_idx < map_shape[1]):
                    detection_maps['point_targets'][velocity_idx, range_idx] = 1.0
                
                # Blob target (realistic spread based on target type)
                blob_size = self._get_target_blob_size(target['category'])
                confidence = self._get_target_confidence(target, prf_type)
                
                for di in range(-blob_size, blob_size + 1):
                    for ri in range(-blob_size, blob_size + 1):
                        v_idx = velocity_idx + di
                        r_idx = range_idx + ri
                        
                        if (0 <= v_idx < map_shape[0] and 0 <= r_idx < map_shape[1]):
                            # Gaussian-like weight
                            distance = np.sqrt(di**2 + ri**2)
                            weight = np.exp(-distance**2 / (2 * (blob_size/2)**2))
                            
                            detection_maps['blob_targets'][v_idx, r_idx] = max(
                                detection_maps['blob_targets'][v_idx, r_idx], 
                                weight
                            )
                            
                            detection_maps['confidence_map'][v_idx, r_idx] = max(
                                detection_maps['confidence_map'][v_idx, r_idx],
                                confidence * weight
                            )
                            
                            # Class map (using target type index)
                            class_mapping = {
                                'fighter_jet': 1, 'commercial_aircraft': 2, 'helicopter': 3,
                                'cruise_missile': 4, 'drone_small': 5, 'drone_large': 6,
                                'bird_large': 7, 'bird_small': 8
                            }
                            class_idx = class_mapping.get(target['category'], 0)
                            if weight > 0.5:  # Only assign class to strong detections
                                detection_maps['target_class_map'][v_idx, r_idx] = class_idx
            
            detection_labels[prf_type] = detection_maps
        
        return detection_labels
    
    def _get_target_blob_size(self, target_category: str) -> int:
        """Get appropriate blob size for target category"""
        blob_sizes = {
            'fighter_jet': 4,
            'commercial_aircraft': 6,
            'helicopter': 5,
            'cruise_missile': 3,
            'drone_small': 2,
            'drone_large': 3,
            'bird_large': 2,
            'bird_small': 1
        }
        return blob_sizes.get(target_category, 3)
    
    def _get_target_confidence(self, target: Dict, prf_type: str) -> float:
        """Calculate target detection confidence based on parameters"""
        
        # Base confidence from RCS
        rcs = target['rcs']
        if rcs > 10:
            base_confidence = 0.95
        elif rcs > 1:
            base_confidence = 0.85
        elif rcs > 0.1:
            base_confidence = 0.75
        else:
            base_confidence = 0.6
        
        # PRF-specific adjustments
        prf_adjustments = {
            'low': {'velocity_factor': 0.8, 'range_factor': 1.0},
            'medium': {'velocity_factor': 0.9, 'range_factor': 0.9},
            'high': {'velocity_factor': 1.0, 'range_factor': 0.7}
        }
        
        adj = prf_adjustments.get(prf_type, {'velocity_factor': 0.9, 'range_factor': 0.9})
        
        # Velocity-based adjustment
        velocity = abs(target['velocity'])
        if velocity < 10:
            velocity_conf = 0.7  # Slow targets harder to detect
        elif velocity > 300:
            velocity_conf = adj['velocity_factor']
        else:
            velocity_conf = 0.9
        
        # Range-based adjustment
        target_range = target['range']
        if target_range > 50000:
            range_conf = adj['range_factor'] * 0.8
        elif target_range < 1000:
            range_conf = adj['range_factor'] * 0.9
        else:
            range_conf = adj['range_factor']
        
        final_confidence = base_confidence * velocity_conf * range_conf
        return np.clip(final_confidence, 0.1, 1.0)
    
    def _create_enhanced_classification_labels(self, true_targets: List[Dict], 
                                             matches: List[Dict]) -> Dict:
        """Create enhanced classification labels with multiple label types"""
        
        # Comprehensive class mapping
        class_hierarchy = {
            'fighter_jet': {'main': 0, 'vehicle_type': 0, 'size': 2, 'threat': 3},
            'commercial_aircraft': {'main': 1, 'vehicle_type': 0, 'size': 3, 'threat': 0},
            'helicopter': {'main': 2, 'vehicle_type': 1, 'size': 2, 'threat': 1},
            'cruise_missile': {'main': 3, 'vehicle_type': 2, 'size': 1, 'threat': 4},
            'drone_small': {'main': 4, 'vehicle_type': 3, 'size': 0, 'threat': 1},
            'drone_large': {'main': 5, 'vehicle_type': 3, 'size': 1, 'threat': 2},
            'bird_large': {'main': 6, 'vehicle_type': 4, 'size': 0, 'threat': 0},
            'bird_small': {'main': 7, 'vehicle_type': 4, 'size': 0, 'threat': 0}
        }
        
        classification_labels = {
            'main_class': [],      # Primary classification
            'vehicle_type': [],    # Aircraft/Rotorcraft/Missile/UAV/Bird
            'size_class': [],      # Size category
            'threat_level': [],    # Threat assessment
            'characteristics': []   # Physical characteristics
        }
        
        for target in true_targets:
            target_class = target.get('category', 'unknown')
            
            if target_class in class_hierarchy:
                hierarchy = class_hierarchy[target_class]
                
                classification_labels['main_class'].append({
                    'class_idx': hierarchy['main'],
                    'class_name': target_class,
                    'confidence': 1.0
                })
                
                classification_labels['vehicle_type'].append(hierarchy['vehicle_type'])
                classification_labels['size_class'].append(hierarchy['size'])
                classification_labels['threat_level'].append(hierarchy['threat'])
                
                # Physical characteristics
                characteristics = {
                    'velocity': target['velocity'],
                    'rcs': target['rcs'],
                    'maneuverability': target.get('maneuverability', 'unknown'),
                    'signature_stability': target.get('signature_stability', 'unknown'),
                    'micro_doppler': target.get('micro_doppler', 'minimal')
                }
                classification_labels['characteristics'].append(characteristics)
            else:
                # Unknown target
                classification_labels['main_class'].append({
                    'class_idx': 8,  # Unknown class
                    'class_name': 'unknown',
                    'confidence': 0.5
                })
                classification_labels['vehicle_type'].append(5)  # Unknown type
                classification_labels['size_class'].append(2)   # Medium size default
                classification_labels['threat_level'].append(2) # Medium threat default
                classification_labels['characteristics'].append({})
        
        return classification_labels
    
    def _create_enhanced_parameter_labels(self, true_targets: List[Dict], 
                                        results: Dict) -> Dict:
        """Create enhanced parameter labels for regression tasks"""
        
        parameter_labels = {
            'targets': [],
            'scene_parameters': {},
            'performance_metrics': {}
        }
        
        # Per-target parameters
        for target in true_targets:
            target_params = {
                'kinematics': {
                    'range': target['range'],
                    'velocity': target['velocity'],
                    'azimuth': target.get('azimuth', 0),
                    'elevation': target.get('altitude', 1000) / target['range']  # Elevation angle
                },
                'signature': {
                    'rcs': target['rcs'],
                    'rcs_log': np.log10(max(target['rcs'], 1e-6)),
                    'estimated_snr': None,  # Will be filled from simulation
                    'signature_stability': target.get('signature_stability', 'stable')
                },
                'classification_confidence': {
                    'velocity_confidence': self._calculate_velocity_confidence(target),
                    'rcs_confidence': self._calculate_rcs_confidence(target),
                    'overall_confidence': None  # Computed later
                }
            }
            parameter_labels['targets'].append(target_params)
        
        # Scene-level parameters
        parameter_labels['scene_parameters'] = {
            'num_targets': len(true_targets),
            'target_density': len(true_targets) / (50000 * 1000),  # targets per m²
            'range_spread': self._calculate_range_spread(true_targets),
            'velocity_spread': self._calculate_velocity_spread(true_targets),
            'rcs_distribution': self._calculate_rcs_distribution(true_targets)
        }
        
        # Performance prediction labels
        parameter_labels['performance_metrics'] = {
            'predicted_detection_rate': None,  # Based on SNR and clutter
            'predicted_false_alarm_rate': None,
            'predicted_classification_accuracy': None,
            'computational_complexity': None
        }
        
        return parameter_labels
    
    def _calculate_velocity_confidence(self, target: Dict) -> float:
        """Calculate confidence in velocity measurement"""
        velocity = abs(target['velocity'])
        
        if velocity < 5:
            return 0.6  # Very slow targets hard to distinguish from clutter
        elif velocity < 20:
            return 0.8
        elif velocity > 500:
            return 0.9  # Fast targets easy to detect
        else:
            return 0.85
    
    def _calculate_rcs_confidence(self, target: Dict) -> float:
        """Calculate confidence in RCS-based detection"""
        rcs = target['rcs']
        
        if rcs < 0.01:
            return 0.5  # Very small RCS
        elif rcs < 0.1:
            return 0.7
        elif rcs < 1:
            return 0.8
        elif rcs < 10:
            return 0.9
        else:
            return 0.95  # Large RCS
    
    def _calculate_range_spread(self, targets: List[Dict]) -> float:
        """Calculate spread of target ranges"""
        if len(targets) < 2:
            return 0
        
        ranges = [t['range'] for t in targets]
        return np.std(ranges) / np.mean(ranges)  # Coefficient of variation
    
    def _calculate_velocity_spread(self, targets: List[Dict]) -> float:
        """Calculate spread of target velocities"""
        if len(targets) < 2:
            return 0
        
        velocities = [abs(t['velocity']) for t in targets]
        return np.std(velocities) / (np.mean(velocities) + 1e-6)
    
    def _calculate_rcs_distribution(self, targets: List[Dict]) -> Dict:
        """Calculate RCS distribution statistics"""
        if not targets:
            return {'mean_log_rcs': 0, 'std_log_rcs': 0, 'dynamic_range': 0}
        
        rcs_values = [t['rcs'] for t in targets]
        log_rcs = [np.log10(max(rcs, 1e-6)) for rcs in rcs_values]
        
        return {
            'mean_log_rcs': np.mean(log_rcs),
            'std_log_rcs': np.std(log_rcs),
            'dynamic_range': np.log10(max(rcs_values) / min(rcs_values)) if min(rcs_values) > 0 else 0
        }
    
    def _create_auxiliary_labels(self, scenario: Dict, results: Dict) -> Dict:
        """Create physics-informed auxiliary labels"""
        
        auxiliary = {
            'physics_constraints': {},
            'detection_physics': {},
            'environment_effects': {}
        }
        
        # Physics constraints
        radar_params = scenario['radar_params']
        auxiliary['physics_constraints'] = {
            'max_unambiguous_range': 3e8 / (2 * radar_params.get('prf', 2000)),
            'max_unambiguous_velocity': 3e8 * radar_params.get('prf', 2000) / (4 * radar_params['fc']),
            'range_resolution': 3e8 / (2 * radar_params['B']),
            'velocity_resolution': 3e8 / (2 * radar_params['fc'] * radar_params['num_pulses'] / radar_params.get('prf', 2000))
        }
        
        # Detection physics
        auxiliary['detection_physics'] = {
            'thermal_noise_level': self._calculate_thermal_noise(radar_params),
            'clutter_level': self._estimate_clutter_level(scenario['environment']),
            'interference_level': scenario['environment'].get('interference_level', 0)
        }
        
        # Environment effects
        auxiliary['environment_effects'] = {
            'weather_attenuation': scenario['environment'].get('weather_attenuation', 0),
            'multipath_factor': scenario['environment'].get('multipath_factor', 0),
            'atmospheric_loss': self._calculate_atmospheric_loss(radar_params)
        }
        
        return auxiliary
    
    def _calculate_thermal_noise(self, radar_params: Dict) -> float:
        """Calculate thermal noise level"""
        # Simplified thermal noise calculation
        k_boltzmann = 1.38e-23
        temperature = 290  # K
        bandwidth = radar_params['B']
        
        thermal_noise_power = k_boltzmann * temperature * bandwidth
        return 10 * np.log10(thermal_noise_power)  # dBW
    
    def _estimate_clutter_level(self, environment: Dict) -> float:
        """Estimate clutter power level"""
        clutter_strength = environment.get('clutter_strength', 0)
        
        # Convert to dB relative to thermal noise
        if clutter_strength < 0.1:
            return -20  # Very low clutter
        elif clutter_strength < 0.3:
            return -10  # Light clutter
        elif clutter_strength < 0.6:
            return 0    # Moderate clutter
        else:
            return 10   # Heavy clutter
    
    def _calculate_atmospheric_loss(self, radar_params: Dict) -> float:
        """Calculate atmospheric propagation loss"""
        freq_ghz = radar_params['fc'] / 1e9
        
        # Simplified atmospheric loss (clear weather)
        if freq_ghz < 10:
            return 0.1 * freq_ghz  # dB/km
        else:
            return 0.1 * freq_ghz + 0.01 * (freq_ghz - 10)**2
    
    def _calculate_quality_metrics(self, scenario: Dict, results: Dict) -> Dict:
        """Calculate comprehensive quality metrics"""
        
        return {
            'scenario_difficulty_score': self._calculate_difficulty_score(scenario),
            'simulation_realism_score': self._calculate_realism_score(results),
            'data_completeness_score': self._calculate_completeness_score(results),
            'label_quality_score': self._calculate_label_quality_score(scenario, results)
        }
    
    def _calculate_difficulty_score(self, scenario: Dict) -> float:
        """Calculate scenario difficulty score (0-1)"""
        
        score = 0
        
        # SNR contribution
        snr = scenario['radar_params']['snr_db']
        if snr < 10:
            score += 0.3
        elif snr < 20:
            score += 0.2
        else:
            score += 0.1
        
        # Number of targets
        num_targets = len(scenario['targets'])
        if num_targets > 3:
            score += 0.3
        elif num_targets > 1:
            score += 0.2
        else:
            score += 0.1
        
        # Environment
        env_type = scenario['environment']['type']
        env_scores = {
            'clean': 0.1,
            'light_clutter': 0.2,
            'heavy_clutter': 0.3,
            'weather': 0.25,
            'interference': 0.35,
            'extreme_weather': 0.4,
            'jamming': 0.4
        }
        score += env_scores.get(env_type, 0.2)
        
        return min(score, 1.0)
    
    def _calculate_realism_score(self, results: Dict) -> float:
        """Calculate simulation realism score"""
        
        # Check for realistic detection patterns
        detection_rate = results.get('detection_rate_percent', 0) / 100
        false_alarm_rate = results.get('false_alarm_count', 0) / max(1, results.get('detected_targets_count', 1))
        
        realism_score = 1.0
        
        # Penalize unrealistic detection rates
        if detection_rate > 0.98:  # Too perfect
            realism_score *= 0.8
        if false_alarm_rate > 0.3:  # Too many false alarms
            realism_score *= 0.7
        
        return realism_score
    
    def _calculate_completeness_score(self, results: Dict) -> float:
        """Calculate data completeness score"""
        
        score = 1.0
        
        # Check for missing data
        required_fields = ['prf_results', 'target_detections', 'target_matches']
        for field in required_fields:
            if field not in results or not results[field]:
                score *= 0.7
        
        return score
    
    def _calculate_label_quality_score(self, scenario: Dict, results: Dict) -> float:
        """Calculate label quality score"""
        
        # Check label consistency
        expected_targets = len(scenario['targets'])
        detected_targets = results.get('matched_targets_count', 0)
        
        if expected_targets == 0:
            return 1.0 if detected_targets == 0 else 0.8
        
        label_quality = detected_targets / expected_targets
        return min(label_quality, 1.0)
    
    def _save_training_sample(self, training_sample: Dict):
        """Save training sample with optimized storage"""
        
        scenario_id = training_sample['scenario_id']
        
        # Save large arrays as compressed HDF5
        h5_path = self.output_dir / "range_doppler_maps" / f"scenario_{scenario_id:06d}.h5"
        with h5py.File(h5_path, 'w') as f:
            # Use compression for large arrays
            for prf_type, data in training_sample['inputs']['range_doppler_maps'].items():
                grp = f.create_group(prf_type)
                
                if data['range_doppler_map'] is not None:
                    # Store complex data as separate real/imaginary arrays
                    rd_map = data['range_doppler_map']
                    grp.create_dataset('range_doppler_real', data=rd_map.real, 
                                     compression='gzip', compression_opts=6)
                    grp.create_dataset('range_doppler_imag', data=rd_map.imag,
                                     compression='gzip', compression_opts=6)
                    
                    grp.create_dataset('range_bins', data=data['range_bins'],
                                     compression='gzip', compression_opts=6)
                    grp.create_dataset('doppler_velocities', data=data['doppler_velocities'],
                                     compression='gzip', compression_opts=6)
                    
                    # Store detection labels
                    detection_labels = training_sample['labels']['detections'].get(prf_type, {})
                    for label_type, label_data in detection_labels.items():
                        grp.create_dataset(f'labels_{label_type}', data=label_data,
                                         compression='gzip', compression_opts=6)
        
        # Save metadata and small labels as JSON
        json_data = {
            'scenario_id': scenario_id,
            'category': training_sample['category'],
            'difficulty': training_sample['difficulty'],
            'labels': {
                'classifications': training_sample['labels']['classifications'],
                'parameters': training_sample['labels']['parameters'],
                'auxiliary': training_sample['labels']['auxiliary']
            },
            'metadata': training_sample['metadata'],
            'radar_params': training_sample['inputs']['radar_params'],
            'environment': training_sample['inputs']['environment']
        }
        
        json_path = self.output_dir / "labels" / f"scenario_{scenario_id:06d}.json"
        with open(json_path, 'w') as f:
            json.dump(json_data, f, indent=2, default=str)
    
    def _save_augmented_sample(self, aug_sample: Dict, aug_idx: int):
        """Save augmented sample"""
        scenario_id = aug_sample['scenario_id']
        aug_path = self.output_dir / "augmented" / f"scenario_{scenario_id:06d}_aug_{aug_idx:02d}.json"
        
        with open(aug_path, 'w') as f:
            json.dump(aug_sample, f, indent=2, default=str)
    
    def _generate_quality_report(self, scenarios: List[Dict], 
                               successful: int, failed: int):
        """Generate comprehensive quality report"""
        
        report = {
            'generation_summary': {
                'total_scenarios': len(scenarios),
                'successful_samples': successful,
                'failed_samples': failed,
                'success_rate': successful / len(scenarios) if scenarios else 0,
                'generation_date': datetime.now().isoformat()
            },
            'scenario_distribution': self._analyze_scenario_distribution(scenarios),
            'target_distribution': self._analyze_target_distribution(scenarios),
            'difficulty_distribution': self._analyze_difficulty_distribution(scenarios),
            'quality_metrics': {
                'expected_performance': self._estimate_expected_performance(scenarios),
                'data_coverage': self._calculate_data_coverage(scenarios)
            }
        }
        
        report_path = self.output_dir / "quality_reports" / f"quality_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        self.logger.info(f"📊 Quality report saved: {report_path}")
    
    def _analyze_scenario_distribution(self, scenarios: List[Dict]) -> Dict:
        """Analyze distribution of scenario categories"""
        
        category_counts = {}
        for scenario in scenarios:
            category = scenario['category']
            category_counts[category] = category_counts.get(category, 0) + 1
        
        return category_counts
    
    def _analyze_target_distribution(self, scenarios: List[Dict]) -> Dict:
        """Analyze distribution of target types"""
        
        target_counts = {}
        total_targets = 0
        
        for scenario in scenarios:
            for target in scenario['targets']:
                target_type = target['category']
                target_counts[target_type] = target_counts.get(target_type, 0) + 1
                total_targets += 1
        
        # Convert to percentages
        target_percentages = {
            target_type: (count / total_targets * 100) if total_targets > 0 else 0
            for target_type, count in target_counts.items()
        }
        
        return {
            'counts': target_counts,
            'percentages': target_percentages,
            'total_targets': total_targets
        }
    
    def _analyze_difficulty_distribution(self, scenarios: List[Dict]) -> Dict:
        """Analyze distribution of difficulty levels"""
        
        difficulty_counts = {}
        for scenario in scenarios:
            difficulty = scenario['difficulty']
            difficulty_counts[difficulty] = difficulty_counts.get(difficulty, 0) + 1
        
        return difficulty_counts
    
    def _estimate_expected_performance(self, scenarios: List[Dict]) -> Dict:
        """Estimate expected ML model performance on this dataset"""
        
        # Analyze scenario complexity
        easy_scenarios = sum(1 for s in scenarios if s['difficulty'] == 'easy')
        medium_scenarios = sum(1 for s in scenarios if s['difficulty'] == 'medium')
        hard_scenarios = sum(1 for s in scenarios if s['difficulty'] == 'hard')
        extreme_scenarios = sum(1 for s in scenarios if s['difficulty'] == 'extreme')
        
        total = len(scenarios)
        
        # Estimate performance based on difficulty distribution
        expected_detection_rate = (
            (easy_scenarios / total) * 0.95 +
            (medium_scenarios / total) * 0.85 +
            (hard_scenarios / total) * 0.70 +
            (extreme_scenarios / total) * 0.50
        ) if total > 0 else 0
        
        expected_classification_accuracy = (
            (easy_scenarios / total) * 0.90 +
            (medium_scenarios / total) * 0.80 +
            (hard_scenarios / total) * 0.70 +
            (extreme_scenarios / total) * 0.60
        ) if total > 0 else 0
        
        return {
            'expected_detection_rate': expected_detection_rate,
            'expected_classification_accuracy': expected_classification_accuracy,
            'difficulty_balance': {
                'easy': easy_scenarios / total if total > 0 else 0,
                'medium': medium_scenarios / total if total > 0 else 0,
                'hard': hard_scenarios / total if total > 0 else 0,
                'extreme': extreme_scenarios / total if total > 0 else 0
            }
        }
    
    def _calculate_data_coverage(self, scenarios: List[Dict]) -> Dict:
        """Calculate parameter space coverage"""
        
        # Extract parameter ranges
        snr_values = [s['radar_params']['snr_db'] for s in scenarios]
        freq_values = [s['radar_params']['fc'] for s in scenarios]
        num_targets = [len(s['targets']) for s in scenarios]
        
        # Calculate coverage metrics
        coverage = {
            'snr_range': {
                'min': min(snr_values) if snr_values else 0,
                'max': max(snr_values) if snr_values else 0,
                'std': np.std(snr_values) if snr_values else 0
            },
            'frequency_range': {
                'min': min(freq_values) if freq_values else 0,
                'max': max(freq_values) if freq_values else 0,
                'std': np.std(freq_values) if freq_values else 0
            },
            'target_count_distribution': {
                'min': min(num_targets) if num_targets else 0,
                'max': max(num_targets) if num_targets else 0,
                'mean': np.mean(num_targets) if num_targets else 0
            }
        }
        
        return coverage
    
    def _generate_dataset_statistics(self):
        """Generate comprehensive dataset statistics"""
        
        stats = {
            'file_statistics': self._calculate_file_statistics(),
            'data_size_analysis': self._calculate_data_sizes(),
            'storage_optimization': self._analyze_storage_efficiency()
        }
        
        stats_path = self.output_dir / "metadata" / "dataset_statistics.json"
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2, default=str)
        
        self.logger.info(f"📈 Dataset statistics saved: {stats_path}")
    
    def _calculate_file_statistics(self) -> Dict:
        """Calculate file count and distribution statistics"""
        
        h5_files = list((self.output_dir / "range_doppler_maps").glob("*.h5"))
        json_files = list((self.output_dir / "labels").glob("*.json"))
        aug_files = list((self.output_dir / "augmented").glob("*.json"))
        
        return {
            'h5_files': len(h5_files),
            'label_files': len(json_files),
            'augmented_files': len(aug_files),
            'total_samples': len(h5_files),
            'augmentation_ratio': len(aug_files) / max(1, len(h5_files))
        }
    
    def _calculate_data_sizes(self) -> Dict:
        """Calculate storage size statistics"""
        
        def get_dir_size(directory):
            return sum(f.stat().st_size for f in directory.rglob('*') if f.is_file())
        
        h5_size = get_dir_size(self.output_dir / "range_doppler_maps")
        labels_size = get_dir_size(self.output_dir / "labels")
        aug_size = get_dir_size(self.output_dir / "augmented")
        total_size = get_dir_size(self.output_dir)
        
        return {
            'range_doppler_maps_mb': h5_size / (1024**2),
            'labels_mb': labels_size / (1024**2),
            'augmented_mb': aug_size / (1024**2),
            'total_size_mb': total_size / (1024**2),
            'avg_sample_size_mb': h5_size / max(1, len(list((self.output_dir / "range_doppler_maps").glob("*.h5")))) / (1024**2)
        }
    
    def _analyze_storage_efficiency(self) -> Dict:
        """Analyze storage efficiency and compression ratios"""
        
        # Sample a few files to estimate compression efficiency
        h5_files = list((self.output_dir / "range_doppler_maps").glob("*.h5"))
        
        if not h5_files:
            return {'compression_ratio': 0, 'efficiency_score': 0}
        
        # Check compression ratio on first file
        sample_file = h5_files[0]
        uncompressed_estimate = 0
        compressed_size = sample_file.stat().st_size
        
        try:
            with h5py.File(sample_file, 'r') as f:
                for prf_type in f.keys():
                    grp = f[prf_type]
                    for dataset_name in grp.keys():
                        dataset = grp[dataset_name]
                        uncompressed_estimate += dataset.size * 8  # Estimate 8 bytes per element
        except:
            uncompressed_estimate = compressed_size * 3  # Conservative estimate
        
        compression_ratio = uncompressed_estimate / max(1, compressed_size)
        
        return {
            'compression_ratio': compression_ratio,
            'efficiency_score': min(compression_ratio / 3.0, 1.0),  # Good compression is 3:1 or better
            'sample_file_size_mb': compressed_size / (1024**2)
        }

# USAGE EXAMPLES AND QUICK START
def create_quick_test_dataset():
    """Create a small test dataset for development"""
    
    generator = EliteRadarDataGenerator(output_dir="data/test_dataset")
    
    print("🧪 Creating test dataset (100 samples)...")
    successful, failed = generator.generate_elite_dataset(
        num_scenarios=100, 
        use_parallel=False
    )
    
    print(f"✅ Test dataset complete: {successful} successful samples")
    return generator

def create_development_dataset():
    """Create a medium dataset for development and validation"""
    
    generator = EliteRadarDataGenerator(output_dir="data/dev_dataset")
    
    print("🔧 Creating development dataset (1,000 samples)...")
    successful, failed = generator.generate_elite_dataset(
        num_scenarios=1000,
        use_parallel=True
    )
    
    print(f"✅ Development dataset complete: {successful} successful samples")
    return generator

def create_production_dataset():
    """Create a large dataset for production ML training"""
    
    generator = EliteRadarDataGenerator(output_dir="data/production_dataset")
    
    print("🚀 Creating production dataset (10,000 samples)...")
    successful, failed = generator.generate_elite_dataset(
        num_scenarios=10000,
        use_parallel=True
    )
    
    print(f"✅ Production dataset complete: {successful} successful samples")
    return generator

def create_massive_dataset():
    """Create a massive dataset for advanced ML research"""
    
    generator = EliteRadarDataGenerator(output_dir="data/massive_dataset")
    
    print("🌟 Creating massive dataset (50,000 samples)...")
    successful, failed = generator.generate_elite_dataset(
        num_scenarios=50000,
        use_parallel=True
    )
    
    print(f"✅ Massive dataset complete: {successful} successful samples")
    return generator

# MAIN EXECUTION
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Elite Radar Training Data Generator')
    parser.add_argument('--mode', choices=['test', 'dev', 'production', 'massive'], 
                       default='test', help='Dataset size mode')
    parser.add_argument('--output_dir', type=str, default=None, 
                       help='Custom output directory')
    parser.add_argument('--num_scenarios', type=int, default=None,
                       help='Custom number of scenarios')
    parser.add_argument('--parallel', action='store_true', 
                       help='Use parallel processing')
    
    args = parser.parse_args()
    
    # Setup based on mode
    if args.mode == 'test':
        output_dir = args.output_dir or "data/test_dataset"
        num_scenarios = args.num_scenarios or 5
    elif args.mode == 'dev':
        output_dir = args.output_dir or "data/dev_dataset"
        num_scenarios = args.num_scenarios or 100
    elif args.mode == 'production':
        output_dir = args.output_dir or "data/production_dataset"
        num_scenarios = args.num_scenarios or 1000
    elif args.mode == 'massive':
        output_dir = args.output_dir or "data/massive_dataset"
        num_scenarios = args.num_scenarios or 5000
    
    # Create generator and run
    generator = EliteRadarDataGenerator(output_dir=output_dir)
    
    print(f"🎯 Starting {args.mode} dataset generation...")
    print(f"📁 Output: {output_dir}")
    print(f"🎲 Scenarios: {num_scenarios}")
    print(f"⚡ Parallel: {args.parallel}")
    
    successful, failed = generator.generate_elite_dataset(
        num_scenarios=num_scenarios,
        use_parallel=args.parallel
    )
    
    print(f"\n🎉 ELITE DATASET GENERATION COMPLETE!")
    print(f"✅ Successful: {successful}")
    print(f"❌ Failed: {failed}")
    print(f"📊 Success Rate: {successful/(successful+failed)*100:.1f}%")
    print(f"💾 Data Location: {output_dir}")
    
    # Quick validation
    h5_files = len(list(Path(output_dir).glob("range_doppler_maps/*.h5")))
    json_files = len(list(Path(output_dir).glob("labels/*.json")))
    
    print(f"\n📋 DATASET VALIDATION:")
    print(f"🗂️  HDF5 files: {h5_files}")
    print(f"🏷️  Label files: {json_files}")
    print(f"✅ Data integrity: {'PASS' if h5_files == json_files else 'FAIL'}")
    
    if h5_files == json_files and h5_files > 0:
        print(f"\n🎊 Ready for ML training! Use the data loader to start training your models.")
    else:
        print(f"\n⚠️  Data integrity check failed. Please review the generation logs.")
