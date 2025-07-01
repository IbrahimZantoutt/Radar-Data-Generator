professional ML data factory

This is a radar data generator script using python , it allows its user to generate data that would be very difficult to accuire under normal circumstances.

used for model training and development then testing

structure:
data/production_dataset/
├── range_doppler_maps/     # 10,000 HDF5 files (~5GB compressed)
├── labels/                 # 10,000 JSON files (~500MB)
├── splits/                 # train/val/test splits
├── quality_reports/        # Performance analysis
└── metadata/              # Dataset statistics

offers:
1. Systematic Parameter Coverage (Not Random!)

Physics-informed scenarios based on real radar systems
Balanced target distribution using actual aircraft/missile characteristics
Systematic radar parameter grids covering all frequency bands
Environment-aware generation with realistic clutter/weather models

2. Advanced Label Generation

Multi-scale detection labels: Point targets, blob targets, confidence maps
Hierarchical classification: Main class, vehicle type, size, threat level
Physics-informed regression labels: SNR estimation, RCS prediction
Auxiliary labels: Physics constraints, environment effects

3. Quality Assurance System

Automatic quality validation for each scenario
Performance prediction based on scenario difficulty
Data integrity checks throughout generation
Comprehensive quality reports

4. Production-Ready Optimizations

Efficient HDF5 storage with compression (3x size reduction)
Balanced train/val/test splits maintaining category distribution
Parallel processing for 4-8x speed improvement
Memory-optimized processing for large datasets

5. Advanced Data Augmentation

SNR variation simulating different conditions
Frequency shifts for Doppler uncertainty
Range shifts for timing variations
Physics-aware augmentation (not just image transforms)

Perfect for ML Training
This generator creates exactly the right data for your Multi-Task Learning Network:

Detection Task: Perfect pixel-wise labels with confidence
Classification Task: Hierarchical labels (8 aircraft types)
Parameter Estimation: Velocity, RCS, range refinement
Physics Constraints: Built-in radar physics validation
