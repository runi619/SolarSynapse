 ☀️ SolarSynapse
 Genetic Algorithm-Based Optimization of Solar Panel Tilt Angles

SolarSynapse is an evolutionary computing project that optimizes solar panel tilt angles to maximize estimated solar energy generation using Genetic Algorithms and PVGIS Typical Meteorological Year (TMY) weather data.

 Project Overview :

Solar panel tilt angle significantly influences the amount of solar irradiance received. SolarSynapse uses a Genetic Algorithm to identify an optimal tilt angle for a given location and evaluates seasonal variations in solar energy generation.

 Features :

- Genetic Algorithm-based solar tilt optimization
- PVGIS TMY weather data integration
- Solar irradiance calculation using pvlib
- Interactive Streamlit dashboard
- Adjustable GA parameters:
  - Population Size
  - Number of Generations
  - Mutation Rate
  - Crossover Rate
  - Mutation Strength (σ)
- GA convergence visualization
- Population diversity visualization
- Seasonal solar optimization:
  - Summer
  - Monsoon
  - Winter
- Seasonal energy comparison graphs
- GA, Grid Search, and Random Search comparison
- Statistical analysis across multiple runs

 Technologies Used :

- Python
- Streamlit
- NumPy
- Pandas
- Matplotlib
- pvlib
- Genetic Algorithms
- PVGIS TMY Weather Data

 Study Location :

Tiruchirappalli, Tamil Nadu, India

 Installation & Execution :

1. Clone the repository 

```bash
git clone https://github.com/runi619/SolarSynapse.git
cd SolarSynapse
```

 2. Install dependencies

```bash
pip install streamlit numpy pandas matplotlib pvlib
```

 3. Run the application

```bash
python -m streamlit run app.py
```

 Optimization Approach :

SolarSynapse uses a Genetic Algorithm involving:

1. Population Initialization
2. Fitness Evaluation
3. Tournament Selection
4. Crossover
5. Mutation
6. Elitism
7. Iterative Evolution

The fitness function estimates average daily solar energy generation based on plane-of-array irradiance.

 Seasonal Optimization :

The application evaluates solar panel tilt angles across different seasons and compares estimated daily energy generation.

Academic Project :

Developed as part of an Evolutionary Computing assignment to demonstrate Genetic Algorithm-based optimization and comparative analysis of evolutionary and conventional optimization methods.

 Author

**Tharunika**

 Repository:

[GitHub – SolarSynapse](https://github.com/runi619/SolarSynapse)
