# 🐜 Multi-Objective Evolutionary Algorithms with Wasserstein

**MOEAW** is a research-oriented Python project that allows you to experiment with multiple **multi-objective evolutionary algorithms** on both benchmark functions and real-world problems such as:

- **Top-K Recommendation** systems
- **Optimal Sensor Placement (OSP)**

The project is built on top of the [pymoo](https://pymoo.org/) framework, and includes support for standard and novel algorithmic variants, including **Wasserstein-based approaches**.

## 🔍 Features

- Run benchmark experiments on:
  - DTLZ family
  - WFG family
  - DASCMOP (Constrained multi-objective)
- Solve real-world problems with:
  - Top-K Recommendation (e.g., MovieLens)
  - Optimal Sensor Placement (e.g., Hanoi, Neptun, etc.)
- Compare and analyze results for:
  - NSGA-II
  - MOEA/D
  - **NSGA-II/W** (Wasserstein-based)
  - **MOEA/D/W** (Wasserstein-based)
- Save results efficiently as compressed `.parquet` files

## 📁 Project Structure

```
.
├── run_experiment_benchmark.py  # For benchmark problems (DTLZ, WFG, DASCMOP)
├── run_experiment_rs.py         # For Top-K Recommendation experiments
├── run_experiment_osp.py        # For Optimal Sensor Placement experiments
├── .data/                       # Input data for RS and OSP problems
├── .results/                    # Output directory for experiment results
├── config/
│   ├── settings.json            # Benchmark experiment settings
│   ├── settings_rs.json         # RS experiment settings
│   └── settings_osp.json        # OSP experiment settings
├── algorithms/
│   ├── moead.py                 # Implementation of MOEA/D algorithms that support constraint
│   ├── moeadw.py                # Implementation of the Wasserstein-based MOEA/D algorithm
│   ├── sensor_crossover.py      # Implementation of a combinatorial binary crossover operator
│   └── wselection.py            # Implementation of a the NSGA-II/W Wasserstein-based selection operator
├── problems/
│   ├── osp.py                   # The Optimal Sensor Placement problem Pymoo implementation
│   └── top_k_rs.py              # The Top-K Recommendation Lists problem Pymoo implementation
└── experiment.py                # Core experiment runner and algorithm selector

````

## ⚙️ Setup

### Requirements

- Python 3.9+
- [pymoo](https://github.com/anyoptimization/pymoo)
- pandas
- pyarrow

### Install dependencies

```bash
pip install -r requirements.txt
````

## 🚀 Usage

All scripts can be launched from the command line with a few arguments:

```bash
python run_experiment_benchmark.py -a MOEAD -p DTLZ2 -t 10
python run_experiment_rs.py -a NSGA2W -p MovieLens1k -t 10
python run_experiment_osp.py -a MOEADW -p Hanoi -t 10
```

### Arguments

* `-a`, `--algorithm`: Algorithm to use (`NSGA2`, `MOEAD`, `NSGA2W`, `MOEADW`)
* `-p`, `--problem`: Problem name (e.g., `DTLZ2`, `MovieLens1k`, `Hanoi`)
* `-t`, `--trial`: Number of independent trials (default: `10`)

## 🧪 Supported Problems

### Benchmark Functions

* **DTLZ** (1–7)
* **WFG** (1–9)
* **DASCMOP** (1–9)

All implemented via `pymoo`.

### Real-World Problems

#### Top-K Recommendation

* Input: user-item rating matrix
* Goal: Find K-item lists optimizing multiple objectives (e.g., novelty, diversity)
* Example dataset: `MovieLens1k`

#### Optimal Sensor Placement

* Input: Detection time and volume contamination impact matrices
* Goal: Find sensor positions optimizing detection time, volume coverage, and cost
* Example dataset: `Hanoi`, `Anytown`, `Neptun` and `Apulian5`


## 📊 Results

All experiment results are saved in `.results/` as compressed `.parquet` files using [Brotli](https://brotli.org/) compression.

Filename pattern:

```
<problem>_<n_var>_<n_obj>[_difficulty|_budget|_k]_<algorithm>_<trial>.parquet
```
