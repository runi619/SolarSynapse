
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time

st.set_page_config(page_title="SolarSynapse", page_icon="☀️", layout="wide")

# ---------------- THEME ----------------
st.markdown("""
<style>
.stApp {background-color:#F8FAFC;color:#0F172A;}
[data-testid="stSidebar"] {background-color:#FFFFFF;}
div.stButton > button {
background-color:#F59E0B;color:white;border-radius:8px;
border:none;font-weight:600;
}
</style>
""", unsafe_allow_html=True)

st.title("☀️ SolarSynapse")
st.caption("Evolutionary Computing for Solar Energy Optimization")
st.markdown("**Location:** Tiruchirappalli, Tamil Nadu | **Tilt:** 0°–90°")
st.divider()


# ---------------- SEASONAL OPTIMIZATION ----------------

from solar_data import seasonal_results

st.header("🌤️ Seasonal Solar Optimization")

st.write(
    "Optimal solar panel tilt angles for different seasons "
    "based on PVGIS TMY weather data."
)

season_names = list(seasonal_results.keys())

cols = st.columns(len(season_names))

for col, season in zip(cols, season_names):

    result = seasonal_results[season]

    with col:
        st.metric(
            label=f"{season} Best Tilt",
            value=f"{result['angle']}°",
            delta=f"{result['energy']:.3f} kWh/day"
        )

# Seasonal comparison graph
st.subheader("Seasonal Energy Comparison")

fig, ax = plt.subplots(figsize=(8, 4))

for season, result in seasonal_results.items():
    ax.plot(
        result["angles"],
        result["energy_curve"],
        label=season
    )

ax.set_xlabel("Tilt Angle (degrees)")
ax.set_ylabel("Average Daily Energy (kWh/day)")
ax.set_title("Season-wise Energy vs Tilt Angle")
ax.legend()
ax.grid(alpha=0.3)

st.pyplot(fig, use_container_width=True)
plt.close(fig)


# ---------------- ACTUAL PVGIS ENERGY DATA ----------------
from solar_data import angles, energy

angles = np.array(angles, dtype=float)
energy = np.array(energy, dtype=float)

def fitness(angle):
    return float(np.interp(
        np.clip(angle, 0, 90),
        angles,
        energy
    ))

TARGET = float(np.max(energy))
TOLERANCE = 0.01


# ---------------- GENETIC ALGORITHM ----------------

# ---------------- IMPROVED GENETIC ALGORITHM ----------------
def run_ga(pop_size, generations, mutation_rate,
           crossover_rate, mutation_sigma, seed):

    rng = np.random.default_rng(seed)
    start = time.perf_counter()

    population = rng.uniform(0, 90, pop_size)
    scores = np.array([fitness(a) for a in population])

    best_idx = np.argmax(scores)
    best_angle = float(population[best_idx])
    best_score = float(scores[best_idx])

    history = []
    diversity = []
    populations = [population.copy()]

    for gen in range(generations):

        scores = np.array([fitness(a) for a in population])

        # Elitism
        idx = np.argmax(scores)
        if scores[idx] > best_score:
            best_score = float(scores[idx])
            best_angle = float(population[idx])

        new_population = [best_angle]

        # Tournament selection
        while len(new_population) < pop_size:

            ids = rng.choice(pop_size, 3, replace=False)
            p1 = population[ids[np.argmax(scores[ids])]]

            ids = rng.choice(pop_size, 3, replace=False)
            p2 = population[ids[np.argmax(scores[ids])]]

            # Crossover
            if rng.random() < crossover_rate:
                alpha = rng.random()
                child = alpha*p1 + (1-alpha)*p2
            else:
                child = p1

            # Mutation
            if rng.random() < mutation_rate:
                child += rng.normal(0, mutation_sigma)

            child = np.clip(child, 0, 90)
            new_population.append(float(child))

        population = np.array(new_population)

        scores = np.array([fitness(a) for a in population])
        idx = np.argmax(scores)

        if scores[idx] > best_score:
            best_score = float(scores[idx])
            best_angle = float(population[idx])

        history.append(best_score)
        diversity.append(float(np.std(population)))
        populations.append(population.copy())

    return {
        "angle": best_angle,
        "energy": best_score,
        "history": history,
        "diversity": diversity,
        "populations": populations,
        "time": time.perf_counter()-start,
        "evaluations": pop_size*(generations+1)
    }


# ---------------- RANDOM SEARCH ----------------
def run_random_search(budget,seed):

    rng = np.random.default_rng(seed)
    start = time.perf_counter()

    best = -np.inf
    best_angle = 0
    history = []

    for _ in range(budget):

        angle = rng.uniform(0,90)
        score = fitness(angle)

        if score>best:
            best = score
            best_angle = angle

        history.append(best)

    return {
        "angle":best_angle,
        "energy":best,
        "history":history,
        "time":time.perf_counter()-start,
        "evaluations":budget
    }

# ---------------- GRID SEARCH ----------------
def run_grid_search():

    start = time.perf_counter()

    candidates = np.arange(0,91,1)
    scores = np.array([fitness(a) for a in candidates])

    idx = np.argmax(scores)

    return {
        "angle":float(candidates[idx]),
        "energy":float(scores[idx]),
        "history":np.maximum.accumulate(scores).tolist(),
        "time":time.perf_counter()-start,
        "evaluations":len(candidates)
    }

# ---------------- SIDEBAR SETTINGS ----------------
DEFAULTS = {
    "population_size":20,
    "generations":50,
    "mutation_rate":0.2,
    "crossover_rate":0.8,
    "mutation_sigma":5.0,
    "number_of_runs":30,
    "random_seed":42
}

st.sidebar.header("⚙️ GA Settings")

if st.sidebar.button("🔄 Reset Settings"):
    for key,value in DEFAULTS.items():
        st.session_state[key]=value
    st.rerun()

population_size = st.sidebar.slider(
    "Population Size",10,100,20,step=10,
    key="population_size"
)

generations = st.sidebar.slider(
    "Generations",10,200,50,step=10,
    key="generations"
)

mutation_rate = st.sidebar.slider(
    "Mutation Rate",0.0,1.0,0.2,step=0.1,
    key="mutation_rate"
)

crossover_rate = st.sidebar.slider(
    "Crossover Rate",0.0,1.0,0.8,step=0.1,
    key="crossover_rate"
)

mutation_sigma = st.sidebar.slider(
    "Mutation Strength (σ)",0.1,20.0,5.0,step=0.5,
    key="mutation_sigma"
)

number_of_runs = st.sidebar.selectbox(
    "Statistical Runs",[10,20,30],index=2,
    key="number_of_runs"
)

random_seed = st.sidebar.number_input(
    "Base Random Seed",0,999999,42,
    key="random_seed"
)

# Parameter summary
st.sidebar.divider()
st.sidebar.subheader("📋 Parameters Summary")
st.sidebar.write(f"Population: **{population_size}**")
st.sidebar.write(f"Generations: **{generations}**")
st.sidebar.write(f"Mutation Rate: **{mutation_rate:.2f}**")
st.sidebar.write(f"Crossover Rate: **{crossover_rate:.2f}**")
st.sidebar.write(f"Mutation σ: **{mutation_sigma:.1f}°**")
st.sidebar.write(f"Runs: **{number_of_runs}**")
st.sidebar.write(f"Seed: **{random_seed}**")

# ---------------- ENERGY GRAPH ----------------
st.subheader("☀️ PVGIS Energy Landscape")

x = np.linspace(0,90,181)
y = [fitness(a) for a in x]

fig,ax = plt.subplots(figsize=(5,2.5))
ax.plot(x,y,color="#15803D",linewidth=1.5)
ax.scatter([x[np.argmax(y)]],[max(y)],color="#F59E0B")
ax.set_xlabel("Tilt Angle (degrees)")
ax.set_ylabel("Energy (kWh/day)")
ax.set_title("Solar Energy vs Panel Tilt")
ax.grid(alpha=0.2)
fig.tight_layout()
st.pyplot(fig,use_container_width=False)
plt.close(fig)

# ---------------- RUN OPTIMIZATION ----------------
if st.button("🚀 Run Optimization & Statistical Comparison"):

    ga_runs=[]
    random_runs=[]
    grid_runs=[]

    progress=st.progress(0)

    for run in range(number_of_runs):

        seed=int(random_seed)+run

        ga=run_ga(
            population_size,generations,
            mutation_rate,crossover_rate,
            mutation_sigma,seed
        )

        ga_runs.append(ga)

        random_runs.append(
            run_random_search(ga["evaluations"],seed+100000)
        )

        grid_runs.append(run_grid_search())

        progress.progress((run+1)/number_of_runs)

    progress.empty()

    st.session_state["experiment"]={
        "ga":ga_runs,
        "random":random_runs,
        "grid":grid_runs,
        "pop_size":population_size,
        "generations":generations,
        "mutation":mutation_rate,
        "crossover":crossover_rate,
        "sigma":mutation_sigma,
        "runs":number_of_runs,
        "seed":int(random_seed)
    }

    st.session_state["result"]=run_ga(
        population_size,generations,
        mutation_rate,crossover_rate,
        mutation_sigma,int(random_seed)
    )

# ---------------- GA RESULTS ----------------
if "result" in st.session_state:

    result=st.session_state["result"]

    st.divider()
    st.header("🧬 Genetic Algorithm Results")

    c1,c2,c3=st.columns(3)
    c1.metric("Best Tilt",f"{result['angle']:.2f}°")
    c2.metric("Energy",f"{result['energy']:.4f} kWh/day")
    c3.metric("Evaluations",f"{result['evaluations']:,}")

    if "experiment" in st.session_state:
        e=st.session_state["experiment"]
        st.caption(
            f"Population {e['pop_size']} | "
            f"Generations {e['generations']} | "
            f"Mutation {e['mutation']} | "
            f"Crossover {e['crossover']} | "
            f"Sigma {e['sigma']}°"
        )

    tab1,tab2=st.tabs(["🌱 Fitness Landscape","📈 GA Convergence"])

    with tab1:

        gen=st.slider(
            "Explore Generation",
            1,len(result["populations"]),
            len(result["populations"])
        )

        pop=result["populations"][gen-1]

        fig,ax=plt.subplots(figsize=(5,2.5))
        ax.plot(x,y,color="#15803D")
        ax.scatter(pop,[fitness(a) for a in pop],
                   color="#F59E0B",s=25)
        ax.set_xlabel("Tilt Angle")
        ax.set_ylabel("Energy")
        ax.set_title(f"Generation {gen-1}")
        ax.grid(alpha=0.2)
        fig.tight_layout()
        st.pyplot(fig,use_container_width=False)
        plt.close(fig)

    with tab2:

        fig,ax=plt.subplots(figsize=(5,2.5))
        ax.plot(result["history"],color="#15803D")
        ax.set_xlabel("Fitness Evaluation")
        
        ax.set_ylabel("Best Energy")
        ax.set_title("GA Convergence")
        ax.grid(alpha=0.2)
        fig.tight_layout()
        st.pyplot(fig,use_container_width=False)
        plt.close(fig)



    st.subheader("🧬 Population Diversity")

    fig, ax = plt.subplots(figsize=(5, 2.5))
    ax.plot(
        range(1, len(result["diversity"])+1),
        result["diversity"],
        color="#F59E0B"
    )
    ax.set_xlabel("Generation")
    ax.set_ylabel("Population Std. Deviation")
    ax.set_title("GA Population Diversity")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    st.pyplot(fig, use_container_width=False)
    plt.close(fig)

# ---------------- COMPARISON ----------------
if "experiment" in st.session_state:

    exp=st.session_state["experiment"]
    ga_runs=exp["ga"]
    random_runs=exp["random"]
    grid_runs=exp["grid"]

    st.divider()
    st.header("📊 Statistical Comparison")

    methods={
        "Genetic Algorithm":ga_runs,
        "Random Search":random_runs,
        "Grid Search":grid_runs
    }

    rows=[]

    for name,runs in methods.items():

        scores=np.array([r["energy"] for r in runs])
        times=np.array([r["time"] for r in runs])
        successes=np.sum(scores>=TARGET-TOLERANCE)

        rows.append({
            "Method":name,
            "Mean Energy":scores.mean(),
            "Std Dev":scores.std(ddof=1),
            "Mean Time":times.mean(),
            "Success Rate (%)":100*successes/len(scores),
            "Evaluations":int(np.mean([r["evaluations"] for r in runs]))
        })

    comparison=pd.DataFrame(rows)

    st.dataframe(
        comparison.style.format({
            "Mean Energy":"{:.4f}",
            "Std Dev":"{:.5f}",
            "Mean Time":"{:.6f}",
            "Success Rate (%)":"{:.1f}%",
            "Evaluations":"{:.0f}"
        }),
        use_container_width=True,
        hide_index=True
    )

    st.subheader("Average Convergence ± 1 SD")

    gh=np.array([r["history"] for r in ga_runs])
    rh=np.array([r["history"] for r in random_runs])

    gm=gh.mean(axis=0)
    gs=gh.std(axis=0,ddof=1)

    rm=rh.mean(axis=0)
    rs=rh.std(axis=0,ddof=1)

    fig,ax=plt.subplots(figsize=(5,2.5))

    gx=np.arange(1,len(gm)+1)
    rx=np.arange(1,len(rm)+1)

    ax.plot(gx,gm,color="#15803D",label="GA mean")
    ax.fill_between(gx,gm-gs,gm+gs,color="#15803D",alpha=0.18)

    ax.plot(rx,rm,color="#F59E0B",label="Random mean")
    ax.fill_between(rx,rm-rs,rm+rs,color="#F59E0B",alpha=0.18)

    ghistory=grid_runs[0]["history"]
    ax.step(np.arange(1,len(ghistory)+1),ghistory,
            where="post",color="#64748B",label="Grid")

    ax.set_xlabel("Evaluation")
    ax.set_ylabel("Best Energy")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    st.pyplot(fig,use_container_width=False)
    plt.close(fig)

    st.subheader("Execution Time Comparison")

    fig,ax=plt.subplots(figsize=(5,2.5))
    ax.bar(comparison["Method"],comparison["Mean Time"],
           color=["#15803D","#F59E0B","#64748B"])
    ax.tick_params(axis="x",rotation=10,labelsize=8)
    ax.set_ylabel("Seconds")
    fig.tight_layout()
    st.pyplot(fig,use_container_width=False)
    plt.close(fig)

    st.subheader("Success Rate Comparison")

    fig,ax=plt.subplots(figsize=(5,2.5))
    ax.bar(comparison["Method"],comparison["Success Rate (%)"],
           color=["#15803D","#F59E0B","#64748B"])
    ax.set_ylim(0,105)
    ax.tick_params(axis="x",rotation=10,labelsize=8)
    ax.set_ylabel("Success (%)")
    fig.tight_layout()
    st.pyplot(fig,use_container_width=False)
    plt.close(fig)

    st.download_button(
        "⬇️ Download Comparison Results",
        comparison.to_csv(index=False),
        "solarsynapse_comparison.csv",
        "text/csv"
    )

st.info(
    "Research note: optimization uses the current PVGIS-derived "
    "energy curve with linear interpolation. This is not yet a "
    "full pvlib electrical simulation."
)
