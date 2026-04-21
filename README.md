# Schelling Segregation Model

University of Waterloo — Psychology 420: Introduction to Computational Neuroscience Methods

A computational demonstration using NumPy, SciPy, and Matplotlib.

---

## About

We are a group of students from the University of Waterloo currently enrolled in Psychology 420: An Introduction to Computational Neuroscience Methods. As part of our coursework focusing on Agent-Based Modeling, we have completed our final project. Please find the source code in this repository, with the presentation slides and final research paper attached separately.

---

## Overview

This project implements Thomas Schelling's (1971) residential segregation model as an agent-based simulation. The core idea is simple: even mild individual preferences for living near similar neighbours can produce strong city-wide segregation — an emergent outcome nobody explicitly chose.

In this model, agents live in a 2D grid and decide whether to move or stay based on the similarity of their immediate neighbours. Agents are happier when surrounded by others who share their characteristics, and less happy when surrounded by those who differ. Once an agent becomes unhappy, it moves to a new location chosen at random from all vacant spots.

At the end of each time period, we measure the number of agents who moved (the unhappy count) and the overall satisfaction level across the whole system. The simulation runs until no agent wishes to move, meaning the system has reached equilibrium.

This dynamic allows us to observe how a simple local rule produces complex global outcomes. In particular, small differences in preferences lead to significant variations in how agents distribute themselves across the grid.

---

## What is an Agent-Based Model?

An agent-based model is a computational approach where individual agents each follow simple local rules. There is no central authority directing the outcome — segregation emerges from thousands of individual decisions. This makes ABMs well-suited to social phenomena where macro patterns arise from micro behaviour.

Schelling's model is one of the most famous ABMs in social science because it shows that global patterns like segregated neighbourhoods can emerge even when no individual agent wants segregation — they only want a modest fraction of similar neighbours nearby.

---

## The Three-Part Visualisation

The display is split into three panels that work together.

1. Grid (left) — A live view of the city. Blue and red squares are residents of two groups; grey squares are vacant houses. Watch how random mixing gradually clusters into segregated patches as generations pass.

2. Average Happiness (top right) — Happiness for each resident is the fraction of their neighbours who belong to the same group. This graph shows the city-wide average over time. It almost always rises quickly because unhappy agents keep moving until they find somewhere comfortable. A flat line means the city has reached equilibrium.

3. Unhappy People (bottom right) — The count of residents who moved in each generation. High early on, falling toward zero as the model settles. When this line hits zero, the simulation stops automatically: everyone is satisfied, and no further movement is possible.

Together, the three panels tell a story: disorder to movement to clustering to stability. You can see the grid become patchier at the same time as the unhappy count falls and happiness rises.

---

## User Interface

The following controls are available in the interactive window.

Init — Resets the grid using the current slider values. The initial conditions for both the grid configuration and parameter values are recreated each time Init is pressed. Use this whenever you want to test a new combination of parameters from scratch.

Step — Advances one iteration of the simulation. Each click moves any unhappy agents to vacant homes and updates both charts. Useful for debugging or exploring intermediate states one generation at a time.

Run — Starts the automated simulation. The display updates continuously at regular intervals until equilibrium is reached or Stop is pressed.

Stop — Halts automatic updating. Press Run again to resume from where the simulation left off.

Sliders — Three adjustable sliders let you define the starting conditions before pressing Init or Run.

- Grid size: width and height of the square grid in cells (20 to 100)
- Threshold: the minimum fraction of same-group neighbours an agent needs to be happy (0.05 to 0.95)
- Vacant %: the proportion of houses left empty at the start (5% to 40%)

---

## Demo Walkthrough

Step 1 — Default run

Leave all sliders at their starting values (Threshold = 0.40, Grid = 60, Vacant = 10%) and press Run. Within roughly 20-30 generations the unhappy count will drop to zero and the grid will show clear coloured patches. The happiness graph will plateau near 0.80-0.90 even though the threshold was only 0.40 — this overshoot is a key Schelling insight.

Step 2 — Raise the threshold

Press Init to reset, then drag the Threshold slider to 0.70. Press Run. Agents are now harder to satisfy, so movement continues for many more generations. The final grid shows much larger, more solid clusters. Happiness still rises but it takes longer, and the patches are coarser.

Step 3 — Lower the vacancy rate

Reset, set Vacant % to 0.05 (5%). With fewer empty houses there are fewer places to move, so the model takes longer to reach equilibrium — or may oscillate without fully settling. Watch the unhappy count bounce.

Step 4 — Use Step mode

Press Init, then press Step repeatedly. Each click advances exactly one generation so you can observe exactly which agents moved and how the cluster boundaries shift generation by generation.

Key result to note: even with a threshold of 0.30 (agents are happy if only 30% of neighbours are like them — a fairly tolerant preference) the final grid is clearly segregated. This is Schelling's central finding.

---

## Package Comparison

Three packages do all the heavy lifting. Below is a brief comparison of each with the alternatives considered.

### NumPy

Used for the grid (a 2-D integer array), random shuffling, vectorised comparisons, and building the list of new residents to scatter into vacant houses.

A 60x60 grid is 3,600 cells. Checking each cell's neighbours with a Python for-loop requires roughly 29,000 comparisons per step. NumPy performs the same work as a single C-level array operation, making it 10-100x faster in practice. For larger grids (100x100) this difference becomes essential for real-time animation.

Alternative considered: pandas DataFrames can hold 2-D data but carry overhead designed for labelled, heterogeneous data. For a uniform grid of integers, a NumPy array is simpler, faster, and uses less memory.

### SciPy — convolve2d

Used for counting each cell's neighbours in a single call. A 3x3 kernel of ones (with the centre set to zero) is convolved with the grid to produce a new array where each cell contains the count of same-group neighbours — without a single explicit loop.

The manual approach would nest four Python for-loops (rows x cols x kernel rows x kernel cols). convolve2d replaces this with an optimised C/Fortran routine. The boundary='wrap' argument handles edge cells automatically by treating the grid as a torus, removing edge-case logic entirely.

Alternative considered: scipy.ndimage.generic_filter can apply arbitrary functions per cell but is slower for fixed arithmetic kernels. A pure NumPy rolling-sum with np.roll is possible but requires eight separate shift operations and additional bookkeeping. convolve2d is both cleaner to read and faster to execute for this use case.

### Matplotlib

Used for the interactive figure — imshow for the grid, FuncAnimation for the real-time loop, Slider and Button widgets for controls, and line plots for the two time-series graphs.

Matplotlib ships with NumPy and SciPy as part of the standard scientific Python stack, so no extra dependencies are needed. Its FuncAnimation class integrates directly with the same figure that holds the imshow grid, meaning all three panels update in one coordinated loop.
,
Alternatives considered:

- Pygame: excellent for custom game-style rendering but requires manual drawing logic for axes, labels, line plots, and widgets — far more code for no gain here.
- Plotly/Dash: produces beautiful browser-based output but adds a web server layer or requires exporting frames, neither of which suits a self-contained .py simulation.
- Mesa: a dedicated Python ABM framework with a built-in browser visualisation server. Mesa is a good choice when you need agent heterogeneity, complex scheduling, or large multi-run experiments. For this model, where agents are identical except for colour and the update rule is a single vectorised convolution, Mesa's overhead would add complexity without benefit.

---

## Requirements

Python 3.8+

```
pip install numpy scipy matplotlib
```

## Running

```
python schelling_segregation.py
```
