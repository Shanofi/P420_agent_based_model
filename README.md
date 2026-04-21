Introduction:
---------------------------------------------------------------------------------------------------------------------------------------------

We are a group of students from the University of Waterloo currently enrolled in Psychology 420: An Introduction to Computational Neuroscience Methods.

As part of our coursework focusing on Agent-Based Modeling (ABM), we have completed our final project. Please find our presentation slides, source code, and final research paper attached for your review.

In Regards to the Segregation model
---------------------------------------------------------------------------------------------------------------------------------------------

In this model, we simulate a 2D space where the agents can live in different locations. Agents can also decide whether they want to move or stay at their location based on the similarity of their neighbors. Agents are happier if they live near other agents with similar characteristics and less happy when they are surrounded by agent(s) of a different characteristic.

Once an agent becomes unhappy, the agent moves to a new location chosen at random from among all vacant locations. At the end of each time period, we measure the number of agents that would like to move (the "unhappy" count) as well as the overall level of satisfaction (or "happiness") in the entire system.

We continue to run the simulation until no more agents wish to move, indicating that the system has reached equilibrium.

This dynamic behavior allows us to see how a simple local rule used by individuals in the system leads to complex global outcomes. In particular, we find that small differences in preferences lead to significant variations in the distribution of agents across locations.

Key Features
---------------------------------------------------------------------------------------------------------------------------------------------

• Interactive 2-D Grid
• Real-Time Charts 
• User-Defined Parameters 
• Two Modes of Operation

User Interface
---------------------------------------------------------------------------------------------------------------------------------------------

Below is a list of controls available within the user interface:

• INIT Button: Resets the grid using current values for parameters. This means that the initial conditions for both the grid configuration and parameter values are recreated each time INIT is pressed. It is assumed that users intend to test various combinations of parameters.

• STEP Button: Advances one iteration of the simulation. Each time STEP is clicked, the number of unhappy agents and the average happiness score increase and/or decrease depending upon whether there are enough similar neighbors present. If there aren't sufficient numbers of similarly colored neighbors nearby then those unhappy agents are moved to vacant homes. The STEP button is useful for debugging purposes or to explore intermediate states. Users can advance as few or as many iterations as desired; however, they cannot observe changes during an ongoing simulation.

• RUN Button: Starts the automated simulation process. When RUN is selected, the simulation proceeds continuously and automatically updates the display at regular intervals. Pressing STOP will halt further progress until RUN is once again activated. If RUN is clicked while already running, the next frame will be displayed immediately after clicking STOP.

• STOP Button: Stops automatic updating of the graph when RUN is active.

• PAUSE Button: Pauses but doesn't stop simulations completely. When PAUSE is pressed, users can make additional selections or modify parameters without losing their position in the simulation.

• SLIDERS: Three adjustable sliders allow users to define three types of input variables: grid size, threshold value and percent vacant (vacancy percentage).

Requirements
---------------------------------------------------------------------------------------------------------------------------------------------

Python version 3.8+

numpy
matplotlib
scipy

# =============================================================================
#  SCHELLING SEGREGATION MODEL
#  A computational demonstration using NumPy, SciPy, and Matplotlib
# =============================================================================
#
#  OVERVIEW
#  --------
#  This file implements Thomas Schelling's (1971) residential segregation model
#  as an agent-based simulation. The core idea is simple: even mild individual
#  preferences for living near similar neighbours can produce strong city-wide
#  segregation — an emergent outcome nobody explicitly chose.
#
#  The simulation runs interactively. Use the sliders and buttons to explore how
#  changing the threshold, grid size, and vacancy rate affect the outcome.
#
# =============================================================================
#  WHAT IS AN AGENT-BASED MODEL (ABM)?
# =============================================================================
#
#  An agent-based model is a computational approach where individual "agents"
#  (here: residents) each follow simple local rules. There is no central
#  authority directing the outcome — segregation emerges from thousands of
#  individual decisions. This makes ABMs well-suited to social phenomena where
#  macro patterns arise from micro behaviour.
#
#  Schelling's model is one of the most famous ABMs in social science because
#  it shows that global patterns (segregated neighbourhoods) can emerge even
#  when no individual agent *wants* segregation — they only want a modest
#  fraction of similar neighbours.
#
# =============================================================================
#  THE THREE-PART VISUALISATION
# =============================================================================
#
#  The display is split into three panels that work together:
#
#  1. GRID (left)
#     A live view of the city. Blue and red squares are residents of two
#     groups; grey squares are vacant houses. Watch how random mixing
#     gradually clusters into segregated patches as generations pass.
#
#  2. AVERAGE HAPPINESS (top right)
#     Happiness for each resident is the fraction of their neighbours who
#     belong to the same group. This graph shows the city-wide average over
#     time. It almost always rises quickly — because unhappy agents keep
#     moving until they find somewhere comfortable. A flat line means the
#     city has reached equilibrium.
#
#  3. NUMBER OF UNHAPPY PEOPLE (bottom right)
#     The count of residents who moved in each generation. High early on,
#     falling toward zero as the model settles. When this line hits zero
#     the simulation stops automatically: everyone is satisfied, and no
#     further movement is possible.
#
#  Together the three panels tell a story: disorder → movement → clustering →
#  stability. You can see the grid become patchier at the same time as the
#  unhappy count falls and happiness rises.
#
# =============================================================================
#  HOW TO RUN A DEMO (TUTORIAL WALKTHROUGH)
# =============================================================================
#
#  Step 1 — Default run
#    Leave all sliders at their starting values (Threshold = 0.40, Grid = 60,
#    Vacant = 10 %) and press [Run]. Within roughly 20–30 generations the
#    unhappy count will drop to zero and the grid will show clear coloured
#    patches. The happiness graph will plateau near 0.80–0.90 even though the
#    threshold was only 0.40 — this overshoot is a key Schelling insight.
#
#  Step 2 — Raise the threshold
#    Press [Init] to reset, then drag the Threshold slider to 0.70. Press
#    [Run]. Agents are now harder to satisfy, so movement continues for many
#    more generations. The final grid shows much larger, more solid clusters.
#    Happiness still rises — but it takes longer, and the patches are coarser.
#
#  Step 3 — Lower the vacancy rate
#    Reset, set Vacant % to 0.05 (5 %). With fewer empty houses there are
#    fewer places to move, so the model takes longer to reach equilibrium —
#    or may oscillate without fully settling. Watch the unhappy count bounce.
#
#  Step 4 — Use Step mode
#    Press [Init], then press [Step] repeatedly. Each click advances exactly
#    one generation so you can observe exactly which agents moved and how the
#    cluster boundaries shift generation by generation.
#
#  Key result to note: even with a threshold of 0.30 (agents are happy if
#  only 30 % of neighbours are like them — a fairly tolerant preference)
#  the final grid is clearly segregated. This is Schelling's central finding.
#
# =============================================================================
#  PACKAGE COMPARISON: WHY NUMPY + SCIPY + MATPLOTLIB?
# =============================================================================
#
#  Three packages do all the heavy lifting here. Below is a brief comparison
#  of each with the alternatives considered.
#
#  ── NumPy ──────────────────────────────────────────────────────────────────
#  Used for: the grid (a 2-D integer array), random shuffling, vectorised
#  comparisons (grid == 0, grid != -1), and building the list of new
#  residents to scatter into vacant houses.
#
#  Why NumPy over a plain Python list?
#    A 60×60 grid is 3 600 cells. Checking each cell's neighbours with a
#    Python for-loop requires ~3 600 × 8 = ~29 000 comparisons per step.
#    NumPy performs the same work as a single C-level array operation,
#    making it 10–100× faster in practice. For larger grids (100×100) this
#    difference becomes essential for real-time animation.
#
#    Alternative considered: pandas DataFrames can hold 2-D data but carry
#    overhead designed for labelled, heterogeneous data. For a uniform grid
#    of integers, a NumPy array is simpler, faster, and uses less memory.
#
#  ── SciPy (convolve2d) ─────────────────────────────────────────────────────
#  Used for: counting each cell's neighbours in a single call. A 3×3
#  kernel of ones (with the centre set to zero) is convolved with the grid
#  to produce a new array where each cell contains the count of same-group
#  neighbours — without a single explicit loop.
#
#  Why SciPy's convolve2d over a manual neighbour loop?
#    The manual approach would nest four Python for-loops (rows × cols ×
#    kernel rows × kernel cols). convolve2d replaces this with an optimised
#    C/Fortran routine. The boundary='wrap' argument also handles edge cells
#    automatically by treating the grid as a torus — agents on the left edge
#    see agents on the right edge as neighbours — removing edge-case logic.
#
#    Alternative considered: scipy.ndimage.generic_filter can apply
#    arbitrary functions per cell but is slower for fixed arithmetic kernels.
#    A pure NumPy rolling-sum with np.roll is possible but requires eight
#    separate shift operations and additional bookkeeping. convolve2d is
#    both cleaner to read and faster to execute for this use case.
#
#  ── Matplotlib ─────────────────────────────────────────────────────────────
#  Used for: the interactive figure — imshow for the grid, FuncAnimation
#  for the real-time loop, Slider and Button widgets for controls, and
#  line plots for the two time-series graphs.
#
#  Why Matplotlib over alternatives?
#    Matplotlib ships with NumPy/SciPy as part of the standard scientific
#    Python stack, so no extra dependencies are needed. Its FuncAnimation
#    class integrates directly with the same figure that holds the imshow
#    grid, meaning all three panels (grid + two graphs) update in one
#    coordinated loop.
#
#    Alternatives considered:
#    • Pygame: excellent for custom game-style rendering but requires manual
#      drawing logic for axes, labels, line plots, and widgets — far more
#      code for no gain here.
#    • Plotly/Dash: produces beautiful browser-based output but adds a web
#      server layer (Dash) or requires exporting frames (Plotly), neither of
#      which suits a self-contained .py simulation.
#    • Mesa: a dedicated Python ABM framework with a built-in browser
#      visualisation server. Mesa is a good choice when you need agent
#      heterogeneity, complex scheduling, or large multi-run experiments.
#      For this model, where agents are identical except for colour and the
#      update rule is a single vectorised convolution, Mesa's overhead
#      (defining Agent and Model classes, a server module, etc.) would add
#      complexity without benefit. The NumPy + Matplotlib stack expresses
#      the logic more directly.
