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
#
# =============================================================================
#  SETTINGS — change these to experiment before running
# =============================================================================

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.animation as animation
from matplotlib.widgets import Slider, Button
from matplotlib.patches import Patch
from scipy.signal import convolve2d

GRID_SIZE  = 60    # Grid width and height in cells (60 → 60×60 = 3 600 agents)
THRESHOLD  = 0.4   # Minimum fraction of same-group neighbours to be happy (0–1)
VACANT     = 0.1   # Fraction of houses left empty at initialisation (0–1)

# =============================================================================
#  COLOUR MAP
#  -1 = vacant (grey)  |  0 = blue resident  |  1 = red resident
# =============================================================================

COLOURS      = mcolors.ListedColormap(['#cccccc', '#4a90d9', '#e05555'])
COLOUR_RANGE = mcolors.BoundaryNorm([-1.5, -0.5, 0.5, 1.5], COLOURS.N)

# The neighbour kernel: a 3×3 grid of 1s with 0 in the centre.
# When convolved with the grid it counts the 8 surrounding occupied cells.
NEIGHBOUR_KERNEL = np.array([[1, 1, 1],
                              [1, 0, 1],
                              [1, 1, 1]], dtype=np.int8)


# =============================================================================
#  GRID INITIALISATION
# =============================================================================

def make_random_grid(size, vacant_fraction):
    """
    Create a randomly shuffled grid of residents and vacant houses.

    Parameters
    ----------
    size : int
        Width (and height) of the square grid.
    vacant_fraction : float
        Proportion of cells to leave empty (value between 0 and 1).

    Returns
    -------
    numpy.ndarray of shape (size, size) with dtype int8.
        Values: -1 = vacant, 0 = blue, 1 = red.

    Notes
    -----
    Blue and red residents are split as evenly as possible. Any odd resident
    is assigned to red. The array is shuffled before reshaping so the initial
    placement is uniformly random.
    """
    total_squares = size * size
    num_empty     = int(total_squares * vacant_fraction)
    num_people    = total_squares - num_empty
    num_blue      = num_people // 2
    num_red       = num_people - num_blue

    grid = np.zeros(total_squares, dtype=np.int8)
    grid[:num_red]    = 1     # first num_red positions = red
    grid[-num_empty:] = -1    # last num_empty positions = vacant
    # zeros in the middle = blue; shuffle randomises placement
    np.random.shuffle(grid)
    return grid.reshape(size, size)


# =============================================================================
#  SIMULATION STEP
# =============================================================================

def run_one_step(grid, threshold):
    """
    Advance the simulation by one generation.

    Mechanism
    ---------
    1. For each cell, count same-group neighbours and total neighbours using
       convolution (see package notes above for why convolve2d is used).
    2. A resident is *unhappy* if their same-group fraction < threshold.
    3. All unhappy residents vacate simultaneously (their cells become -1).
    4. The vacated cells are filled by shuffling the displaced residents back
       into the empty pool. This preserves total population exactly.

    Parameters
    ----------
    grid : numpy.ndarray
        The current grid state (modified in place and returned).
    threshold : float
        The happiness threshold from the slider.

    Returns
    -------
    grid : numpy.ndarray
        Updated grid after one step.
    num_unhappy : int
        Number of residents who moved this generation.
    """
    options = dict(mode='same', boundary='wrap')  # wrap = toroidal boundary

    blue_neighbours  = convolve2d(grid == 0, NEIGHBOUR_KERNEL, **options)
    red_neighbours   = convolve2d(grid == 1, NEIGHBOUR_KERNEL, **options)
    total_neighbours = convolve2d(grid != -1, NEIGHBOUR_KERNEL, **options)

    # Avoid division by zero for isolated cells with no neighbours at all
    safe_total = np.where(total_neighbours == 0, 1, total_neighbours)

    blue_unhappy = (blue_neighbours / safe_total < threshold) & (grid == 0)
    red_unhappy  = (red_neighbours  / safe_total < threshold) & (grid == 1)

    everyone_unhappy = blue_unhappy | red_unhappy
    grid[everyone_unhappy] = -1   # vacate unhappy residents

    num_empty       = (grid == -1).sum()
    num_blue_moving = blue_unhappy.sum()
    num_red_moving  = red_unhappy.sum()

    # Build a shuffled list of new residents to scatter into vacancies
    new_residents = -np.ones(num_empty, dtype=np.int8)
    new_residents[:num_blue_moving] = 0
    new_residents[num_blue_moving:num_blue_moving + num_red_moving] = 1
    np.random.shuffle(new_residents)

    grid[grid == -1] = new_residents

    return grid, everyone_unhappy.sum()


# =============================================================================
#  HAPPINESS METRIC
# =============================================================================

def average_happiness(grid):
    """
    Compute the city-wide average happiness score.

    For each occupied cell, happiness = (same-group neighbours) /
    (total neighbours). The mean is taken over all occupied cells.
    Vacant cells (-1) are excluded via np.nanmean.

    This is plotted in the top-right graph. Its steady rise toward ~0.8–0.9
    even at a threshold of 0.4 is the hallmark Schelling overshoot: agents
    end up *more* segregated than they strictly required.
    """
    options = dict(mode='same', boundary='wrap')
    blue_neighbours  = convolve2d(grid == 0, NEIGHBOUR_KERNEL, **options)
    red_neighbours   = convolve2d(grid == 1, NEIGHBOUR_KERNEL, **options)
    total_neighbours = convolve2d(grid != -1, NEIGHBOUR_KERNEL, **options)
    safe_total = np.where(total_neighbours == 0, 1, total_neighbours)

    happiness = np.where(grid == 0, blue_neighbours / safe_total,
                np.where(grid == 1, red_neighbours  / safe_total, np.nan))
    return np.nanmean(happiness)


# =============================================================================
#  FIGURE LAYOUT
#  Three panels: grid (left) | happiness line (top right) | unhappy line (bottom right)
# =============================================================================

fig = plt.figure(figsize=(14, 8), facecolor='#1a1a2e')
fig.suptitle("Schelling Segregation Model", color='white', fontsize=14,
             fontweight='bold', y=0.98)

# Left panel — the city grid
ax_grid = fig.add_axes([0.03, 0.15, 0.46, 0.78])
ax_grid.set_facecolor('#1a1a2e')
ax_grid.axis('off')

# Top-right panel — average happiness over time
ax_happy = fig.add_axes([0.57, 0.55, 0.40, 0.36])
ax_happy.set_facecolor('#0d0d1a')
ax_happy.set_xlabel('Generation', color='#aaa', fontsize=9)
ax_happy.set_ylabel('Avg Happiness', color='#aaa', fontsize=9)
ax_happy.tick_params(colors='#aaa', labelsize=8)
ax_happy.set_ylim(0.4, 1.05)
ax_happy.set_xlim(0, 10)
ax_happy.grid(True, color='#2a2a4a', lw=0.5)
for side in ax_happy.spines.values():
    side.set_color('#2a2a4a')
ax_happy.text(0.5, 1.06, 'Average happiness over time',
              transform=ax_happy.transAxes,
              color='white', fontsize=10, ha='center', va='bottom')
happy_line, = ax_happy.plot([], [], color='#f0c040', lw=2)

# Bottom-right panel — number of unhappy movers per generation
ax_unhappy = fig.add_axes([0.57, 0.15, 0.40, 0.30])
ax_unhappy.set_facecolor('#0d0d1a')
ax_unhappy.set_xlabel('Generation', color='#aaa', fontsize=9)
ax_unhappy.set_ylabel('No. of people', color='#aaa', fontsize=9)
ax_unhappy.tick_params(colors='#aaa', labelsize=8)
ax_unhappy.set_ylim(0, 1)
ax_unhappy.set_xlim(0, 10)
ax_unhappy.grid(True, color='#2a2a4a', lw=0.5)
for side in ax_unhappy.spines.values():
    side.set_color('#2a2a4a')
ax_unhappy.text(0.5, 1.06, 'Unhappy people over time',
                transform=ax_unhappy.transAxes,
                color='white', fontsize=10, ha='center', va='bottom')
unhappy_line, = ax_unhappy.plot([], [], color='#e05555', lw=2)

# Legend
legend_items = [Patch(facecolor='#4a90d9', label='Blue person'),
                Patch(facecolor='#e05555', label='Red person'),
                Patch(facecolor='#cccccc', label='Empty house')]
fig.legend(handles=legend_items, loc='lower right',
           bbox_to_anchor=(0.97, 0.01), ncol=3,
           facecolor='#12122a', labelcolor='white',
           fontsize=9, framealpha=0.9, edgecolor='#333355')


# =============================================================================
#  SIMULATION STATE
# =============================================================================
#  A single dictionary holds all mutable state so functions do not rely on
#  hidden global side-effects. This makes the control flow easier to follow.

state = {
    'grid':            make_random_grid(GRID_SIZE, VACANT),
    'generation':      0,
    'running':         False,
    'happy_history':   [],   # one float per generation (average happiness)
    'unhappy_history': [],   # one int   per generation (number of movers)
}

grid_image  = ax_grid.imshow(state['grid'], cmap=COLOURS, norm=COLOUR_RANGE,
                              interpolation='nearest')
status_text = ax_grid.set_title('Generation: 0  |  Press Run to start',
                                 color='white', fontsize=10, pad=8)


# =============================================================================
#  GRAPH UPDATE HELPER
# =============================================================================

def update_graphs():
    """Redraw both time-series lines using the current history lists."""
    n = len(state['happy_history'])
    if n > 0:
        x = list(range(n))
        happy_line.set_data(x, state['happy_history'])
        unhappy_line.set_data(x, state['unhappy_history'])
        ax_happy.set_xlim(0, max(10, n))
        ax_unhappy.set_xlim(0, max(10, n))
        ax_unhappy.set_ylim(0, max(1, max(state['unhappy_history']) * 1.15))


# =============================================================================
#  SINGLE GENERATION ADVANCE
# =============================================================================

def do_one_generation():
    """
    Run one simulation step and update all three panels.

    Reads the current threshold from the slider so changes take effect
    immediately even during an ongoing [Run] sequence. Stops the animation
    automatically when no agent moves (equilibrium reached).
    """
    state['grid'], num_unhappy = run_one_step(state['grid'], sl_threshold.val)
    state['generation'] += 1
    happiness = average_happiness(state['grid'])

    state['happy_history'].append(happiness)
    state['unhappy_history'].append(num_unhappy)

    grid_image.set_data(state['grid'])
    status_text.set_text(
        f"Generation: {state['generation']}  |  "
        f"Unhappy: {num_unhappy}  |  "
        f"Happiness: {happiness:.3f}")

    update_graphs()

    # Equilibrium: nobody moved — simulation is complete
    if num_unhappy == 0:
        state['running'] = False
        status_text.set_text(
            f"Done! Equilibrium at generation {state['generation']}  |  "
            f"Happiness: {happiness:.3f}")

    return num_unhappy


# =============================================================================
#  ANIMATION LOOP
#  Called every 80 ms by FuncAnimation. Only advances the model when the
#  'running' flag is True; otherwise returns the unchanged artists so
#  Matplotlib does not need to redraw anything.
# =============================================================================

def animation_loop(frame):
    if state['running']:
        do_one_generation()
    return [grid_image, status_text, happy_line, unhappy_line]


# =============================================================================
#  SLIDERS
# =============================================================================

ax_sl_size      = fig.add_axes([0.12, 0.10,  0.18, 0.025])
ax_sl_threshold = fig.add_axes([0.12, 0.065, 0.18, 0.025])
ax_sl_vacant    = fig.add_axes([0.12, 0.03,  0.18, 0.025])

for ax in [ax_sl_size, ax_sl_threshold, ax_sl_vacant]:
    ax.set_facecolor('#1a1a2e')

sl_size      = Slider(ax_sl_size,      'Grid size',  20, 100, valinit=GRID_SIZE,  valstep=5,    color='#4a90d9')
sl_threshold = Slider(ax_sl_threshold, 'Threshold',  0.05, 0.95, valinit=THRESHOLD, valstep=0.05, color='#f0c040')
sl_vacant    = Slider(ax_sl_vacant,    'Vacant %',   0.05, 0.40, valinit=VACANT,    valstep=0.05, color='#aaaaaa')

for slider in [sl_size, sl_threshold, sl_vacant]:
    slider.label.set_color('white')
    slider.valtext.set_color('white')


# =============================================================================
#  BUTTONS
# =============================================================================

ax_btn_init = fig.add_axes([0.33, 0.06, 0.07, 0.045])
ax_btn_step = fig.add_axes([0.41, 0.06, 0.07, 0.045])
ax_btn_run  = fig.add_axes([0.33, 0.01, 0.07, 0.045])
ax_btn_stop = fig.add_axes([0.41, 0.01, 0.07, 0.045])

btn_init = Button(ax_btn_init, 'Init',  color='#223344', hovercolor='#334455')
btn_step = Button(ax_btn_step, 'Step',  color='#223344', hovercolor='#334455')
btn_run  = Button(ax_btn_run,  'Run',   color='#224422', hovercolor='#335533')
btn_stop = Button(ax_btn_stop, 'Stop',  color='#442222', hovercolor='#553333')

for btn in [btn_init, btn_step, btn_run, btn_stop]:
    btn.label.set_color('white')
    btn.label.set_fontsize(10)


# =============================================================================
#  BUTTON CALLBACKS
# =============================================================================

def click_init(event):
    """
    Reset everything: generate a new random grid using the current slider
    values, clear all history, and reset the graphs.
    Call this whenever you change Grid size or Vacant % before pressing Run.
    """
    state['running'] = False
    new_size = int(sl_size.val)
    state['grid'] = make_random_grid(new_size, sl_vacant.val)
    state['generation'] = 0
    state['happy_history'].clear()
    state['unhappy_history'].clear()
    grid_image.set_data(state['grid'])
    grid_image.set_extent([-0.5, new_size - 0.5, new_size - 0.5, -0.5])
    status_text.set_text('Generation: 0  |  Press Run to start')
    happy_line.set_data([], [])
    unhappy_line.set_data([], [])
    ax_happy.set_xlim(0, 10)
    ax_unhappy.set_xlim(0, 10)
    fig.canvas.draw_idle()

def click_step(event):
    """Advance exactly one generation (useful for studying individual steps)."""
    state['running'] = False
    do_one_generation()
    fig.canvas.draw_idle()

def click_run(event):
    """Start continuous animation at ~12 generations per second (80 ms/frame)."""
    state['running'] = True

def click_stop(event):
    """Pause the animation without resetting state. Press Run to resume."""
    state['running'] = False

btn_init.on_clicked(click_init)
btn_step.on_clicked(click_step)
btn_run.on_clicked(click_run)
btn_stop.on_clicked(click_stop)


# =============================================================================
#  START THE INTERACTIVE WINDOW
# =============================================================================

anim = animation.FuncAnimation(fig, animation_loop, interval=80,
                                blit=False, cache_frame_data=False)
plt.show()
