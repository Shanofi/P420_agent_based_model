import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.animation as animation
from matplotlib.widgets import Slider, Button
from matplotlib.patches import Patch
from scipy.signal import convolve2d

# ==============================================================================
# SETTINGS - change these to experiment!
# ==============================================================================
GRID_SIZE  = 60    # how big the grid is (60 means 60x60 squares)
THRESHOLD  = 0.4   # how similar your neighbours need to be (0 to 1)
VACANT     = 0.1   # fraction of empty houses (0.1 = 10% empty)

# ==============================================================================
# COLOURS FOR THE GRID
# -1 = empty (grey), 0 = blue person, 1 = red person
# ==============================================================================
COLOURS = mcolors.ListedColormap(['#cccccc', '#4a90d9', '#e05555'])
COLOUR_RANGE = mcolors.BoundaryNorm([-1.5, -0.5, 0.5, 1.5], COLOURS.N)

# This kernel is used to count neighbours (the 8 squares around each person)
NEIGHBOUR_KERNEL = np.array([[1, 1, 1],
                              [1, 0, 1],
                              [1, 1, 1]], dtype=np.int8)

# ==============================================================================
# FUNCTION: create a random starting grid
# ==============================================================================
def make_random_grid(size, vacant_fraction):
    total_squares = size * size
    num_empty     = int(total_squares * vacant_fraction)
    num_people    = total_squares - num_empty
    num_blue      = num_people // 2
    num_red       = num_people - num_blue

    grid = np.zeros(total_squares, dtype=np.int8)
    grid[:num_red]    = 1
    grid[-num_empty:] = -1
    np.random.shuffle(grid)
    return grid.reshape(size, size)

# ==============================================================================
# FUNCTION: run one step of the simulation
# ==============================================================================
def run_one_step(grid, threshold):
    options = dict(mode='same', boundary='wrap')

    blue_neighbours  = convolve2d(grid == 0, NEIGHBOUR_KERNEL, **options)
    red_neighbours   = convolve2d(grid == 1, NEIGHBOUR_KERNEL, **options)
    total_neighbours = convolve2d(grid != -1, NEIGHBOUR_KERNEL, **options)

    safe_total = np.where(total_neighbours == 0, 1, total_neighbours)

    blue_unhappy = (blue_neighbours / safe_total < threshold) & (grid == 0)
    red_unhappy  = (red_neighbours  / safe_total < threshold) & (grid == 1)

    everyone_unhappy = blue_unhappy | red_unhappy
    grid[everyone_unhappy] = -1

    num_empty          = (grid == -1).sum()
    num_blue_moving    = blue_unhappy.sum()
    num_red_moving     = red_unhappy.sum()

    new_residents = -np.ones(num_empty, dtype=np.int8)
    new_residents[:num_blue_moving] = 0
    new_residents[num_blue_moving:num_blue_moving + num_red_moving] = 1
    np.random.shuffle(new_residents)

    grid[grid == -1] = new_residents

    return grid, everyone_unhappy.sum()

# ==============================================================================
# FUNCTION: calculate average happiness across all people
# ==============================================================================
def average_happiness(grid):
    options = dict(mode='same', boundary='wrap')
    blue_neighbours  = convolve2d(grid == 0, NEIGHBOUR_KERNEL, **options)
    red_neighbours   = convolve2d(grid == 1, NEIGHBOUR_KERNEL, **options)
    total_neighbours = convolve2d(grid != -1, NEIGHBOUR_KERNEL, **options)
    safe_total = np.where(total_neighbours == 0, 1, total_neighbours)

    happiness = np.where(grid == 0, blue_neighbours / safe_total,
                np.where(grid == 1, red_neighbours  / safe_total, np.nan))
    return np.nanmean(happiness)

# ==============================================================================
# BUILD THE WINDOW
# ==============================================================================
fig = plt.figure(figsize=(14, 8), facecolor='#1a1a2e')
fig.suptitle("Schelling Segregation Model", color='white', fontsize=14, fontweight='bold', y=0.98)

# --- left side: the grid (takes up left half) ---
ax_grid = fig.add_axes([0.03, 0.15, 0.46, 0.78])
ax_grid.set_facecolor('#1a1a2e')
ax_grid.axis('off')

# --- top right: happiness over time ---
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
# title sits above the axes as plain text so it never overlaps
ax_happy.text(0.5, 1.06, 'Average happiness over time',
              transform=ax_happy.transAxes,
              color='white', fontsize=10, ha='center', va='bottom')
happy_line, = ax_happy.plot([], [], color='#f0c040', lw=2)

# --- bottom right: unhappy people over time ---
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

# --- colour legend sits bottom right, below the unhappy graph ---
legend_items = [Patch(facecolor='#4a90d9', label='Blue person'),
                Patch(facecolor='#e05555', label='Red person'),
                Patch(facecolor='#cccccc', label='Empty house')]
fig.legend(handles=legend_items, loc='lower right',
           bbox_to_anchor=(0.97, 0.01),
           ncol=3,
           facecolor='#12122a', labelcolor='white',
           fontsize=9, framealpha=0.9,
           edgecolor='#333355')

# ==============================================================================
# SIMULATION STATE
# ==============================================================================
state = {
    'grid':            make_random_grid(GRID_SIZE, VACANT),
    'generation':      0,
    'running':         False,
    'happy_history':   [],
    'unhappy_history': [],
}

grid_image  = ax_grid.imshow(state['grid'], cmap=COLOURS, norm=COLOUR_RANGE,
                              interpolation='nearest')
status_text = ax_grid.set_title('Generation: 0  |  Press Run to start',
                                 color='white', fontsize=10, pad=8)

# ==============================================================================
# FUNCTION: refresh both graphs
# ==============================================================================
def update_graphs():
    n = len(state['happy_history'])
    if n > 0:
        x = list(range(n))
        happy_line.set_data(x, state['happy_history'])
        unhappy_line.set_data(x, state['unhappy_history'])
        ax_happy.set_xlim(0, max(10, n))
        ax_unhappy.set_xlim(0, max(10, n))
        ax_unhappy.set_ylim(0, max(1, max(state['unhappy_history']) * 1.15))

# ==============================================================================
# FUNCTION: advance one generation
# ==============================================================================
def do_one_generation():
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

    if num_unhappy == 0:
        state['running'] = False
        status_text.set_text(
            f"Done! Equilibrium at generation {state['generation']}  |  "
            f"Happiness: {happiness:.3f}")

    return num_unhappy

# ==============================================================================
# ANIMATION LOOP - runs automatically every 80ms
# ==============================================================================
def animation_loop(frame):
    if state['running']:
        do_one_generation()
    return [grid_image, status_text, happy_line, unhappy_line]

# ==============================================================================
# SLIDERS  (sit along the bottom left)
# ==============================================================================
ax_sl_size      = fig.add_axes([0.12, 0.10, 0.18, 0.025]); ax_sl_size.set_facecolor('#1a1a2e')
ax_sl_threshold = fig.add_axes([0.12, 0.065, 0.18, 0.025]); ax_sl_threshold.set_facecolor('#1a1a2e')
ax_sl_vacant    = fig.add_axes([0.12, 0.03, 0.18, 0.025]); ax_sl_vacant.set_facecolor('#1a1a2e')

sl_size      = Slider(ax_sl_size,      'Grid size',  20, 100, valinit=GRID_SIZE,  valstep=5,    color='#4a90d9')
sl_threshold = Slider(ax_sl_threshold, 'Threshold',  0.05, 0.95, valinit=THRESHOLD, valstep=0.05, color='#f0c040')
sl_vacant    = Slider(ax_sl_vacant,    'Vacant %',   0.05, 0.40, valinit=VACANT,    valstep=0.05, color='#aaaaaa')

for slider in [sl_size, sl_threshold, sl_vacant]:
    slider.label.set_color('white')
    slider.valtext.set_color('white')

# ==============================================================================
# BUTTONS  (sit to the right of the sliders)
# ==============================================================================
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

# ==============================================================================
# BUTTON ACTIONS
# ==============================================================================
def click_init(event):
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
    state['running'] = False
    do_one_generation()
    fig.canvas.draw_idle()

def click_run(event):
    state['running'] = True

def click_stop(event):
    state['running'] = False

btn_init.on_clicked(click_init)
btn_step.on_clicked(click_step)
btn_run.on_clicked(click_run)
btn_stop.on_clicked(click_stop)

# ==============================================================================
# START
# ==============================================================================
anim = animation.FuncAnimation(fig, animation_loop, interval=80,
                                blit=False, cache_frame_data=False)
plt.show()