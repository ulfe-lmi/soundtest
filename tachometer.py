import curses
import time
import math

# Define constants
X_SCALE_FACTOR = 1.5
RADIUS = 10  # Radius for both tachometers
TMAX = 10  # Total time duration in seconds

def draw_tachometer(stdscr, value, center_y, center_x, title):
    # Display title with percentage in white above the tachometer
    title_text = f"{title}: {int(value)}%"  # Convert percentage to integer
    stdscr.attron(curses.color_pair(6))  # White color for title and percentage
    stdscr.addstr(center_y - RADIUS - 2, center_x - len(title_text) // 2, title_text)
    stdscr.attroff(curses.color_pair(6))

    # Draw the arc for the gauge with different colors, with 2-character thickness
    for angle in range(150, -151, -5):  # Adjust granularity with step size
        radians = math.radians(angle)
        
        # Outer arc position
        outer_y = int(center_y - RADIUS * math.cos(radians))  # Flip vertically
        outer_x = int(center_x - RADIUS * math.sin(radians) * X_SCALE_FACTOR)
        
        # Inner arc position, slightly closer to the center
        inner_y = int(center_y - (RADIUS - 1) * math.cos(radians))
        inner_x = int(center_x - (RADIUS - 1) * math.sin(radians) * X_SCALE_FACTOR)
        
        # Determine the color based on the angle/percentage
        percent = 100 - ((angle + 150) / 300 * 100)
        if percent <= 50:
            color = curses.color_pair(1)  # Green on Green
        elif percent <= 70:
            color = curses.color_pair(2)  # Yellow on Yellow
        elif percent <= 90:
            color = curses.color_pair(3)  # Red on Yellow for Orange effect
        else:
            color = curses.color_pair(4)  # Red on Red
        
        # Draw both outer and inner parts of the arc to make it thicker
        stdscr.attron(color)
        stdscr.addch(outer_y, outer_x, '#')
        stdscr.addch(inner_y, inner_x, '#')
        stdscr.attroff(color)
    
    # Map the value to an angle from -150 (0%) to +150 (100%)
    angle = 150 - (value / 100) * 300  # 300 degrees total
    radians = math.radians(angle)
    
    # Draw the full needle from the center to the endpoint in red-on-red
    stdscr.attron(curses.color_pair(5))  # Red on Red needle
    r = 1.0
    while r < RADIUS:  # Increment by 0.5 for smoother needle
        needle_y = int(center_y - r * math.cos(radians))  # Flip vertically
        needle_x = int(center_x - r * math.sin(radians) * X_SCALE_FACTOR)
        stdscr.addch(needle_y, needle_x, '*')
        r += 0.5
    stdscr.attroff(curses.color_pair(5))

def draw_progress_bar(stdscr, percentage, elapsed_time):
    height, width = stdscr.getmaxyx()
    bar_width = width - 4  # Leave space for padding
    fill_width = int((percentage / 100) * bar_width)

    # Progress bar label with time and percentage
    progress_label = f"Time: {elapsed_time:.1f}s / {TMAX}s - {percentage:.1f}%"
    stdscr.addstr(height - 3, 2, progress_label)

    # Draw the progress bar
    stdscr.addstr(height - 2, 2, "[" + "=" * fill_width + " " * (bar_width - fill_width) + "]")

def main(stdscr):
    curses.curs_set(0)  # Hide cursor
    curses.start_color()
    
    # Initialize color pairs
    curses.init_pair(1, curses.COLOR_GREEN, curses.COLOR_GREEN)   # Green on Green
    curses.init_pair(2, curses.COLOR_YELLOW, curses.COLOR_YELLOW) # Yellow on Yellow
    curses.init_pair(3, curses.COLOR_RED, curses.COLOR_YELLOW)    # Approximate Orange
    curses.init_pair(4, curses.COLOR_RED, curses.COLOR_RED)       # Red on Red for arc
    curses.init_pair(5, curses.COLOR_RED, curses.COLOR_RED)       # Red on Red for needle
    curses.init_pair(6, curses.COLOR_WHITE, curses.COLOR_BLACK)   # White for title text

    height, width = stdscr.getmaxyx()
    
    # Center points for left and right tachometers
    left_center_x = width // 4
    right_center_x = 3 * width // 4
    center_y = height // 2

    start_time = time.time()
    while True:
        # Calculate elapsed time and percentage based on TMAX
        elapsed_time = time.time() - start_time
        percentage = min((elapsed_time / TMAX) * 100, 100)  # Cap at 100%

        # Clear and redraw screen
        stdscr.clear()
        draw_tachometer(stdscr, percentage, center_y, left_center_x, "INPUT BITRATE")
        draw_tachometer(stdscr, percentage, center_y, right_center_x, "OUTPUT BITRATE")
        draw_progress_bar(stdscr, percentage, elapsed_time)
        stdscr.refresh()  # Refresh the entire screen

        # Exit loop if time limit reached
        if elapsed_time >= TMAX:
            break
        time.sleep(0.05)  # Adjust update speed

curses.wrapper(main)
