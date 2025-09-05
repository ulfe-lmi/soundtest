"""
GaugeDisplay - A terminal-based dual tachometer with a progress bar for tracking percentages.
Usage:
1. Import the GaugeDisplay class.
2. Initialize with `gauge_initialize(left_title, right_title)` to show at 0% for each gauge and 00:00:00 for time.
3. Update the display with `gauge_update(percentage1, percentage2, percentage3, elapsed_time)`.
4. Call `gauge_delete` to clear the gauge and progress bar upon completion.

Example:
from gauge_display import GaugeDisplay
gauge = GaugeDisplay()
gauge.gauge_initialize("INPUT BITRATE", "OUTPUT BITRATE")
for i in range(101):
    gauge.update_display(i, i, i, i)
    time.sleep(0.1)
gauge.gauge_delete()
"""

import curses
import time
import math

# Constants
X_SCALE_FACTOR = 1.5
RADIUS = 10

class GaugeDisplay:
    def __init__(self):
        self.screen = None
        self.left_title = ""
        self.right_title = ""

    def gauge_initialize(self, left_title="LEFT GAUGE", right_title="RIGHT GAUGE"):
        # Initialize the curses screen and set titles
        self.screen = curses.initscr()
        self.left_title = left_title
        self.right_title = right_title
        curses.curs_set(0)
        curses.start_color()
        
        # Initialize color pairs
        curses.init_pair(1, curses.COLOR_GREEN, curses.COLOR_GREEN)
        curses.init_pair(2, curses.COLOR_YELLOW, curses.COLOR_YELLOW)
        curses.init_pair(3, curses.COLOR_RED, curses.COLOR_YELLOW)  # Approximated orange
        curses.init_pair(4, curses.COLOR_RED, curses.COLOR_RED)
        curses.init_pair(5, curses.COLOR_RED, curses.COLOR_RED)
        curses.init_pair(6, curses.COLOR_WHITE, curses.COLOR_BLACK)
        
        # Show initial display at 0%
        self.update_display(0, 0, 0, 0)

    def update_display(self, percentage1, percentage2, percentage3, elapsed_time):
        height, width = self.screen.getmaxyx()
        
        # Center points for left and right tachometers
        left_center_x = width // 4
        right_center_x = 3 * width // 4
        center_y = height // 2
        
        # Convert elapsed time to hh:mm:ss format
        formatted_time = time.strftime("%H:%M:%S", time.gmtime(elapsed_time))
        
        # Clear screen
        self.screen.clear()
        
        # Draw tachometers with specified titles
        self._draw_tachometer(percentage1, center_y, left_center_x, self.left_title)
        self._draw_tachometer(percentage2, center_y, right_center_x, self.right_title)
        
        # Draw progress bar
        self._draw_progress_bar(percentage3, formatted_time)
        
        # Refresh screen
        self.screen.refresh()

    def gauge_delete(self):
        # Clear screen and end curses mode
        self.screen.clear()
        curses.endwin()
    
    def _draw_tachometer(self, value, center_y, center_x, title):
        title_text = f"{title}: {int(value)}%"
        self.screen.attron(curses.color_pair(6))
        self.screen.addstr(center_y - RADIUS - 2, center_x - len(title_text) // 2, title_text)
        self.screen.attroff(curses.color_pair(6))

        for angle in range(150, -151, -5):
            radians = math.radians(angle)
            outer_y = int(center_y - RADIUS * math.cos(radians))
            outer_x = int(center_x - RADIUS * math.sin(radians) * X_SCALE_FACTOR)
            inner_y = int(center_y - (RADIUS - 1) * math.cos(radians))
            inner_x = int(center_x - (RADIUS - 1) * math.sin(radians) * X_SCALE_FACTOR)
            
            percent = 100 - ((angle + 150) / 300 * 100)
            color = (curses.color_pair(1) if percent <= 50 else
                     curses.color_pair(2) if percent <= 70 else
                     curses.color_pair(3) if percent <= 90 else
                     curses.color_pair(4))

            self.screen.attron(color)
            self.screen.addch(outer_y, outer_x, '#')
            self.screen.addch(inner_y, inner_x, '#')
            self.screen.attroff(color)
        
        angle = 150 - (value / 100) * 300
        radians = math.radians(angle)
        self.screen.attron(curses.color_pair(5))
        
        r = 1.0
        while r < RADIUS:
            needle_y = int(center_y - r * math.cos(radians))
            needle_x = int(center_x - r * math.sin(radians) * X_SCALE_FACTOR)
            self.screen.addch(needle_y, needle_x, '*')
            r += 0.5
        self.screen.attroff(curses.color_pair(5))

    def _draw_progress_bar(self, percentage, formatted_time):
        height, width = self.screen.getmaxyx()
        bar_width = width - 4
        fill_width = int((percentage / 100) * bar_width)
        
        progress_label = f"Time: {formatted_time} - {int(percentage)}%"
        self.screen.addstr(height - 3, 2, progress_label)
        self.screen.addstr(height - 2, 2, "[" + "=" * fill_width + " " * (bar_width - fill_width) + "]")
