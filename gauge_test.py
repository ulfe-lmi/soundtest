import time
from gauge_display import GaugeDisplay

def main():
    gauge = GaugeDisplay()
    gauge.gauge_initialize("INPUT BITRATE", "OUTPUT BITRATE")
    
    # Run the gauge for 10 seconds
    TMAX = 10
    start_time = time.time()
    while True:
        elapsed_time = time.time() - start_time
        percentage = min((elapsed_time / TMAX) * 100, 100)  # Capped at 100%
        
        # Update the gauge
        gauge.update_display(percentage, percentage, percentage, elapsed_time)
        
        # Exit loop if time limit reached
        if elapsed_time >= TMAX:
            break
        time.sleep(0.05)
    
    gauge.gauge_delete()

if __name__ == "__main__":
    main()
