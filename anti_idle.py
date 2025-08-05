import math
import time
import sys
import os

# Try different mouse control methods
try:
    # Method 1: pyautogui (most compatible)
    import pyautogui
    pyautogui.FAILSAFE = False  # Disable failsafe
    CONTROL_METHOD = "pyautogui"
except ImportError:
    try:
        # Method 2: pynput
        from pynput import mouse
        from pynput.mouse import Button, Listener
        CONTROL_METHOD = "pynput"
    except ImportError:
        print("Please install pyautogui: pip install pyautogui")
        sys.exit(1)

class CircularCursor:
    def __init__(self):
        self.running = False
        self.radius = 100
        self.angle = 0
        self.speed = 0.1
        
        if CONTROL_METHOD == "pyautogui":
            # Get screen center
            screen_width, screen_height = pyautogui.size()
            self.center_x = screen_width // 2
            self.center_y = screen_height // 2
        else:
            # pynput method
            self.mouse_controller = mouse.Controller()
            pos = self.mouse_controller.position
            self.center_x = pos[0]
            self.center_y = pos[1]
        
        print("Circular Cursor Controller")
        print("=========================")
        print(f"Using {CONTROL_METHOD} for mouse control")
        print(f"Center: ({self.center_x}, {self.center_y})")
        print("Press ENTER to start/stop")
        print("Press 'q' then ENTER to quit")
        print("Status: STOPPED")
    
    def move_cursor(self):
        while self.running:
            # Calculate new position
            x = self.center_x + int(self.radius * math.cos(self.angle))
            y = self.center_y + int(self.radius * math.sin(self.angle))
            
            # Move cursor based on available method
            if CONTROL_METHOD == "pyautogui":
                pyautogui.moveTo(x, y, duration=0)
            else:
                self.mouse_controller.position = (x, y)
            
            # Update angle
            self.angle += self.speed
            if self.angle >= 2 * math.pi:
                self.angle = 0
            
            time.sleep(0.03)
    
    def run(self):
        try:
            while True:
                user_input = input().strip().lower()
                
                if user_input == 'q':
                    print("Exiting...")
                    self.running = False
                    break
                elif user_input == '':  # Enter key
                    self.running = not self.running
                    if self.running:
                        print("Status: RUNNING - cursor moving in circle")
                        # Start movement
                        import threading
                        threading.Thread(target=self.move_cursor, daemon=True).start()
                    else:
                        print("Status: STOPPED")
                else:
                    print("Press ENTER to toggle, 'q' + ENTER to quit")
                    
        except KeyboardInterrupt:
            print("\nExiting...")
            self.running = False

if __name__ == "__main__":
    app = CircularCursor()
    app.run()