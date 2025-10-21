# main.py (Example of an orchestrator)

import os
import subprocess
import sys

def run_script(script_name):
    """Executes an external Python script."""
    print(f"\n--- Running {script_name} ---")
    # Use sys.executable to ensure the correct Python environment is used
    result = subprocess.run([sys.executable, script_name], check=True)
    if result.returncode == 0:
        print(f"--- {script_name} finished successfully. ---\n")
    else:
        print(f"--- {script_name} FAILED! Exiting. ---")
        sys.exit(1)

if __name__ == "__main__":
    
    run_script('datafile.py') 

    if not os.path.exists('emotion_detection_model.h5'):
        run_script('model.py')
    else:
        print("Model file already exists. Skipping training. Delete the file to retrain.")

    run_script('webcam_detector.py')