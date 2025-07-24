""" RunPod handler for AI Cover song generation """

import runpod
import subprocess
import os
import time


def handler(job):
    """ Handler function that will be used to process jobs. """
    job_input = job['input']

    # Required input fields
    rvc_dirname = job_input.get('rvc_dirname')
    song_input = job_input.get('song_input')
    pitch_change_all = str(job_input.get('pitch_change_all', 0))

    if not rvc_dirname or not song_input:
        return {"error": "Missing required fields: 'rvc_dirname' or 'song_input'"}

    # Optional: validate if the file exists before running
    if not os.path.exists(song_input):
        return {"error": f"Input file not found: {song_input}"}

    # Construct command
    command = [
        "python3", "run.py",
        "--rvc_dirname", rvc_dirname,
        "--song_input", song_input,
        "--pitch_change_all", pitch_change_all
    ]

    time_start = time.time()
    result = subprocess.run(command, capture_output=True, text=True)
    duration = time.time() - time_start

    return {
        "returncode": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
        "time_taken": duration
    }

runpod.serverless.start({"handler": handler})
