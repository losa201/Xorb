
import asyncio
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
import uvicorn
import subprocess

app = FastAPI(title="XORB RedOps Console", version="1.0.0")

REDOPS_HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>XORB RedOps Console</title>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <style>
        body { font-family: Arial, sans-serif; margin: 0; padding: 20px; background: #1a1a1a; color: #f0f0f0; }
        .header { background: #e74c3c; color: white; padding: 20px; border-radius: 8px; margin-bottom: 20px; text-align: center; }
        .header h1 { margin: 0; }
        .console-grid { display: grid; grid-template-columns: 1fr 2fr; gap: 20px; }
        .control-panel { background: #2c3e50; border-radius: 8px; padding: 20px; }
        .mission-log { background: #2c3e50; border-radius: 8px; padding: 20px; height: 400px; overflow-y: scroll; }
        .btn { background: #e74c3c; color: white; border: none; padding: 10px 20px; border-radius: 4px; cursor: pointer; width: 100%; margin-bottom: 10px; }
        .btn:hover { background: #c0392b; }
        .log-entry { border-bottom: 1px solid #34495e; padding: 5px 0; }
    </style>
</head>
<body>
    <div class="header">
        <h1>RedOps Campaign Console</h1>
    </div>
    <div class="console-grid">
        <div class="control-panel">
            <h2>Launch Simulation</h2>
            <button class="btn" onclick="launchMission()">Launch Mission</button>
            <div id="mission-status">Status: Idle</div>
        </div>
        <div class="mission-log" id="mission-log">
            <div class="log-entry">[SYSTEM] Console Initialized.</div>
        </div>
    </div>

    <script>
        const missionLog = document.getElementById('mission-log');
        const missionStatus = document.getElementById('mission-status');

        function log(message) {
            const entry = document.createElement('div');
            entry.className = 'log-entry';
            entry.textContent = message;
            missionLog.appendChild(entry);
            missionLog.scrollTop = missionLog.scrollHeight;
        }

        async function launchMission() {
            log('[SYSTEM] Launching new Red Team mission...');
            missionStatus.textContent = 'Status: Running';
            
            const response = await fetch('/launch-simulation');
            const reader = response.body.getReader();
            const decoder = new TextDecoder();

            while (true) {
                const { value, done } = await reader.read();
                if (done) break;
                const chunk = decoder.decode(value);
                log(chunk);
            }

            missionStatus.textContent = 'Status: Completed';
        }
    </script>
</body>
</html>
"""

@app.get("/", response_class=HTMLResponse)
async def red_ops_console():
    """Serve the RedOps Console HTML"""
    return REDOPS_HTML

@app.get("/launch-simulation")
async def launch_simulation():
    process = await asyncio.create_subprocess_exec(
        'python', 'simulation/engine/red_vs_blue_simulator.py',
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE
    )

    async def stream_logs(stream):
        while True:
            line = await stream.readline()
            if not line:
                break
            yield line

    return asyncio.as_completed([stream_logs(process.stdout)])

@app.get("/health")
async def health():
    """Health check"""
    return {"status": "healthy", "service": "xorb_redops_console"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=3002)
