import sys
from pyngrok import ngrok, conf

# Optional: Set auth token if provided as argument
if len(sys.argv) > 1:
    token = sys.argv[1]
    conf.get_default().auth_token = token

# Open a HTTP tunnel on the default port 8000
# If you need a different port, change it here
public_url = ngrok.connect(8000).public_url
print(f"Ngrok Tunnel Started: {public_url}")

# Keep the process alive
try:
    # Block until CTRL-C or some other event
    ngrok_process = ngrok.get_ngrok_process()
    ngrok_process.proc.wait()
except KeyboardInterrupt:
    print("Shutting down ngrok...")
    ngrok.kill()
