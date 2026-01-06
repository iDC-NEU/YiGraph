import socket
from route import create_app, socketio

app = create_app()

def find_free_port(start_port=5089, max_tries=50):
    """Find an available port starting from start_port."""
    port = start_port
    for _ in range(max_tries):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("0.0.0.0", port))
                return port  # This port is free
            except OSError:
                port += 1
    raise RuntimeError("No available ports found.")

if __name__ == "__main__":
    port = find_free_port()
    print(f"Using available port: {port}")

    socketio.run(
        app,
        debug=True,
        host="0.0.0.0",
        port=port
    )