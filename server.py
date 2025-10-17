import http.server, socketserver, os

PORT = int(os.environ.get("PORT", 8080))
with socketserver.TCPServer(("", PORT), http.server.SimpleHTTPRequestHandler) as httpd:
    print(f"Serving dummy web app on port {PORT}")
    httpd.serve_forever()
