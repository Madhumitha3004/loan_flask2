from flask import Flask
from auth.auth import auth_bp, init_db
from api.api import api_bp

app = Flask(__name__)
app.secret_key = 'your_secret_key'

app.register_blueprint(auth_bp)
app.register_blueprint(api_bp, url_prefix='/api')

init_db()

import os

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))  # Use Render's port
    app.run(host='0.0.0.0', port=port, debug=True)

