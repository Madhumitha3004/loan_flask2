from flask import Flask
from auth.auth import auth_bp, init_db
from api.api import api_bp

app = Flask(__name__)
app.secret_key = 'your_secret_key'

app.register_blueprint(auth_bp)
app.register_blueprint(api_bp, url_prefix='/api')

init_db()

if __name__ == '__main__':
    app.run(debug=True)
