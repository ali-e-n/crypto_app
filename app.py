from flask import Flask
from routes import setup_routes
from flask_login import LoginManager, UserMixin

app = Flask(__name__)
app.secret_key = "supersecretkey"  # Change in production

login_manager = LoginManager()
login_manager.init_app(app)

# Dummy admin user
class User(UserMixin):
    def __init__(self, id):
        self.id = id
        self.name = "admin"
        self.password = "admin123"

users = {"admin": User("admin")}

@login_manager.user_loader
def load_user(user_id):
    return users.get(user_id)

setup_routes(app, users)

if __name__ == "__main__":
    app.run(debug=True)
