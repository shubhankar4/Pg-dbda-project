from flask import Flask, request, render_template
import pickle
import pandas as pd

app = Flask(__name__)

with open("model.pkl", "rb") as f:
    model = pickle.load(f)

@app.route("/")
def index():
    return render_template("form.html")

@app.route("/submit", methods=["POST"])
def submit():
    data = request.form
    data = {k: int(v) for k, v in data.items()}
    
    if has_diabetes := model.predict(pd.DataFrame([[*data.values()]], columns=data.keys())):
        return "You have diabetes..."
    return "Congrats, you don't have diabetes!"


if __name__ == '__main__':
    app.run(debug=True)