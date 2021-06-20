from flask import Flask, render_template, request, render_template
import pickle
import numpy as np

model= pickle.load(open("ccd_model.pkl","rb"))

app= Flask(__name__)

@app.route('/')
def home():
  return render_template("ccd_prediction.html")

@app.route("/predict", methods=["POST","GET"])
def pred():
  features= [int(x) for x in request.form.values()]
  defaulter= model.predict(np.array([features]))
  if defaulter[0]:
    return render_template("ccd_prediction.html", pred="The person is likely to be a Credit Card Payment Defaulter next month!", c="red-text")
  else:
    return render_template("ccd_prediction.html", pred="The person is not likely to be a Credit Card Payment Defaulter next month!", c="green-text")

if __name__ == "__main__":
  app.run(debug=True)