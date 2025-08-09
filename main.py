# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import pandas as pd
from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)
data = pd.read_csv('Cleaned_data.csv')
pipe = pickle.load(open("RidgeModel.pkl", 'rb'))



@app.route('/')
def index():

    locations= sorted(data['location'].unique())
    return render_template('index.html', locations=locations)


@app.route('/predict', methods=['POST'])
def predict():
    # Read form values
    location = request.form.get('location')       # str
    sqft = float(request.form.get('total_sqft'))  # float
    bath = float(request.form.get('bath'))        # float
    bhk = int(request.form.get('bhk'))            # int

    # Create DataFrame EXACTLY like training
    input_df = pd.DataFrame({
        'location': [location],
        'total_sqft': [sqft],
        'bath': [bath],
        'bhk': [bhk]
    })

    print("DEBUG INPUT DF:")
    print(input_df.dtypes)
    print(input_df)

    # Predict
    prediction = pipe.predict(input_df)[0] * 1e5
    return str(np.round(prediction, 2))


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    app.run(debug=True, port=5001)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
