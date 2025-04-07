from flask import Flask, render_template, request
import numpy as np
import joblib
from pcos_model_on_pynq import score

app = Flask(__name__)

# Load necessary files
top_20_features = joblib.load("selected_features.joblib")
means = np.load("scaler_means.npy")
stds = np.load("scaler_stds.npy")

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        try:
            # Read input values
            user_input = [float(request.form[feature]) for feature in top_20_features]

            # Normalize
            user_input = np.array(user_input)
            normalized = (user_input - means) / stds

            # Predict using the model
            probabilities = score(normalized.flatten())
            predicted_class = int(np.argmax(probabilities))

            result = "PCOS Detected" if predicted_class == 1 else "No PCOS Detected"
            return render_template('index.html', features=top_20_features, result=result,
                                   prob_0=round(probabilities[0], 2), prob_1=round(probabilities[1], 2))

        except Exception as e:
            return render_template('index.html', features=top_20_features, error=str(e))

    return render_template('index.html', features=top_20_features)

if __name__ == '__main__':
    app.run(debug=True)
