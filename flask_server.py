from flask import Flask, request, jsonify
import pandas as pd
import plotly.graph_objects as go
import pickle
from model import ModelItemCodeAnalysis

app = Flask(__name__)


@app.route('/create_pareto', methods=['POST'])
def create_pareto():
    # Get the input JSON data
    data = request.get_json()
    
    # Convert the DataFrame from the dictionary (assume the input DataFrame is provided in JSON format)
    df = pd.DataFrame(data['df'])
    basis_list = data['basis']

    # Initialize ModelItemCodeAnalysis instance
    model = ModelItemCodeAnalysis()

    # Dictionary to hold pickled figures
    pickled_figures = {}

    # Loop through each basis and create a figure
    for basis in basis_list:
        fig = model.create_pareto_chart(df, basis)
        pickled_fig = pickle.dumps(fig)  # Pickle the Plotly figure
        pickled_figures[basis] = pickled_fig.decode('latin1')  # Decode for JSON transmission

    # Return pickled objects as a response
    return jsonify(pickled_figures)


if __name__ == '__main__':
    app.run(debug=True)
