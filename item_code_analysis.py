import streamlit as st 
import pandas as pd
from model import ModelItemCodeAnalysis
import plotly.graph_objects as go
from plotly.graph_objects import Bar, Scatter, Figure
import requests
import pickle

def app():

    st.title("Item Code Analysis")


    pricing_basis = st.radio("Select Pricing Basis", ["GAIA Indonesia", 'GAIA Australia'], key='pricing_basis', index=1)
    clinic_basis = st.selectbox("Select Data", ["Acre Clarity Data"])

    with st.popover("details"):
        st.write("**Pricing Basis** will determine which pricing list to use for the analysis (whether it is Indonesia or Australia)"
                ". It is important to note that this analysis is based on our own pricing list and baseline durations, not the actual pricing list or durations used by the clinic"
                ", as in future the model is intended to serve clinic belong to our group.")
        st.write("The data being extracted and used from the selected clinic are Demand and Dentist Provider details for each Item Code."
                "In this case the selected clinic is as we **Select Data**")

    if pricing_basis == "GAIA Indonesia":
        pricing_basis = "Indonesia"
    elif pricing_basis == "GAIA Australia":
        pricing_basis = "Australia"    
        
    def currency_formatter(value, pricing_basis):
        if pricing_basis == "Indonesia":
            return f"Rp {value:,.0f}"
        elif pricing_basis == "Australia":
            return f"${value:,.0f}"


    clinic_data_df = pd.read_csv("cleaned_clarity_data_2023.csv", dtype={'Code': str})

    st.markdown("## Overall Analysis")

    # Instantiate the class and load data
    model = ModelItemCodeAnalysis()

    result_df = model.calculate_contributions(clinic_data_df, pricing_basis)
    # result_df = result_df['Code'].astype(str)   

    model = ModelItemCodeAnalysis()
    # result_df = pd.read_csv('result_df.csv', dtype={'Code': str})


    st.metric(label="Total Demand", value=f"{result_df['Demand'].sum():,}")
    col2 , col3 = st.columns(2)


    with col2:
        st.metric(label="Total Revenue", value=currency_formatter(result_df['Revenue Contribution'].sum(), pricing_basis))
        
    with col3:
        st.metric(label="Total Profit", value=currency_formatter(result_df['Profit Contribution'].sum(), pricing_basis))

    with st.popover("details"):
        st.write("The total demand is the sum of all demand for each item code, while the total revenue and profit are the sum of revenue and profit contribution for each item code times multiplied by its demand respectively")

    st.markdown("#### Per Item Code Performance")
    st.write("Below are the performance of each item code based on Demand, Revenue Contribution, and Profit Contribution")
    st.dataframe(result_df, width=750, use_container_width=True, hide_index=True)

    st.markdown("#### Pareto Analysis")
    st.write("Below are the Pareto Analysis for overall performance based on Demand, Revenue Contribution, and Profit Contribution")

    tab1, tab2, tab3 = st.tabs(["By Demand", "By Revenue", "By Profit"])

    # Create the request payload with a list of basis
    payload = {
        'df': result_df.to_dict(),
        'basis': ['Demand', 'Revenue Contribution', 'Profit Contribution']  # List of basis for the loop
    }

    # Send POST request to the Flask server
    response = requests.post('http://127.0.0.1:5000/create_pareto', json=payload)

    # Extract and decode the pickled figures
    pickled_figures = response.json()

    # Load and save each figure from the response
    figures = {}  # Dictionary to store the figures as fig_1, fig_2, fig_3

    # Iterate over the pickled figures, unpickle them, and save to the dictionary
    for i, (basis, pickled_fig) in enumerate(pickled_figures.items(), start=1):
        fig = pickle.loads(pickled_fig.encode('latin1'))
        figures[f'fig_{i}'] = fig  # Save the figure in the dictionary with dynamic names like fig_1, fig_2, etc.


    # Now you can access the figures as figures['fig_1'], figures['fig_2'], etc.


    with tab1:

        st.plotly_chart(figures['fig_1'])
        st.caption("Showing only top 20 items based on Demand")


    with tab2:

        st.plotly_chart(figures['fig_2'])
        st.caption("Showing only top 20 items based on Revenue Contribution")
    with tab3:
        
        st.plotly_chart(figures['fig_3'])
        st.caption("Showing only top 20 items based on Profit Contribution")

    st.divider()

    st.markdown("## Per Dentist Analysis")

    st.markdown("#### Dentist Performance Comparison")
    st.write("Below are the performance comparison between dentists based on Demand, Revenue Contribution, and Profit Contribution")

    st.markdown("**Table Summary**")
    dentist_comparison = model.calculate_contributions_by_provider(clinic_data_df, pricing_basis)

    st.dataframe(dentist_comparison, width=650, hide_index=True)

    tab1, tab2, tab3 = st.tabs(["By Demand", "By Revenue", "By Profit"])

    with tab1:

        comparison_bar_chart_demand = model.create_horizontal_bar_chart(dentist_comparison, "Demand")

        st.plotly_chart(comparison_bar_chart_demand)
        

    with tab2:
        
        comparison_bar_chart_revenue = model.create_horizontal_bar_chart(dentist_comparison, "Revenue Contribution")

        st.plotly_chart(comparison_bar_chart_revenue)
        

    with tab3:
            
        comparison_bar_chart_profit = model.create_horizontal_bar_chart(dentist_comparison, "Profit Contribution")

        st.plotly_chart(comparison_bar_chart_profit)
        
    with st.popover("understanding horizontal bar chart", use_container_width=True):
        st.write("The horizontal bar chart above shows the performance comparison between dentists based on Demand, Revenue Contribution, and Profit Contribution. The length of the bar represents the value of each basis, and the color of the bar indicates the dentist's name")
        
    scatter_fig = model.create_bubble_scatter(dentist_comparison)
    st.plotly_chart(scatter_fig)
    with st.popover("understanding scatter bubble plot", use_container_width=True):
        st.write("The scatter bubble plot above compares dentist performance based on two metrics: Duration per Demand and Revenue per Demand.")
        st.write("Each dentist is represented by a bubble, with the size indicating their total demand—the larger the bubble, the greater the demand.")
        st.write("A dentist's bubble located in the top-left area of the plot indicates strong performance, as it reflects high average revenue per demand and low average duration per demand (generating more in less time and working more efficiently overall).")
        st.write("Conversely, a bubble in the bottom-right area suggests poor performance, with the dentist generating lower revenue per demand and taking longer to complete tasks.")
        
        st.write("These insights can be used as a basis for evaluating dentist performance or for salary and wage justification.")
    st.markdown("#### Pareto Analysis (Per Dentist)")
    st.write("Below are the Pareto Analysis for each dentist based on Demand, Revenue Contribution, and Profit Contribution")

    dentist_name = st.selectbox("Select Dentist", clinic_data_df["Provider"].unique())

    filtered_df = clinic_data_df[clinic_data_df["Provider"] == dentist_name]

    filtered_df = model.calculate_contributions(filtered_df, pricing_basis)

    # Create the request payload with a list of basis
    payload = {
        'df': filtered_df.to_dict(),
        'basis': ['Demand', 'Revenue Contribution', 'Profit Contribution']  # List of basis for the loop
    }

    # Send POST request to the Flask server
    response = requests.post('http://127.0.0.1:5000/create_pareto', json=payload)

    # Extract and decode the pickled figures
    pickled_figures = response.json()

    # Load and save each figure from the response
    figures_dentist = {}  # Dictionary to store the figures as fig_1, fig_2, fig_3

    # Iterate over the pickled figures, unpickle them, and save to the dictionary
    for i, (basis, pickled_fig) in enumerate(pickled_figures.items(), start=1):
        fig = pickle.loads(pickled_fig.encode('latin1'))
        figures_dentist[f'fig_{i}'] = fig  # Save the figure in the dictionary with dynamic names like fig_1, fig_2, etc.


    tab1, tab2, tab3 = st.tabs(["By Demand", "By Revenue", "By Profit"])

    with tab1:

        st.plotly_chart(figures_dentist['fig_1'])
        st.caption("Showing only top 20 items based on Demand")


    with tab2:

        st.plotly_chart(figures_dentist['fig_2'])
        st.caption("Showing only top 20 items based on Revenue Contribution")
        
    with tab3:
        
        st.plotly_chart(figures_dentist['fig_3'])
        st.caption("Showing only top 20 items based on Profit Contribution")


        