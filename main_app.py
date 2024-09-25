import streamlit as st
import pandas as pd
import numpy as np
from model import Model

# Load the treatment list from CSV
treatment = pd.read_csv('cleaned_treatment.csv')

# Select the relevant columns
treatment = treatment[['Treatment']]

# Add a new column 'include?' with default False values (unchecked)
treatment['include?'] = False

# Split the DataFrame into two halves (if odd number, second part will have one less row)
half_len = len(treatment) // 2 + len(treatment) % 2  # Handle odd number of rows

# Split the treatment DataFrame into two parts
treatment_1 = treatment.iloc[:half_len].reset_index(drop=True)
treatment_2 = treatment.iloc[half_len:].reset_index(drop=True)

# Concatenate the columns to create the desired layout
treatment_combined = pd.concat([treatment_1, treatment_2], axis=1)

# Rename columns to reflect the split layout
treatment_combined.columns = ['Treatment_1', 'Include_1', 'Treatment_2', 'Include_2']

# set the streamlit on wide mode
st.set_page_config(layout="wide")

# Streamlit page setup
st.title("Upselling Recommendation Strategy")

st.write("This model helps us to analyze how upselling treatments would yield into increase in revenue and profit as well as it analyzes the performance of each item code before and after the upselling strategy.")

with st.container(border=True):

    st.header("Demand Definition")
    
    st.markdown("#### Treatment Selection")
    # Editable data editor for treatments with a checkbox
    treatment_editor = st.data_editor(treatment_combined, num_rows="dynamic", width=1000)

    st.divider()

    st.markdown("#### Demand Allocation")
    
    # Input for total demand
    total_demand = st.number_input("Total Demand", min_value=0, step=1, value=1000)

    # Calculate how many treatments are checked in the previous data editor
    checked_treatments_1 = treatment_editor[treatment_editor['Include_1'] == True]['Treatment_1']
    checked_treatments_2 = treatment_editor[treatment_editor['Include_2'] == True]['Treatment_2']
    total_checked = len(checked_treatments_1) + len(checked_treatments_2)

    # Only create the demand DataFrame if at least one treatment is selected
    if total_checked > 0:
        # Calculate demand per treatment
        demand_per_treatment = np.ceil(total_demand / total_checked)
        
        # Combine the selected treatments into one DataFrame
        demand_df = pd.DataFrame({
            'Treatment': pd.concat([checked_treatments_1, checked_treatments_2]).reset_index(drop=True),
            'Demand': demand_per_treatment
        })
        
        # Display the new DataFrame with calculated demands
        st.write("Demand Distribution:")
        st.dataframe(demand_df, use_container_width=False, width=500)
    else:
        st.warning("Please select at least one treatment to allocate demand.")
        
    st.divider()
    
    
if total_checked > 0:

    with st.container(border=True):

        st.header("Performance Before Upselling")

        # Assuming demand_df is already calculated
        model_instance = Model()

        # Calculate the existing condition using the demand DataFrame
        merged_df, total_revenue, total_cost, total_duration = model_instance.calculate_existing_condition(demand_df)

        with st.expander("Show Calculated Data", expanded=False):
        # Display the results

            st.dataframe(merged_df, use_container_width=True)
            
            st.caption("Note: All price and cost are in $, and duration is in minutes")

        col1, col2 = st.columns(2)

        with col1:
            st.metric("Total Revenue", f"$ {total_revenue:,.0f}")
            st.metric("Total Cost", f"$ {total_cost:,.0f}")
            

        with col2:
            total_profit = total_revenue - total_cost
            st.metric("Total Profit", f"$ {total_profit:,.0f}")
            
        
        st.divider()
        
        st.markdown("#### Item Code Performance Analysis")
        
        # Analyze item code performance using the demand DataFrame
        item_code_df = model_instance.analyze_item_code_performance(demand_df)
        

        
        # Display the results
        st.write("Item Code Performance Analysis:")
        st.dataframe(item_code_df, hide_index=True, width=500)
        

        st.pyplot(model_instance.create_combo_chart(item_code_df))

    st.divider()

    with st.container( border=True):
        
        st.header("Performance After Upselling")
        
        st.write("Please define the conversion rate of upselling for each of the selected treatments.")
        
        conversion_rate_upsell = st.number_input("Collective Conversion Rate (%)", min_value=0, max_value=100, step=1, value=20, help="The percentage of customers who accept the upsell offer.")
        
        upsell_configuration = model_instance.create_upsell_configuration_dataframe(demand_df, conversion_rate_upsell)
        
        upsell_configuration = st.data_editor(upsell_configuration)

        # Display Final Metrics using st.metric
        new_total_revenue, new_total_cost, new_total_duration, updated_df = model_instance.calculate_existing_and_upsell(demand_df, upsell_configuration)
        new_total_profit = new_total_revenue - new_total_cost

        with st.expander("New Treatment Demand and Calculation", expanded =False):
            st.dataframe(updated_df, use_container_width=True)
            
        col1, col2 = st.columns(2)

        with col1:
            st.metric("Total Revenue", f"$ {new_total_revenue:,.0f}", delta=f"$ {(new_total_revenue - total_revenue) / total_revenue:.2%}")
            st.metric("Total Cost", f"$ {new_total_cost:,.0f}", delta=f"$ {(new_total_cost - total_cost)/ total_cost:.2%}")

        with col2:
            st.metric("Total Profit", f"$ {(new_total_profit):,.0f}", delta=f"$ {(new_total_profit - total_profit)/ total_profit:.2%}")
        
        
        st.divider()
        
        st.markdown("#### Item Code Performance Analysis")
        
        # Analyze item code performance using the demand DataFrame
        item_code_df = model_instance.analyze_item_code_performance(updated_df)
        
        
        # Display the results
        st.write("Item Code Performance Analysis:")
        st.dataframe(item_code_df, hide_index=True, width=500)
        

        st.pyplot(model_instance.create_combo_chart(item_code_df))