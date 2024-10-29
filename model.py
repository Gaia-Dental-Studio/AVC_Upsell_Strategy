import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
from plotly.graph_objects import Bar, Scatter, Figure
import plotly.colors as colors
import plotly.express as px


class Model:
    def __init__(self):
        # Load treatment details, cleaned treatment CSV, and cleaned item code CSV into DataFrames
        self.treatment_details_df = pd.read_csv('treatment_with_details.csv')
        self.cleaned_treatment_df = pd.read_csv('cleaned_treatment.csv')
        self.cleaned_item_code_df = pd.read_csv('cleaned_item_code.csv')  # Load item code details
        self.upsell_recommendation_df = pd.read_csv('predicted_upsell_recommendation.csv')  # Load upsell recommendations
        
        
    def pricing_basis_selection(self, pricing_basis):    
        
        if pricing_basis == 'Indonesia':
            self.treatment_details_df = self.treatment_details_df[['Treatment', 'total_price_IDR', 'total_cost_IDR', 'total_duration']]
            self.treatment_details_df.columns = ['Treatment', 'total_price', 'total_cost', 'total_duration']
            
            self.cleaned_item_code_df = self.cleaned_item_code_df[['item_number', 'duration', 'price IDR', 'cost_material IDR' ]]
            self.cleaned_item_code_df.columns = ['item_number', 'duration', 'price', 'cost_material']
            
        elif pricing_basis == 'Australia':
            self.treatment_details_df = self.treatment_details_df[['Treatment', 'total_price_AUD', 'total_cost_AUD', 'total_duration']]
            self.treatment_details_df.columns = ['Treatment', 'total_price', 'total_cost', 'total_duration']
        
            self.cleaned_item_code_df = self.cleaned_item_code_df[['item_number', 'duration', 'price AUD', 'cost_material AUD' ]]
            self.cleaned_item_code_df.columns = ['item_number', 'duration', 'price', 'cost_material']
        
        
    def format_currency(self, value, pricing_basis):
        if pricing_basis == 'Indonesia':
            return f'Rp{value:,.0f}'
        elif pricing_basis == 'Australia':
            return f'${value:,.0f}'
 
    
    def calculate_existing_condition(self, demand_df):
        
        # if pricing_basis == 'Indonesia':
        #     self.treatment_details_df = self.treatment_details_df[['Treatment', 'total_price_IDR', 'total_cost_IDR', 'total_duration']]
        #     self.treatment_details_df.columns = ['Treatment', 'total_price', 'total_cost', 'total_duration']
            
        # elif pricing_basis == 'Australia':
        #     self.treatment_details_df = self.treatment_details_df[['Treatment', 'total_price_AUD', 'total_cost_AUD', 'total_duration']]
        #     self.treatment_details_df.columns = ['Treatment', 'total_price', 'total_cost', 'total_duration']
        
        
        # Merge demand_df with self.treatment_details_df to align treatment names
        merged_df = pd.merge(demand_df, self.treatment_details_df, left_on='Treatment', right_on='Treatment', how='inner')
        
        # Calculate total revenue by multiplying total_price by Demand
        merged_df['Total_Revenue'] = merged_df['total_price'] * merged_df['Demand']
        
        # Optionally, you can calculate other values such as total cost, total duration, etc.
        merged_df['Total_Cost'] = merged_df['total_cost'] * merged_df['Demand']
        merged_df['Total_Duration'] = merged_df['total_duration'] * merged_df['Demand']
        
        # You can sum these values to get overall totals if needed
        total_revenue = merged_df['Total_Revenue'].sum()
        total_cost = merged_df['Total_Cost'].sum()
        total_duration = merged_df['Total_Duration'].sum()
        
        merged_df.columns = ['Treatment', 'Demand', 'Price per Treatment', "Cost per Treatment", "Duration per Treatment", "Total Revenue", "Total Cost", "Total Duration"]
        
        # Return the merged DataFrame with calculated columns and overall totals
        return merged_df, total_revenue, total_cost, total_duration
    
    def get_revenue_contribution_column(self, pricing_basis):
        if pricing_basis == 'Australia':
            return 'Revenue Contribution ($)'
        elif pricing_basis == 'Indonesia':
            return 'Revenue Contribution (Rp.)'
        else:
            raise ValueError("Invalid pricing_basis. Choose either 'Australia' or 'Indonesia'.")


    def analyze_item_code_performance(self, demand_df, pricing_basis):
        # Merge demand_df with cleaned_treatment_df on Treatment
        merged_df = pd.merge(demand_df, self.cleaned_treatment_df[['Treatment', 'cleaned_item_numbers']], on='Treatment', how='inner')
        
        # Convert cleaned_item_numbers column (which is a string of lists) into actual lists
        merged_df['cleaned_item_numbers'] = merged_df['cleaned_item_numbers'].apply(eval)  # Safely convert the string representation to a list
        
        # Create a dictionary to hold the item code counts
        item_code_counts = {}
        
        # Iterate through each row to calculate item code demand
        for _, row in merged_df.iterrows():
            demand = row['Demand']
            item_codes = row['cleaned_item_numbers']
            
            for item_code in item_codes:
                if item_code in item_code_counts:
                    item_code_counts[item_code] += demand
                else:
                    item_code_counts[item_code] = demand
        
        # Convert the dictionary into a DataFrame
        item_code_df = pd.DataFrame(item_code_counts.items(), columns=['Item Code', 'Total Count'])
        
        # Merge item_code_df with cleaned_item_code_df to get price for each item code
        item_code_df = pd.merge(item_code_df, self.cleaned_item_code_df[['item_number', 'price']], left_on='Item Code', right_on='item_number', how='left')
        
        revenue_contribution_column = self.get_revenue_contribution_column(pricing_basis)
        
        # Calculate the revenue contribution for each item code
        item_code_df[revenue_contribution_column] = item_code_df['Total Count'] * item_code_df['price']
        
        # Drop the 'item_number' column after the merge (since it's redundant)
        item_code_df.drop(columns=['item_number'], inplace=True)
        
        item_code_df.columns = ['Item Code', 'Total Count', 'Price per Item', revenue_contribution_column]
        
        return item_code_df

    def create_combo_chart(self, item_code_df, pricing_basis):
        
        # Get the appropriate revenue contribution column based on pricing_basis
        revenue_contribution_column = self.get_revenue_contribution_column(pricing_basis)
        
        # Sort item_code_df by revenue_contribution_column in descending order
        item_code_df = item_code_df.sort_values(by=revenue_contribution_column, ascending=False).reset_index(drop=True)

        # Calculate cumulative revenue contribution percentage for the entire dataset
        total_revenue = item_code_df[revenue_contribution_column].sum()
        item_code_df['Cumulative %'] = item_code_df[revenue_contribution_column].cumsum() / total_revenue * 100

        # Now keep the top 20 items for the x-axis
        top_20_items = item_code_df.head(20).copy()

        # Create a combo chart
        fig, ax1 = plt.subplots(figsize=(10, 6))

        # Bar chart (Revenue Contribution) for top 20 items
        ax1.bar(top_20_items['Item Code'], top_20_items[revenue_contribution_column], color='skyblue')
        ax1.set_xlabel('Item Code')
        ax1.set_ylabel(revenue_contribution_column, color='blue')
        ax1.tick_params(axis='y', labelcolor='blue')

        # Create a second y-axis for the cumulative percentage line chart
        ax2 = ax1.twinx()
        
        # Plot cumulative percentage for top 20 items
        ax2.plot(top_20_items['Item Code'], top_20_items['Cumulative %'], color='red', marker='o')
        ax2.set_ylabel('Cumulative % of Revenue Contribution', color='red')
        ax2.tick_params(axis='y', labelcolor='red')

        # Set the limits for the right y-axis to always range from 0 to 100%
        ax2.set_ylim(0, 110)

        # Set the title
        plt.title('Revenue Contribution and Cumulative % by Top 20 Item Codes')

        # Display the plot
        plt.tight_layout()
        plt.show()
        
        return fig


    
    def create_upsell_configuration_dataframe(self, demand_df, conversion_rate):
        # Merge demand_df with upsell_recommendation_df to align treatment names
        merged_df = pd.merge(demand_df, self.upsell_recommendation_df, left_on='Treatment', right_on='Treatment', how='inner')
        
        merged_df = merged_df[['Treatment', 'Demand', 'Predicted Upsell Treatment']]
        merged_df.columns = ['Treatment', 'Demand', 'Treatment to Upsell']
        
        merged_df['Conversion Rate (%)'] = conversion_rate
        
        # Return the merged DataFrame with calculated columns and overall totals
        return merged_df
    
    def calculate_existing_and_upsell(self, demand_df, upsell_config_df):
        # Step 1: Run the original calculate_existing_condition method
        merged_df, total_revenue, total_cost, total_duration = self.calculate_existing_condition(demand_df)
        
        # Step 2: Initialize variables for tracking upsell metrics
        # additional_revenue = 0
        # additional_cost = 0
        # additional_duration = 0

        # Create a copy of merged_df to update
        updated_df = merged_df.copy()

        # Step 3: Iterate through upsell_config_df to calculate upsell metrics
        for index, row in upsell_config_df.iterrows():
            treatment = row['Treatment']
            upsell_treatment = row['Treatment to Upsell']
            conversion_rate = row['Conversion Rate (%)'] / 100
            demand = row['Demand']
            
            # Calculate the upsell demand
            upsell_demand = round(demand * conversion_rate)
            
            # If 'Treatment to Upsell' is NaN, skip to the next row (no upsell for that row)
            if pd.isna(upsell_treatment):
                continue
            
            # Step 4: Retrieve upsell treatment details from self.treatment_details_df
            upsell_details = self.treatment_details_df[self.treatment_details_df['Treatment'] == upsell_treatment]
            
            if upsell_details.empty:
                continue  # Skip if no details found for the upsell treatment
            
            upsell_price = upsell_details['total_price'].values[0]
            upsell_cost = upsell_details['total_cost'].values[0]
            upsell_duration = upsell_details['total_duration'].values[0]
            
            # Step 5: Calculate the additional revenue, cost, and duration from upselling
            # additional_revenue += upsell_price * upsell_demand
            # additional_cost += upsell_cost * upsell_demand
            # additional_duration += upsell_duration * upsell_demand

            # Step 6: Check if upsell treatment already exists in updated_df, sum the demand if so
            if upsell_treatment in updated_df['Treatment'].values:
                updated_df.loc[updated_df['Treatment'] == upsell_treatment, 'Demand'] += upsell_demand
            else:
                # Append a new row for the upsell treatment
                new_row = {
                    'Treatment': upsell_treatment,
                    'Demand': upsell_demand,
                    'Price per Treatment': upsell_price,
                    'Cost per Treatment': upsell_cost,
                    'Duration per Treatment': upsell_duration,
                    'Total Revenue': upsell_price * upsell_demand,
                    'Total Cost': upsell_cost * upsell_demand,
                    'Total Duration': upsell_duration * upsell_demand
                }
                updated_df = updated_df._append(new_row, ignore_index=True)

        # Step 7: Recalculate the totals for the updated DataFrame
        updated_df['Total Revenue'] = updated_df['Price per Treatment'] * updated_df['Demand']
        updated_df['Total Cost'] = updated_df['Cost per Treatment'] * updated_df['Demand']
        updated_df['Total Duration'] = updated_df['Duration per Treatment'] * updated_df['Demand']
        
        
        

        # Update the final totals
        final_total_revenue = updated_df['Total Revenue'].sum()
        final_total_cost = updated_df['Total Cost'].sum()
        final_total_duration = updated_df['Total Duration'].sum()
        
        # updated_df = updated_df[['Treatment', 'Demand']]

        # Step 8: Return final totals and the updated DataFrame
        return final_total_revenue, final_total_cost, final_total_duration, updated_df


import pandas as pd

class ModelItemCodeAnalysis:
    def __init__(self):
        self.item_code_df = None
        self.item_code_df = pd.read_csv('cleaned_item_code.csv', dtype={'item_number': str})
        


    def calculate_contributions(self, clinic_data_df, pricing_basis):
        """Calculate and return the extended DataFrame with additional columns."""
        if self.item_code_df is None:
            raise ValueError("Item code data not loaded. Please load data using load_data() method.")


        clinic_data_df = clinic_data_df.groupby('Code')['Demand'].sum().reset_index()

        # Ensure Code and item_number are strings for a proper merge
        clinic_data_df['Code'] = clinic_data_df['Code'].astype(str)
        self.item_code_df['item_number'] = self.item_code_df['item_number'].astype(str)

        # Determine which pricing columns to use based on the pricing_basis argument
        if pricing_basis == 'Indonesia':
            price_column = 'price IDR'
            cost_column = 'cost_material IDR'
        elif pricing_basis == 'Australia':
            price_column = 'price AUD'
            cost_column = 'cost_material AUD'
        else:
            raise ValueError("Invalid pricing_basis. Use 'Indonesia' or 'Australia'.")
        
        duration_column = 'duration'

        # Merge clinic_data with item_code based on the item_number and Code
        merged_df = pd.merge(clinic_data_df, self.item_code_df, how='left', left_on='Code', right_on='item_number')

        # Calculate Price per Item and Profit per Item
        merged_df['Price per Item'] = merged_df[price_column]
        merged_df['Profit per Item'] = merged_df[price_column] - merged_df[cost_column]
        merged_df['Duration per Item'] = merged_df[duration_column]

        # Fill missing values (NaN) with 0
        merged_df[['Duration per Item','Price per Item', 'Profit per Item']] =  merged_df[['Duration per Item','Price per Item', 'Profit per Item']].fillna(0)

        # Calculate Revenue Contribution and Profit Contribution
        merged_df['Revenue Contribution'] = merged_df['Price per Item'] * merged_df['Demand']
        merged_df['Profit Contribution'] = merged_df['Profit per Item'] * merged_df['Demand']
        merged_df['Total Duration'] = merged_df['Duration per Item'] * merged_df['Demand']

        # Select relevant columns for the final output
        final_columns = ['Code', 'Demand', 'Total Duration','Price per Item', 'Revenue Contribution', 'Profit per Item', 'Profit Contribution']
        final_df = merged_df[final_columns]

        return final_df
    
            
    def get_summary(self, clinic_data_df):
        total_demand = clinic_data_df['Demand'].sum()
        total_revenue = clinic_data_df['Revenue Contribution'].sum()
        total_profit = clinic_data_df['Profit Contribution'].sum()
        
        return total_demand, total_revenue, total_profit


    def calculate_contributions_by_provider(self, clinic_data_df, pricing_basis):
        """
        Calculate and return the extended DataFrame with additional columns, grouped by 'Provider'.
        """
        if self.item_code_df is None:
            raise ValueError("Item code data not loaded. Please load data using load_data() method.")

        # Ensure Code and item_number are strings for a proper merge
        clinic_data_df['Code'] = clinic_data_df['Code'].astype(str)
        self.item_code_df['item_number'] = self.item_code_df['item_number'].astype(str)

        # Determine which pricing columns to use based on the pricing_basis argument
        if pricing_basis == 'Indonesia':
            price_column = 'price IDR'
            cost_column = 'cost_material IDR'
        elif pricing_basis == 'Australia':
            price_column = 'price AUD'
            cost_column = 'cost_material AUD'
        else:
            raise ValueError("Invalid pricing_basis. Use 'Indonesia' or 'Australia'.")
        
        duration_column = 'duration'

        # Merge clinic_data with item_code based on the item_number and Code
        merged_df = pd.merge(clinic_data_df, self.item_code_df, how='left', left_on='Code', right_on='item_number')

        # Calculate Price per Item and Profit per Item
        merged_df['Price per Item'] = merged_df[price_column]
        merged_df['Profit per Item'] = merged_df[price_column] - merged_df[cost_column]
        merged_df['Duration per Item'] = merged_df[duration_column]

        # Fill missing values (NaN) with 0
        merged_df[['Duration per Item','Price per Item', 'Profit per Item']] =  merged_df[['Duration per Item','Price per Item', 'Profit per Item']].fillna(0)

        # Calculate Revenue Contribution and Profit Contribution
        merged_df['Revenue Contribution'] = merged_df['Price per Item'] * merged_df['Demand']
        merged_df['Profit Contribution'] = merged_df['Profit per Item'] * merged_df['Demand']
        merged_df['Total Duration'] = merged_df['Duration per Item'] * merged_df['Demand']

        # Group by Provider and sum the relevant columns
        final_df = merged_df.groupby('Provider').agg({
            'Demand': 'sum',
            'Total Duration': 'sum',
            'Revenue Contribution': 'sum',
            'Profit Contribution': 'sum'
        }).reset_index()

        return final_df
    # def create_pareto_chart(self, df, basis, top=20):
    #     """Create a Pareto chart based on the given basis (Demand, Revenue Contribution, or Profit Contribution)."""
    #     if basis not in ['Demand', 'Revenue Contribution', 'Profit Contribution']:
    #         raise ValueError("Invalid basis. Choose from 'Demand', 'Revenue Contribution', or 'Profit Contribution'.")

    #     # df = df.groupby('Code')['Demand'].sum().reset_index()
    #     # Sort df by the chosen basis in descending order
    #     df = df.sort_values(by=basis, ascending=False).reset_index(drop=True).head(top)

    #     # Calculate cumulative percentage for the chosen basis
    #     total_value = df[basis].sum()
    #     df['Cumulative %'] = df[basis].cumsum() / total_value * 100

    #     # Create a combo chart (Pareto chart)
    #     fig, ax1 = plt.subplots(figsize=(10, 6))

    #     # Bar chart for the chosen basis
    #     ax1.bar(df['Code'], df[basis], color='skyblue')
    #     ax1.set_xlabel('Code')
    #     ax1.set_ylabel(basis, color='blue')
    #     ax1.tick_params(axis='y', labelcolor='blue')

    #     # Create a second y-axis for the cumulative percentage line chart
    #     ax2 = ax1.twinx()
    #     ax2.plot(df['Code'], df['Cumulative %'], color='red', marker='o')
    #     ax2.set_ylabel('Cumulative % of ' + basis, color='red')
    #     ax2.tick_params(axis='y', labelcolor='red')

    #     # Set the limits for the right y-axis to always range from 0 to 100%
    #     ax2.set_ylim(0, 110)

    #     # Set the title
    #     plt.title(f'Pareto Chart for {basis}')

    #     # Display the plot
    #     plt.tight_layout()
   

    #     return fig
    



    def create_pareto_chart(self, df, basis, top=20):
        """Create a Pareto chart based on the given basis (Demand, Revenue Contribution, or Profit Contribution) using Plotly."""
        if basis not in ['Demand', 'Revenue Contribution', 'Profit Contribution']:
            raise ValueError("Invalid basis. Choose from 'Demand', 'Revenue Contribution', or 'Profit Contribution'.")

        # Sort df by the chosen basis in descending order and select top N
        df = df.sort_values(by=basis, ascending=False).reset_index(drop=True).head(top)

        # Calculate cumulative percentage for the chosen basis
        total_value = df[basis].sum()
        df['Cumulative %'] = df[basis].cumsum() / total_value * 100

        # Create a bar chart for the selected basis
        fig = go.Figure()

        # Add bar chart for basis (e.g., Demand or Revenue Contribution)
        fig.add_trace(
            go.Bar(
                x=df['Code'], 
                y=df[basis], 
                name=basis, 
                marker_color='skyblue', 
                yaxis='y1'
            )
        )

        # Add line chart for cumulative percentage
        fig.add_trace(
            go.Scatter(
                x=df['Code'], 
                y=df['Cumulative %'], 
                name='Cumulative %', 
                mode='lines+markers', 
                marker=dict(color='red'),
                yaxis='y2'
            )
        )

        # Set up the layout for the dual-axis chart
        fig.update_layout(
            title=f'Pareto Chart for {basis}',
            xaxis=dict(title='Code'),
            yaxis=dict(
                title=basis,
                titlefont=dict(color='blue'),
                tickfont=dict(color='blue'),
            ),
            yaxis2=dict(
                title='Cumulative %',
                overlaying='y',
                side='right',
                range=[0, 110],
                titlefont=dict(color='red'),
                tickfont=dict(color='red'),
            ),
            legend=dict(x=0.1, y=1.1),
            width=900,
            height=600,
            bargap=0.2,
        )

        # Show the chart
        return fig


    def create_horizontal_bar_chart(self, final_df, basis):
        """
        Create a horizontal bar chart based on the given basis (e.g., 'Demand', 'Revenue Contribution', etc.)
        and return the Plotly figure.
        
        Parameters:
        - final_df: DataFrame returned from calculate_contributions_by_provider.
        - basis: The column to base the bar chart on ('Demand', 'Revenue Contribution', 'Profit Contribution').
        
        Returns:
        - Plotly horizontal bar chart figure.
        """
        
        if basis not in final_df.columns:
            raise ValueError(f"Invalid basis: {basis}. Choose from {list(final_df.columns)}")

        # Sort the DataFrame by the selected basis to have an ordered bar chart
        final_df = final_df.sort_values(by=basis, ascending=True)

        # Generate pastel-like colors for each bar
        pastel_colors = colors.qualitative.Pastel

        # Create a horizontal bar chart
        fig = go.Figure(go.Bar(
            x=final_df[basis],  # X-axis is the selected basis column
            y=final_df['Provider'],  # Y-axis is the Provider column
            orientation='h',  # 'h' means horizontal bar chart
            marker=dict(color=pastel_colors[:len(final_df)])  # Assign pastel colors to each bar
        ))

        # Set up the layout
        fig.update_layout(
            title=f'Horizontal Bar Chart for {basis}',
            xaxis_title=basis,
            yaxis_title='Provider',
            yaxis=dict(tickmode='linear'),
            height=600,  # Adjust height as needed
            width=900,  # Adjust width as needed
            bargap=0.2,  # Gap between bars
        )

        return fig
    
    



    def create_bubble_scatter(self, df):
        """
        Creates a scatter plot with quadrants using the provided DataFrame with calculated columns:
        'Duration per Demand' and 'Revenue per Demand'. The plot will have quadrant lines based on the midpoints of the axes.
        
        Parameters:
        - df: DataFrame with columns 'Provider', 'Demand', 'Total Duration', 'Revenue Contribution', 'Profit Contribution'.
        
        Returns:
        - Plotly scatter plot figure with quadrants.
        """

        # Create new columns: Duration per Demand and Revenue per Demand
        df['Duration per Demand'] = df['Total Duration'] / df['Demand']
        df['Revenue per Demand'] = df['Revenue Contribution'] / df['Demand']

        # Scale Demand for better visualization in the scatter plot
        df['Scaled Demand'] = df['Demand'] / df['Demand'].max() * 100  # Scale between 0-100

        # Find the midpoint for x and y to draw the quadrant lines
        x_mid = 30
        y_mid = 50

        # Get the min and max values for x and y axes to set their respective ranges
        x_min, x_max = 0, 60
        y_min, y_max = 0, 100

        # Create the scatter plot using Plotly Express
        fig = px.scatter(
            df,
            x='Duration per Demand',
            y='Revenue per Demand',
            size='Scaled Demand',  # Scaled size for better visualization
            color='Provider',  # Color by Provider (if categorical)
            hover_name='Provider',  # Hover information
            size_max=60,  # Set a maximum bubble size
            title='Scatter Plot: Dentists Comparison of Duration per Demand vs. Revenue per Demand'
        )

        # Update layout to make the plot more square-like and draw quadrant lines
        fig.update_layout(
            width=700,  # Adjust figure size if needed
            height=700,  # Adjust figure size if needed
        #     xaxis=dict(range=[x_min, x_max], title="Duration per Demand"),
        #     yaxis=dict(range=[y_min, y_max], title="Revenue per Demand"),
        #     shapes=[
        #         # Horizontal line (y = y_mid)
        #         dict(
        #             type="line",
        #             x0=x_min, x1=x_max, y0=y_mid, y1=y_mid,
        #             line=dict(color="black", width=2, dash="dash")
        #         ),
        #         # Vertical line (x = x_mid)
        #         dict(
        #             type="line",
        #             x0=x_mid, x1=x_mid, y0=y_min, y1=y_max,
        #             line=dict(color="black", width=2, dash="dash")
        #         )
        #     ]
        )

        return fig