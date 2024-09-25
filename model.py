import pandas as pd
import matplotlib.pyplot as plt

class Model:
    def __init__(self):
        # Load treatment details, cleaned treatment CSV, and cleaned item code CSV into DataFrames
        self.treatment_details_df = pd.read_csv('treatment_with_details.csv')
        self.cleaned_treatment_df = pd.read_csv('cleaned_treatment.csv')
        self.cleaned_item_code_df = pd.read_csv('cleaned_item_code.csv')  # Load item code details
        self.upsell_recommendation_df = pd.read_csv('predicted_upsell_recommendation.csv')  # Load upsell recommendations
    
    def calculate_existing_condition(self, demand_df):
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

    def analyze_item_code_performance(self, demand_df):
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
        
        # Calculate the revenue contribution for each item code
        item_code_df['Revenue Contribution ($)'] = item_code_df['Total Count'] * item_code_df['price']
        
        # Drop the 'item_number' column after the merge (since it's redundant)
        item_code_df.drop(columns=['item_number'], inplace=True)
        
        item_code_df.columns = ['Item Code', 'Total Count', 'Price per Item', 'Revenue Contribution ($)']
        
        return item_code_df

    def create_combo_chart(self, item_code_df):
        # Sort item_code_df by 'Revenue Contribution ($)' in descending order
        item_code_df = item_code_df.sort_values(by='Revenue Contribution ($)', ascending=False).reset_index(drop=True)

        # Calculate cumulative revenue contribution percentage
        total_revenue = item_code_df['Revenue Contribution ($)'].sum()
        item_code_df['Cumulative %'] = item_code_df['Revenue Contribution ($)'].cumsum() / total_revenue * 100
        
        # Create a combo chart
        fig, ax1 = plt.subplots(figsize=(10, 6))

        # Bar chart (Revenue Contribution)
        ax1.bar(item_code_df['Item Code'], item_code_df['Revenue Contribution ($)'], color='skyblue')
        ax1.set_xlabel('Item Code')
        ax1.set_ylabel('Revenue Contribution ($)', color='blue')
        ax1.tick_params(axis='y', labelcolor='blue')

        # Create a second y-axis for the cumulative percentage line chart
        ax2 = ax1.twinx()
        ax2.plot(item_code_df['Item Code'], item_code_df['Cumulative %'], color='red', marker='o')
        ax2.set_ylabel('Cumulative % of Revenue Contribution', color='red')
        ax2.tick_params(axis='y', labelcolor='red')
        
        # Set the limits for the right y-axis to always range from 0 to 100%
        ax2.set_ylim(0, 110)

        # Set the title
        plt.title('Revenue Contribution and Cumulative % by Item Code')

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
        merged_df, total_revenue, total_cost, total_duration = self.calculate_existing_and_upsell(demand_df)
        
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
                updated_df = updated_df.append(new_row, ignore_index=True)

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

