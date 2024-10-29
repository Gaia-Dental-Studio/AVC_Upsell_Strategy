import streamlit as st
import pandas as pd
import main_app
import item_code_analysis

# Create sidebar navigation with four options
selection = st.sidebar.radio(
    "Select Page",
    ["Treatment Menu Upselling", "Item Code Analysis"],
    index=1
    )

if selection == "Treatment Menu Upselling":
    main_app.app()

elif selection == "Item Code Analysis":
    item_code_analysis.app()