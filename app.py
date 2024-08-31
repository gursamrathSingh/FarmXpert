import streamlit as st
import pandas as pd
import numpy as np
import os
import pickle
import warnings

st.set_page_config(page_title="FarmXpert", page_icon="🌱", layout='centered', initial_sidebar_state="collapsed")

def load_model(model_file):
    model = pickle.load(open(model_file, 'rb'))
    return model

def main():
    # title
    html_temp = """
        <div>
        <h1 style="color:MEDIUMSEAGREEN;text-align:left;"> Farm Expert  🌱 </h1>
        </div>
        """
    st.markdown(html_temp, unsafe_allow_html=True)

    col1, spacer, col2 = st.columns([4, 0.5, 3.25])

    with col1:
        with st.expander("Information", expanded=True):
            st.write("""
            Choosing the right crops for your land can be a complex decision influenced by various factors such as soil quality, climate conditions, and seasonal variations. This project simplifies the process by leveraging machine learning algorithms to provide tailored crop recommendations based on your unique environmental and soil conditions.
                   Crop recommendations are based on a number of factors. While the "site-specific" methodology has improved performance, there is still a need to monitor the systems' outcomes. Precision agriculture systems aren't all created equal. 
                   However, in agriculture, it is critical that the recommendations made are correct and precise, as errors can result in significant material and capital loss.

                   """)
        '''
        ## How does it work 
        Complete all the parameters and the machine learning model will predict the most suitable crops to grow in a particular farm based on various parameters
        '''

    # To increase space between columns
    with spacer:
        st.write("")


    with col2:
        st.subheader(" Find out the most suitable crop to grow in your farm!")
        N = st.number_input("Nitrogen", 1, 10000)
        P = st.number_input("Phosporus", 1, 10000)
        K = st.number_input("Potassium", 1, 10000)
        temp = st.number_input("Temperature", 0.0, 100000.0)
        humidity = st.number_input("Humidity in %", 0.0, 100000.0)
        ph = st.number_input("Ph", 0.0, 100000.0)
        rainfall = st.number_input("Rainfall in mm", 0.0, 100000.0)

        feature_list = [N, P, K, temp, humidity, ph, rainfall]
        single_pred = np.array(feature_list).reshape(1, -1)

        if st.button('Predict'):

            model = load_model('model.pkl')
            prediction = model.predict(single_pred)
            col1.write('''
            		    ## Results  
            		    ''')
            col1.success(f" The Machine learning model has predicted the following crop based on your farm parameters- {prediction.item().title()}")



hide_menu_style = """
        <style>
        #MainMenu {visibility: hidden;}
        </style>
        """
st.markdown(hide_menu_style, unsafe_allow_html=True)

if __name__ == '__main__':
	main()