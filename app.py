# %matplotlib inline 
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import sys
import sklearn
print("Python version {}".format(sys.version))
# print("Scikit-learn version {}".format(sklearn))

insurance = pd.read_csv(r"C:\Users\eklas\OneDrive\Desktop\CSV\insurance_claims.csv")
pd.set_option('display.max_rows', 11)


insurance.fraud_reported.replace(('Y', 'N'), (True, False), inplace=True)
insurance.police_report_available.replace(('YES', 'NO'), (1, 0), inplace=True)

# Sex is antiquated, change that (to Gender) 
insurance=insurance.rename(columns={'insured_sex':'insured_gender'})

# There are several columns that are nonsense and/or have a high number of distinct values. Drop those suckers 

insurance.drop(columns=['_c39', 'policy_number', 'policy_bind_date', 'incident_location', 'incident_date', 
                       'insured_zip', 'policy_csl', 'insured_occupation', 'auto_model'], inplace=True)


# before Machine LEarning, I'll need to convert non-numerical to numerical.

for col_name in insurance.columns:
    if(insurance[col_name].dtype == 'object'):
        insurance[col_name] = insurance[col_name].astype('category')
        insurance[col_name] = insurance[col_name].cat.codes

        
dataset = insurance


from pycaret.classification import *
# import the classification module 
from pycaret import classification

# split data for model and prediction

data = insurance.sample(frac = 0.7, random_state = 123)
data_unseen = dataset.drop(data.index).reset_index(drop = True)
data.reset_index(drop=True, inplace=True)

print('Data for Modelling: ' + str(data.shape))
print('Unseen Data for Predictions: ' + str(data_unseen.shape))


# Set up PyCaret for Modelling, add in paramaters 

exp_model = setup(data = data, target = 'fraud_reported', session_id = 123,
                  remove_multicollinearity = True,
                  bin_numeric_features = ['age'],
                  ignore_low_variance = True,
                  fix_imbalance = True,
                  normalize = True,
                  transformation = True,
                  feature_selection = True,
                  multicollinearity_threshold = 0.95, html = False)


gbc = create_model('gbc')


# saving model
save_model(gbc, model_name = 'gbc_deploy_test')

######################


# build the interface

from pycaret.classification import load_model, predict_model
import streamlit as st
import pandas as pd
import numpy as np

# load trained model for predictions
model = load_model('gbc_deploy_test')

# main function starts here

def predict(model, input_df):
    predictions_df = predict_model(estimator = model, data = input_df)
    predictions = predictions_df['Label'][0]
    return predictions


def run():
    from PIL import image
    image_fraud = Image.open('vehicle_fraud.jpg')
    
    st.image(image_fraud, use_column_width = False)
    
    add_selectbox = st.sidebar.selectbox(
    "How would you like to predict potential fraud? "
    ('Online', 'Batch'))
    
    st.sidebar.info('This app detects potentially fraudulent vehicle insurance claims. ')
    st.sidebar.image(image_fraud)
    
    st.title('Vehicle Insurance Claims: Fraud Detection App')



# load trained model for predictions
# model = load_model('gbc_deploy_test')




# for online predictions

    if add_selectbox == 'Online':


        age = st.number_input('age', min_value = 1, max_value = 100, value = 30)
        gender = st.selectbox('insured_gender', ['MALE', 'FEMALE'])
        incident_severity = st.selectbox('Incident Severity',[1, 2, 3] )
        num_veh_involved = st.number_input('Number of vehicles involved', min_value = 1, max_value = 5, value = 1)
        hobby = st.selectbox('insured_hobbies', [
                'cross-fit','chess',
                  'sleeping','reading','yachting',
                  'bungie-jumping', 'base-jumping', 
                  'board-games', 'camping', 'dancing',
                  'movies', 'skydiving', 'polo',
                  'basketball', 'video-games', 'golf', 
                  'exercise'])
        state = st.selectbox('State', ['IL', 'OH', 'MN'])
        #     if st.checkbox('High Damage'):
        #         total_claim_amount > 28000 = 'yes'
        #             else:
        #         high_damage = 'no'
        output=""


        input_dict = {'age': age, 'gender': gender, 
                          'incident severity': incident_severity, 
                          'number of vehicles involved': num_veh_involved, 
                          'hobby': hobby,
                         'state': state, 
                         'high damage': high_damage}
        input_df = pd.DataFrame([input_dict])

        if st.button("Predict Potential Fraud"):
            output = predict(model = model, input_df = input_df)
            output = str(output)

        st.success('Potentially fraudulent claim? : {}'.format(output))
