
# conda activate myenv
# cd Documents
# streamlit run fraud_app.py

# %matplotlib inline 
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import sys
import sklearn
print("Python version {}".format(sys.version))
print("Scikit-learn version {}".format(sklearn))

insurance = pd.read_csv(r"C:\Users\eklas\OneDrive\Desktop\CSV\insurance_claims_em.csv")
pd.set_option('display.max_rows', 11)

# Sex is antiquated, change that (to Gender) 
insurance=insurance.rename(columns={'insured_sex':'gender'})

insurance.fraud_reported.replace(('Y', 'N'), (True, False), inplace=True)
insurance.police_report_available.replace(('YES', 'NO'), ('Yes', 'No'), inplace=True)
insurance.gender.replace(('MALE', 'FEMALE'), ('Male', 'Female'), inplace = True)

# make int
insurance['bodily_injuries'] = insurance['bodily_injuries'].astype(int)


# There are several columns that are nonsense and/or have a high number of distinct values. Drop those suckers 

insurance.drop(columns=['_c39', 'policy_number', 'policy_bind_date', 'incident_location', 'incident_date', 
                       'insured_zip', 'policy_csl', 'insured_occupation', 'auto_model', 'umbrella_limit',
                       'policy_deductable', 'witnesses',
 'authorities_contacted', 'incident_hour_of_the_day',    
'bodily_injuries',
'policy_state', 
'incident_city',
'capital-gains',
'capital-loss',
'property_damage'
], inplace=True)

        
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
                  bin_numeric_features = ['age', 'months_as_customer'],
                  ignore_low_variance = True,
#                  fix_imbalance = True,
                  normalize = True,
                  transformation = True,
                  feature_selection = True,
                  multicollinearity_threshold = 0.95, html = False)


gbc = create_model('gbc')

# saving model
save_model(gbc, model_name = 'gbc_deploy_test')

###############################################

###############################################

# build the interface



from pycaret.classification import load_model, predict_model
import streamlit as st
import pandas as pd
import numpy as np


# load trained model for predictions
model = load_model('gbc_deploy_test')
st.set_option('deprecation.showfileUploaderEncoding', False)

# main function starts here

def predict(model, input_df):
    predictions_df = predict_model(estimator = model, data = input_df)
    predictions = predictions_df['Label'][0]
    score = round(predictions_df['Score'][0]*100)
    if score < 40:
        st.write("\n*This claim has NOT been flagged for fraud.*")
    elif score > 39 and score < 49:
        st.write("\n*This claim has NOT been flagged, however it is BORDERLINE.*")
    elif score > 50 and score < 60:
        st.write("\n*This claim has been FLAGGED as BORDERLINE fraudulent.*")
    elif score > 59 and score < 79:
        st.write("\n*This claim has been FLAGGED as POTENTIALLY FRAUDULENT, so may be worth further investigation.*")
    elif score > 79:
        st.write("\n*This claim has been FLAGGED as LIKELY FRAUDULENT, so is probably worth further investigation.*")
    score_str = str(score) + '%'
    return predictions, score_str

def run():
    from PIL import Image
    image_fraud = Image.open('image_fraud.jpg')

    add_selectbox = st.sidebar.selectbox(
    'How would you like to input your data for analysis?',
    ('Manual Input', 'Upload CSV'))
    
    st.sidebar.info('This app predicts potentially fraudulent vehicular insurance claims. \n\n It cannot replace human judgement; rather, acts to help you sift through claims and flag those are worth further investigation.\n')

    st.sidebar.image(image_fraud)
    
    st.title('Fraud Flagger:')
    st.subheader('\nVehicle Insurance Claims\n')

#
#    st.write('An Em + Dan Jam')


# for online predictions

    if add_selectbox == 'Manual Input':


        age = st.number_input('Age', min_value = 1, max_value = 100, value = 30) #1
        gender = st.selectbox('Gender', ['Male', 'Female']) #2
        insured_hobbies = st.selectbox('Hobbies', [ 
                'cross-fit','chess',
                  'sleeping','reading','yachting',
                  'bungie-jumping', 'base-jumping', 
                  'board-games', 'camping', 'dancing',
                  'movies', 'skydiving', 'polo',
                  'basketball', 'video-games', 'golf', 
                  'exercise'])
        insured_education_level = st.selectbox('Education level', [ #4
		'High School', 'Associate','College', 'Masters', 'JD', 'PhD', 'MD' ])
        insured_relationship = st.selectbox('Relationship', [ #5
            'husband', 'other-relative', 'own-child', 'unmarried', 'wife','not-in-family'        ])
        insured_occupation = st.selectbox('Occupation category', [ #6
            'craft-repair', 'machine-op-inspct', 'sales', 'armed-forces',
       'tech-support', 'prof-specialty', 'other-service',
       'priv-house-serv', 'exec-managerial', 'protective-serv',
       'transport-moving', 'handlers-cleaners', 'adm-clerical',
       'farming-fishing'     ])
        
        policy_annual_premium = st.number_input('Annual premium ($USD)', min_value = 300, max_value = 100000, value = 1000)       
        auto_make = st.selectbox('Auto Make', ['Saab', 'Mercedes', 'Dodge', #8
                                          'Chevrolet', 'Accura', 'Nissan', 
                                          'Audi', 'Toyota', 'Ford', 
                                          'Suburu', 'BMW', 'Jeep',
                                          'Honda','Volkswagen'])
        auto_year = st.number_input('Auto year', min_value = 1995, max_value = 2015, value = 2015) #9

        incident_type = st.selectbox('Incident type', ['Single Vehicle Collision', 'Vehicle Theft', #10
       'Multi-vehicle Collision', 'Parked Car'])
        collision_type = st.selectbox('Collision type', ['Side Collision', '?', 'Rear Collision', 'Front Collision']) #11
        incident_severity = st.selectbox('Incident severity', ['Trivial Damage', 'Minor Damage', 'Major Damage', 'Total Loss']) #12

        vehicle_claim = st.number_input('Vehicular damage claim reported (USD $)', min_value = 1, max_value = 2000000) #13
        property_claim = st.number_input('Property claim reported (USD $)', min_value = 1, max_value = 2000000) #14
        injury_claim = st.number_input('Injury claim reported (USD $)', min_value = 1, max_value = 2000000) #15
        total_claim_amount = st.number_input('Total damage reported (USD $)', min_value = 1, max_value = 2000000) #16
#        bodily_injuries = st.selectbox('Bodily injuries', [0, 1, 2, 3, 4]) #17
        number_of_vehicles_involved = st.number_input('Number of vehicles involved', min_value = 1, max_value = 5, value = 1) #18
        police_report_available = st.selectbox('Police report available', ['Yes', 'No', '?']) #19
        authorities_contacted = st.selectbox('Authorities contacted', ['Police', 'None', 'Fire', 'Other', 'Ambulance']) #20
        
        months_as_customer = st.number_input('Months as customer', min_value = 1, max_value = 5000, value = 12) #21
        incident_state = st.selectbox('State where incident occurred', ['IL', 'OH', 'MN']) #22
        

        output=""


        input_dict = {'age': age, 
                      'gender': gender,
                      'insured_hobbies': insured_hobbies,
                      'insured_education_level': insured_education_level,
                      'insured_relationship': insured_relationship,
                      'insured_occupation': insured_occupation,
                      'policy_annual_premium': policy_annual_premium,
                      'police_report_available': police_report_available,
                      'authorities_contacted': authorities_contacted,
                      'incident_type': incident_type,
                      'incident_severity': incident_severity, 
                      'collision_type': collision_type,
                      'vehicle_claim': vehicle_claim,
                      'property_claim': property_claim,
                      'injury_claim': injury_claim,
                      'total_claim_amount': total_claim_amount,
                      'number_of_vehicles_involved': number_of_vehicles_involved, 
                      'incident_state': incident_state, 
#                      'bodily_injuries': bodily_injuries,
                      'months_as_customer': months_as_customer,
                      'auto_make': auto_make,
                      'auto_year': auto_year

                                           }

        input_df = pd.DataFrame([input_dict])




        if st.button("\nPredict\n"):
            output = predict(model = model, input_df = input_df)

            output = str(output)

    

        st.success('The likelihood of fraud wtih this claim is: {}'.format(output))

        from PIL import Image

        fraud_pred_1 = Image.open('fraud_pred_1.jpg')
        fraud_pred_2 = Image.open('fraud_pred_2.jpg')


        st.subheader('What Influences Fraud to be True?\n\n\n')


        st.write(' ')   

        st.image(fraud_pred_1)
        st.image(fraud_pred_2)


############################ Batch with CSV upload



    if add_selectbox == 'Upload CSV':

    
        file_upload = st.file_uploader('Upload your CSV for predictions here ', type = ['csv'])
    
        
        if file_upload is not None:
            data = pd.read_csv(file_upload)
            predictions = predict_model(estimator = model, data = data)



	
            st.write(predictions)
            st.write('\nCheck the *Label* column to the right to see whether each claim is predicted fraudulent or not, where 0 = Not Fraud, 1 = Fraud.\n\nThe predicted % likelihood of fraud is in the *Score* column.')

if __name__ == '__main__':
   run()

