import joblib
import streamlit as st
from predictor import get_car_mark, get_car_model, get_fuel_type, predict_car_price

car_fuel_type = get_fuel_type()
car_model = get_car_model()
car_mark = get_car_mark()

DT_model_path = './DT_model.pkl'
SVR_model_path = './linear_svr_model.pkl'
GB_model_path = './GB.pkl'
x_columns_path = './x_columns.pkl'
scaler_path = './scaler.pkl'

# Load the models
model_DT = joblib.load(DT_model_path)
model_linear_svr = joblib.load(SVR_model_path)
model_GB = joblib.load(GB_model_path)
x_columns = joblib.load(x_columns_path)
scaler = joblib.load(scaler_path)

def main():
    st.title('Car Price Prediction')

    year = st.slider('Year', min_value=1945, max_value=2024, step=1)

    mileage = st.number_input('Mileage (in km)')

    vol_engine = st.number_input('Engine Volume (in cc)')

    mark = st.selectbox('Car Make', car_mark)

    fuel = st.selectbox('Fuel Type', car_fuel_type)

    model = st.selectbox('Car Model', car_model)


    if st.button('Predict'):
        
        GB_prediction= predict_car_price(mark, model, fuel, year, mileage, vol_engine, x_columns, scaler, model_GB)
        DT_prediction= predict_car_price(mark, model, fuel, year, mileage, vol_engine, x_columns, scaler, model_DT)
        linear_svr = predict_car_price(mark, model, fuel, year, mileage, vol_engine, x_columns, scaler, model_linear_svr)

        GB_prediction_formatted = "{:.2f}".format(GB_prediction)
        DT_prediction_formatted = "{:.2f}".format(DT_prediction)
        linear_svr_formatted = "{:.2f}".format(linear_svr)

        st.write('GB Predicted Price:', GB_prediction_formatted)
        st.write('DT Predicted Price:', DT_prediction_formatted)
        st.write("Linear_SVR Predicted Price:", linear_svr_formatted)


if __name__ == "__main__":
    main()