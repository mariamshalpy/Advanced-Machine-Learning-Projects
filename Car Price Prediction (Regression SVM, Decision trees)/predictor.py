import joblib
import numpy as np

DT_model_path = './DT_model.pkl'
SVR_model_path = './linear_svr_model.pkl'
x_columns_path = './x_columns.pkl'
scaler_path = './scaler.pkl'

# Load the models
model_DT = joblib.load(DT_model_path)
model_linear_svr = joblib.load(SVR_model_path)
x_columns = joblib.load(x_columns_path)
scaler = joblib.load(scaler_path)


def predict_car_price(mark, model, fuel, year, mileage, vol_engine, x_columns, scaler, model3):
    try:
        brand_index = np.where(x_columns == mark)[0][0]
        model_index = np.where(x_columns == model)[0][0]
        fuel_index = np.where(x_columns == fuel)[0][0]
    except IndexError:
        print("One or more categorical inputs do not match the feature names.")
        return None

    features = np.zeros(len(x_columns))
    features[0] = year
    features[1] = mileage
    features[2] = vol_engine

    if brand_index >= 0:
        features[brand_index] = 1
    if model_index >= 0:
        features[model_index] = 1
    if fuel_index >= 0:
        features[fuel_index] = 1

    features_scaled = scaler.transform([features])
    return model3.predict(features_scaled)[0]





def get_car_model():
    car_model = ['combo', 'vectra', 'adam', 'agila', 'ampera', 'antara', 'astra',
       'corsa', 'crossland-x', 'frontera', 'grandland-x', 'insignia',
       'vivaro', 'zafira', 'a3', 'karl', 'meriva', 'mokka', 'omega',
       'signum', 'tigra', '80', 'a1', 'a2', 'a4', 'a4-allroad', 'a5',
       'a6', 'a6-allroad', 'a7', 'a8', 'e-tron', 'q2', 'q3',
       'q4-sportback', 'q5', 'q7', 'q8', 'rs3', 'rs5', 'rs6', 'rs-q3',
       's3', 's5', 's8', 'sq5', 'tt', '3gt', '5gt', 'i3', 'm2', 'm3',
       'm4', 'm5', 'm8', 'seria-1', 'seria-2', 'seria-3', 'seria-4',
       'seria-5', 'seria-6', 'seria-7', 'seria-8', 'x1', 'x2', 'x3', 'x4',
       'x5', 'x5-m', 'x6', 'x6-m', 'x7', 'amarok', 'arteon', 'beetle',
       'caddy', 'california', 'caravelle', 'cc', 'crafter', 'eos', 'fox',
       'golf', 'golf-plus', 'golf-sportsvan', 'id4', 'jetta', 'lupo',
       'multivan', 'new-beetle', 'passat', 'passat-cc', 'phaeton', 'polo',
       'scirocco', 'sharan', 't-cross', 't-roc', 'tiguan',
       'tiguan-allspace', 'touareg', 'touran', 'transporter', 'up',
       'b-max', 'c-max', 'ecosport', 'edge', 'escape', 'explorer', 'f150',
       'fiesta', 'focus', 'focus-c-max', 'fusion', 'galaxy',
       'grand-c-max', 'ka', 'kuga', 'mondeo', 'mustang', 'mustang-mach-e',
       'puma', 'ranger', 's-max', 'tourneo-connect', 'tourneo-courier',
       'tourneo-custom', 'transit', 'transit-connect', 'transit-custom',
       'amg-gt', 'citan', 'cl-klasa', 'cla-klasa', 'clk-klasa',
       'cls-klasa', 'gl-klasa', 'gla-klasa', 'glb-klasa', 'glc-klasa',
       'gle-klasa', 'glk-klasa', 'gls-klasa', 'a-klasa', 'b-klasa',
       'c-klasa', 'e-klasa', 'g-klasa', 'r-klasa', 's-klasa', 'v-klasa',
       'm-klasa', 'sl', 'slk-klasa', 'sprinter', 'viano', 'vito',
       'arkana', 'captur', 'clio', 'espace', 'grand-espace',
       'grand-scenic', 'fluence', 'kadjar', 'kangoo', 'koleos', 'laguna',
       'megane', 'modus', 'scenic', 'talisman', 'thalia', 'trafic',
       'twingo', 'zoe', 'auris', 'avensis', 'aygo', 'c-hr', 'camry',
       'corolla', 'corolla-verso', 'land-cruiser', 'prius',
       'proace-verso', 'rav4', 'sienna', 'yaris', 'verso', 'citigo',
       'enyaq', 'fabia', 'kamiq', 'karoq', 'kodiaq', 'octavia', 'rapid',
       'roomster', 'scala', 'superb', 'yeti', '147', '159', 'giulia',
       'giulietta', 'mito', 'aveo', 'camaro', 'cruze', 'orlando',
       'berlingo', 'c3-aircross', 'c3-picasso', 'c4-cactus',
       'c4-grand-picasso', 'c4-picasso', 'c5', 'c5-aircross', 'ds3',
       'ds4', 'ds5', 'xsara-picasso', '500', '500l', '500x', 'bravo',
       'doblo', 'freemont', 'grande-punto', 'panda', 'punto', 'punto-evo',
       'tipo', 'accord', 'cr-v', 'hr-v', 'jazz', 'civic', 'elantra',
       'i10', 'i20', 'i30', 'i40', 'ix20', 'ix35', 'kona', 'santa-fe',
       'tucson', 'carens', 'ceed', 'optima', 'picanto', 'pro-ceed',
       'sorento', 'soul', 'sportage', 'stinger', 'stonic', 'venga',
       'xceed', '2', '3', '5', '6', 'cx-3', 'cx-5', 'cx-7', 'cx-9',
       'cx-30', 'mx-5', 'clubman', 'cooper', 'cooper-s', 'countryman',
       'one', 'asx', 'colt', 'eclipse-cross', 'lancer', 'outlander',
       'space-star', 'almera', 'juke', 'leaf', 'micra', 'murano', 'note',
       'patrol', 'primera', 'qashqai', 'qashqai-2', 'x-trail', '206',
       '207', '208', '307', '308', '407', '508', '2008', '3008', '5008',
       'expert', 'partner', 'alhambra', 'altea', 'altea-xl', 'arona',
       'ateca', 'exeo', 'ibiza', 'leon', 'toledo', 'c30', 's40', 's60',
       's80', 'v40', 'v50', 'v60', 'v70', 'v90', 'xc-40', 'xc-60',
       'xc-70', 'xc-90']
    return car_model

def get_car_mark():
    car_mark = ['opel', 'audi', 'bmw', 'volkswagen', 'ford', 'mercedes-benz',
       'renault', 'toyota', 'skoda', 'alfa-romeo', 'chevrolet', 'citroen',
       'fiat', 'honda', 'hyundai', 'kia', 'mazda', 'mini', 'mitsubishi',
       'nissan', 'peugeot', 'seat', 'volvo']
    return car_mark

def get_fuel_type():   
    car_fuel_type = ['Diesel', 'CNG', 'Gasoline', 'LPG', 'Hybrid', 'Electric']
    return car_fuel_type








