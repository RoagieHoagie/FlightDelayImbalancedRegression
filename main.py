import gradio as gr
import pandas as pd
import numpy as np
import pickle
import datetime
import json
import os
from pandas.tseries.holiday import USFederalHolidayCalendar

# ==========================================
# 1. INICJALIZACJA ZASOBÓW ŚRODOWISKOWYCH
# ==========================================

cal = USFederalHolidayCalendar()
holidays = cal.holidays(start='2018-01-01', end='2030-12-31')

TRAINED_DIR = "trained"
try:
    AVAILABLE_MODEL_FILES = [f for f in os.listdir(TRAINED_DIR) if f.endswith('.pkl')]
    AVAILABLE_MODEL_FILES.sort() 
except FileNotFoundError:
    print(f"Ostrzeżenie: Katalog '{TRAINED_DIR}' nie istnieje.")
    AVAILABLE_MODEL_FILES = []

try:
    with open("route_topology.json", "r") as f:
        topology = json.load(f)
    CITIES = topology.get('cities', [])
    CITY_TO_STATE = topology.get('city_to_state', {})
    ROUTES = topology.get('routes', {})
except FileNotFoundError:
    print("Ostrzeżenie: Brak pliku route_topology.json. Asynchroniczne mapowanie tras nieaktywne.")
    CITIES, CITY_TO_STATE, ROUTES = [], {}, {}


STATES = sorted(list(set(CITY_TO_STATE.values()))) if CITY_TO_STATE else []
CARRIERS = sorted([
    "United Airlines", "Delta Airlines", "Frontier Airlines", "Spirit Airlines", 
    "American Airlines", "Southwest Airlines", "Alaska Airlines", "Hawaiian Airlines", 
    "Virgin America", "JetBlue Airways", "Allegiant Air"
])

# ==========================================
# 2. ŁADOWANIE MODELU (LAZY LOADING)
# ==========================================

def load_selected_model(filename):
    if not filename:
        return None, gr.update(choices=[], value=None, interactive=False), "Błąd 400: Brak pliku."
        
    filepath = os.path.join(TRAINED_DIR, filename)
    try:
        with open(filepath, 'rb') as f:
            model_dict = pickle.load(f)
            
        strategies = list(model_dict.keys())
        default_strategy = "WERCS" if "WERCS" in strategies else (strategies[0] if strategies else None)
        status_msg = f"Status 200: Załadowano {filepath} (Liczba strategii: {len(strategies)})"
        
        return model_dict, gr.update(choices=strategies, value=default_strategy, interactive=True), status_msg
        
    except FileNotFoundError:
        return None, gr.update(choices=[], value=None, interactive=False), f"Błąd 404: Brak pliku {filepath}"

# ==========================================
# 3. WARSTWA PRZETWARZANIA WSTĘPNEGO
# ==========================================

def update_route_parameters(orig_city, dest_city):
    orig_state = CITY_TO_STATE.get(orig_city, "")
    dest_state = CITY_TO_STATE.get(dest_city, "")
    
    new_distance, new_air_time, new_tz_shift = None, None, None
    
    if orig_city in ROUTES and dest_city in ROUTES[orig_city]:
        route_params = ROUTES[orig_city][dest_city]
        new_distance = route_params['distance']
        new_air_time = route_params['air_time']
        new_tz_shift = route_params['tz_shift']
        
    return (
        gr.update(value=orig_state),
        gr.update(value=dest_state),
        gr.update(value=new_distance) if new_distance is not None else gr.update(),
        gr.update(value=new_air_time) if new_air_time is not None else gr.update(),
        gr.update(value=new_tz_shift) if new_tz_shift is not None else gr.update()
    )

def update_arrival_time(dep_time_str, air_time, tz_shift):
    try:
        dep_time = datetime.datetime.strptime(dep_time_str, "%H:%M")
        air_t = float(air_time) if air_time is not None else 0.0
        tz_s = float(tz_shift) if tz_shift is not None else 0.0
        
        total_shift_minutes = air_t + tz_s
        arr_time = dep_time + datetime.timedelta(minutes=total_shift_minutes)
        return arr_time.strftime("%H:%M")
    except (ValueError, TypeError):
        return gr.update()

def extract_features(date_str, carrier, origin_city, origin_state, dest_city, 
                     dest_state, distance, sched_air_time, tz_shift, 
                     dep_time_str, arr_time_str):
    
    flight_date = datetime.datetime.strptime(date_str, "%Y-%m-%d")
    dep_time = datetime.datetime.strptime(dep_time_str, "%H:%M").time()
    arr_time = datetime.datetime.strptime(arr_time_str, "%H:%M").time()
    
    dep_minutes = dep_time.hour * 60 + dep_time.minute
    arr_minutes = arr_time.hour * 60 + arr_time.minute
    
    data = {
        'year': flight_date.year,
        'month': flight_date.month,
        'dayOfMonth': flight_date.day,
        'scheduledAirTime': sched_air_time,
        'distance': distance,
        'dayOfWeek': flight_date.weekday() + 1,
        'isHoliday': flight_date in holidays,
        'scheduledDeparture_sin': np.sin(2 * np.pi * dep_minutes / 1440.0),
        'scheduledDeparture_cos': np.cos(2 * np.pi * dep_minutes / 1440.0),
        'scheduledArrival_sin': np.sin(2 * np.pi * arr_minutes / 1440.0),
        'scheduledArrival_cos': np.cos(2 * np.pi * arr_minutes / 1440.0),
        'timeZoneShift': tz_shift,
        'originCity': origin_city,
        'originState': origin_state,
        'destCity': dest_city,
        'destState': dest_state,
        'carrierName': carrier
    }
    
    features = pd.DataFrame([data])
    
    # Rzutowanie zmiennych numerycznych
    dtypes_numeric = {
        'year': 'int16', 'month': 'int8', 'dayOfMonth': 'int8',
        'scheduledAirTime': 'float32', 'distance': 'float32',
        'dayOfWeek': 'int32', 'isHoliday': 'bool',
        'scheduledDeparture_sin': 'float64', 'scheduledDeparture_cos': 'float64',
        'scheduledArrival_sin': 'float64', 'scheduledArrival_cos': 'float64',
        'timeZoneShift': 'float32'
    }
    features = features.astype(dtypes_numeric)
    
    city_dtype = pd.CategoricalDtype(categories=CITIES, ordered=False)
    state_dtype = pd.CategoricalDtype(categories=STATES, ordered=False)
    carrier_dtype = pd.CategoricalDtype(categories=CARRIERS, ordered=False)
    
    features['originCity'] = features['originCity'].astype(city_dtype)
    features['destCity'] = features['destCity'].astype(city_dtype)
    features['originState'] = features['originState'].astype(state_dtype)
    features['destState'] = features['destState'].astype(state_dtype)
    features['carrierName'] = features['carrierName'].astype(carrier_dtype)
    
    return features

# ==========================================
# 4. WARSTWA INFERENCYJNA
# ==========================================

def classify_risk(delay_minutes):
    if delay_minutes <= 15.0:
        return "Nominalny (≤15 min)"
    elif delay_minutes <= 60.0:
        return "Umiarkowany (15-60 min)"
    else:
        return "Krytyczny (>60 min)"



def predict_delay(loaded_model_dict, selected_strategy, *args):
    if not loaded_model_dict:
        return "Błąd: Brak modelu w pamięci. Wymagana inicjalizacja operacji I/O."
        
    if selected_strategy not in loaded_model_dict:
        return "Błąd: Strategia preprocessingu poza słownikiem."
        
    X_input = extract_features(*args)
    estimator = loaded_model_dict[selected_strategy]
    
    cat_cols = ['originCity', 'originState', 'destCity', 'destState', 'carrierName']
    X_input = X_input.copy()
    
    if type(estimator).__name__ == "LGBMRegressor":
        # 1. Ekstrakcja kategorii bezpośrednio z pamięci wewnętrznej modelu LightGBM
        if hasattr(estimator, 'booster_') and hasattr(estimator.booster_, 'pandas_categorical'):
            lgbm_cats = estimator.booster_.pandas_categorical
            
            # Zabezpieczenie kompatybilności dla różnych wersji LightGBM
            feature_names = estimator.feature_name_ if hasattr(estimator, 'feature_name_') else estimator.booster_.feature_name()
            cat_feature_names = [f for f in feature_names if f in cat_cols]
            
            # Wymuszenie identycznego mapowania ciągów znaków (String)
            if isinstance(lgbm_cats, list) and len(lgbm_cats) == len(cat_feature_names):
                for idx, col in enumerate(cat_feature_names):
                    X_input[col] = pd.Categorical(X_input[col], categories=lgbm_cats[idx], ordered=False)
            elif isinstance(lgbm_cats, dict):
                for col in cat_cols:
                    if col in lgbm_cats:
                        X_input[col] = pd.Categorical(X_input[col], categories=lgbm_cats[col], ordered=False)
        
        # 2. Zamiana stringów na dokładne kody numeryczne zapisane w węzłach drzew decyzyjnych
        for col in cat_cols:
            X_input[col] = X_input[col].cat.codes
            
        # Zabezpieczenie całej macierzy i wymuszenie typu numerycznego
        X_input = X_input.astype(np.float32)
        
        # Wykorzystanie .values przekształca DataFrame w numpy.ndarray. 
        # Omija to błąd "categorical_feature do not match".
        prediction = estimator.predict(X_input.values)[0]
        
    else:
        # Pętla wykonawcza dla modeli klasycznych (RandomForest, SVR, LinearRegression)
        for col in cat_cols:
            X_input[col] = X_input[col].cat.codes
        X_input = X_input.astype(np.float32)
        
        prediction = estimator.predict(X_input)[0]
        
    final_delay = max(0.0, float(prediction))

    risk_category = classify_risk(final_delay)
    return f"{final_delay:.2f} min", risk_category

# ==========================================
# 5. STRUKTURA INTERFEJSU GRAFICZNEGO
# ==========================================

with gr.Blocks(title="Ewaluacja Opóźnień Lotów (USA)") as app:
    
    model_state = gr.State(None)
    
    gr.Markdown("## Moduł Ewaluacji Opóźnień Lotów (Niezbalansowana Regresja)")
    
    with gr.Row():
        with gr.Column():
            gr.Markdown("### I. Konfiguracja Przestrzeni Modelu")
            
            model_selector = gr.Dropdown(
                label="Wybierz Model", 
                choices=AVAILABLE_MODEL_FILES, 
                value=AVAILABLE_MODEL_FILES[0] if AVAILABLE_MODEL_FILES else None
            )
            load_btn = gr.Button("Załaduj model", variant="primary")
            
            strategy_selector = gr.Dropdown(
                label="Metodologia preprocessingu", 
                choices=[], 
                interactive=False
            )
            status_output = gr.Textbox(label="Status systemu", interactive=False)
            
        with gr.Column():
            gr.Markdown("### II. Wektor Cech Niezależnych")
            date_in = gr.Textbox(label="Data (YYYY-MM-DD)", value="2026-06-15")
            dep_in = gr.Textbox(label="Odlot (HH:MM)", value="14:30")
            arr_in = gr.Textbox(label="Przylot (HH:MM)", value="16:45")
            
            carrier_in = gr.Dropdown(
                label="Przewoźnik komercyjny", 
                choices=CARRIERS, 
                allow_custom_value=True
            )
            
            orig_city_in = gr.Dropdown(label="Węzeł początkowy (City)", choices=CITIES, allow_custom_value=True)
            orig_state_in = gr.Textbox(label="Węzeł początkowy (State)")
            dest_city_in = gr.Dropdown(label="Węzeł docelowy (City)", choices=CITIES, allow_custom_value=True)
            dest_state_in = gr.Textbox(label="Węzeł docelowy (State)")
            
            distance_in = gr.Number(label="Dystans (mi)")
            air_time_in = gr.Number(label="Czas przelotu (min)")
            tz_shift_in = gr.Number(label="Korekta strefy czasowej (min)", value=0)
            
    predict_btn = gr.Button("Wykonaj ewaluację", variant="secondary")
    output_delay = gr.Textbox(label="Estymowane opóźnienie (min)", interactive=False)
    
    load_btn.click(
        fn=load_selected_model,
        inputs=model_selector,
        outputs=[model_state, strategy_selector, status_output]
    )
    
    route_inputs = [orig_city_in, dest_city_in]
    route_outputs = [orig_state_in, dest_state_in, distance_in, air_time_in, tz_shift_in]
    
    orig_city_in.change(fn=update_route_parameters, inputs=route_inputs, outputs=route_outputs)
    dest_city_in.change(fn=update_route_parameters, inputs=route_inputs, outputs=route_outputs)
    
    time_dependencies = [dep_in, air_time_in, tz_shift_in]
    
    dep_in.change(fn=update_arrival_time, inputs=time_dependencies, outputs=arr_in)
    air_time_in.change(fn=update_arrival_time, inputs=time_dependencies, outputs=arr_in)
    tz_shift_in.change(fn=update_arrival_time, inputs=time_dependencies, outputs=arr_in)
    
    predict_btn.click(
        fn=predict_delay,
        inputs=[
            model_state, strategy_selector,
            date_in, carrier_in, orig_city_in, orig_state_in, 
            dest_city_in, dest_state_in, distance_in, air_time_in, 
            tz_shift_in, dep_in, arr_in
        ],
        outputs=output_delay
    )

if __name__ == "__main__":
    app.launch()