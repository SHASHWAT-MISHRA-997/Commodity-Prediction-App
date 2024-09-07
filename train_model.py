import pandas as pd
import joblib
from sklearn.linear_model import LinearRegression

def train_and_save_model(commodity_data, model_filename):
    X = commodity_data[['Date']].values.reshape(-1, 1)  # Assuming date is transformed to numeric
    y = commodity_data['Price']  # Assuming price column is the target
    model = LinearRegression()
    model.fit(X, y)
    joblib.dump(model, model_filename)

# Load your dataset
df = pd.read_csv('Commodity_prices.csv')

# List of commodities to create models for
commodities = [
    'NATURAL GAS', 'GOLD', 'WTI CRUDE', 'BRENT CRUDE', 'SOYBEANS',
    'CORN', 'COPPER', 'SILVER', 'LOW SULPHUR GAS OIL', 'LIVE CATTLE',
    'SOYBEAN OIL', 'ALUMINIUM', 'SOYBEAN MEAL', 'ZINC', 'ULS DIESEL',
    'NICKEL', 'WHEAT', 'SUGAR', 'GASOLINE', 'COFFEE', 'LEAN HOGS',
    'HRW WHEAT', 'COTTON'
]

for commodity in commodities:
    if commodity in df.columns:
        train_and_save_model(df[['Date', commodity]], f'models/model_{commodity.lower().replace(" ", "_")}.pkl')
    else:
        print(f"Commodity data not found: {commodity}")
