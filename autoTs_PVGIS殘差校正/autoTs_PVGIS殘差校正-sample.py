# df: columns = [date, y_obs, y_pvgis_daily]
df['month'] = df['date'].dt.month

# Step1: monthly scaling
k = (df.groupby('month')['y_obs'].sum() / df.groupby('month')['y_pvgis_daily'].sum())
df['y_pvgis_scaled'] = df.apply(lambda r: k.loc[r['month']] * r['y_pvgis_daily'], axis=1)

# Step2: residual for AutoTS
df['residual'] = df['y_obs'] - df['y_pvgis_scaled']

# AutoTS fit on residual (daily frequency), then forecast 365 days
# y_hat_nextyear = y_pvgis_scaled_nextyear + residual_hat_nextyear