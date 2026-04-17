import pandas as pd
from autots import AutoTS

# df_wide: index = DatetimeIndex
# columns 包含: y (目標), tmy_ghi, tmy_temp ... (你的季節性先驗)
# weights: 目標權重大，其它先驗權重小
weights = {"y": 20, "tmy_ghi": 1, "tmy_temp": 1}

model = AutoTS(
    forecast_length=48,
    frequency="H",
    ensemble="simple",
    max_generations=10,
    num_validations=2,
    validation_method="backwards",  # 也可用 'seasonal n'（教學頁有說明）[7](https://winedarksea.github.io/AutoTS/build/html/source/tutorial.html)
)

model = model.fit(df_wide, weights=weights)
pred = model.predict()
yhat = pred.forecast[["y"]]