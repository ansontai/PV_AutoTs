from autots import AutoTS
# create AutoTS and import the template to reuse the saved best-model settings
model = AutoTS(model_list='default', n_jobs=1)
model.import_template(r'''t:\OneDrive\1TB\School\python_local\Power_day_v3\output\6v3-autoTs_WeatherToDayWh_260324_013114-208\autoTs_template\autoTs_template_90d.csv''')
# then call model.fit()/predict() as usual
