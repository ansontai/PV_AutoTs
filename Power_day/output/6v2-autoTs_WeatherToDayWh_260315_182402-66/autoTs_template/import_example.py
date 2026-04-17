from autots import AutoTS
# create AutoTS and import the template to reuse the saved best-model settings
model = AutoTS(model_list='default', n_jobs=1)
model.import_template(r'''t:\OneDrive\1TB\School\python_local\Power_day\output\6v2-autoTs_WeatherToDayWh_260315_182402-66\autoTs_template\autoTs_template_9d.csv''')
# then call model.fit()/predict() as usual
