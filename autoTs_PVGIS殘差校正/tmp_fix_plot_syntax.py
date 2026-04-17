from pathlib import Path

path = Path(__file__).with_name('autoTs_PVGIS殘差校正.py')
text = path.read_text(encoding='utf-8')
old = '''						pvgis_scaled_test = pvgis_test.values * scale_vals
						residual_pred_aligned = residual_pred_series.reindex(test_idx).fillna(0).values
						corrected_preds = pvgis_scaled_test + residual_pred_aligned
						corrected_scores, *_ = compute_forecast_scores(test_df_base["y_obs"].astype(float).values, corrected_preds.astype(float), train_df_base["y_obs"].astype(float).values)
					try:
						y_true_train = train_df_base["y_obs"].astype(float).values
						y_naive = np.r_[y_true_train[-1], test_df_base["y_obs"].astype(float).values[:-1]]
						plot_path = out_base / f'forecast_vs_actual_vs_naive_lag1_vs_PVGIS_{h}d.png'
						plot_actual_vs_pvgis_vs_corrected(
							plot_path,
							test_idx,
							test_df_base["y_obs"].astype(float).values,
							pvgis_test.values,
							corrected_preds,
							y_naive=y_naive,
							title=f'Actual vs PVGIS vs Corrected Forecast ({h}d)',
						)
					except Exception as e:
						print(f'Failed to save PVGIS comparison plot for {h}d:', e)
				except Exception:
					corrected_scores = None
'''
new = '''						pvgis_scaled_test = pvgis_test.values * scale_vals
						residual_pred_aligned = residual_pred_series.reindex(test_idx).fillna(0).values
						corrected_preds = pvgis_scaled_test + residual_pred_aligned
						corrected_scores, *_ = compute_forecast_scores(test_df_base["y_obs"].astype(float).values, corrected_preds.astype(float), train_df_base["y_obs"].astype(float).values)
						try:
							y_true_train = train_df_base["y_obs"].astype(float).values
							y_naive = np.r_[y_true_train[-1], test_df_base["y_obs"].astype(float).values[:-1]]
							plot_path = out_base / f'forecast_vs_actual_vs_naive_lag1_vs_PVGIS_{h}d.png'
							plot_actual_vs_pvgis_vs_corrected(
								plot_path,
								test_idx,
								test_df_base["y_obs"].astype(float).values,
								pvgis_test.values,
								corrected_preds,
								y_naive=y_naive,
								title=f'Actual vs PVGIS vs Corrected Forecast ({h}d)',
							)
							except Exception as e:
								print(f'Failed to save PVGIS comparison plot for {h}d:', e)
						except Exception:
						corrected_scores = None
'''
if old not in text:
    raise SystemExit('Old block not found')
path.write_text(text.replace(old, new), encoding='utf-8')
print('patched')
