# Alpha-Loss-Hyperparameter-Tuning

## Alternate Updating for GMM(Gaussian Mixture Models) and Mnist with different noise levels


### package structure
* **gmm** contains files for GMM binary classification w.r.t differnt noise levels(0, 10%, 20%, 30%, 40%).
	* `baseline_plot.py` to run the baseline models and compare how different <img src="https://render.githubusercontent.com/render/math?math=\alpha"> changes result(fixed <img src="https://render.githubusercontent.com/render/math?math=\alpha">, find optimal weight via gradient descent) at multiple noise levels.
	* `run_gibbs_gmm.py` to run alternate updating(slice sampling to sample <img src="https://render.githubusercontent.com/render/math?math=\alpha"> and gradient descent to update weights) with multiple noise levels.
	* `util.py` includes all utility functions(gmm_generator_function[generate data], likelihood_calc, likelihood_calc, etc.).
	* `plot.py` to generate all desired figures like posterior distributions, baseline plot, Geweke Diagnostic, etc.
	
* **Mnist** contain files for binary classification(extract 1 and 7 here) and result comparison:
	* `baseline_plot.py` to run the baseline models and compare how different <img src="https://render.githubusercontent.com/render/math?math=\alpha"> changes result(fixed <img src="https://render.githubusercontent.com/render/math?math=\alpha">, find optimal weight via stochastic gradient descent) at multiple noise levels.
	* `run_gibbs_gmm.py` to run alternate updating(slice sampling to sample <img src="https://render.githubusercontent.com/render/math?math=\alpha"> and stochastic gradient descent to update weights) with multiple noise levels.
	* `util.py` includes all utility functions(get_data[generate data], likelihood_calc, likelihood_calc, etc.).
	* `plot.py` to generate all desired figures like posterior distributions, baseline plot, Geweke Diagnostic, etc.
	
