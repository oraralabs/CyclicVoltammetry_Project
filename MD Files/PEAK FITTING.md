Write the overview of the task with each stage and link concepts



# 15 Jan 
## 1. What we have
Have a script that performs Physical Seperation 

- Input: Raw, messy CSV data from the potentiostat
- Processing:
	- Standardises time ordering 
	- Interpolates forward and reverse scans onto a shared voltage grid 
	- Calculates the Capacitive Baseline 
	- Subtracts the baseline from the raw signal 
- Output: Flat line with humps (extracted oxididation and reduction signals)
- Current detection method: `np.argmax`

## 2. Gap 
We need to apply peak fitting to accuractly gauge and extract the concentration

1. Noise Vulnerability: 
	1. Peak fitting ignores spikes and looks for shape 
2. Overlapping peaks
	1. Peak fitting mathematically will seperate peaks 
3. Area vs Height
	1. Fitting gives the area under the curve which is a better measure 

## 3. Next Steps 
Module 1: Signal Extractor (DONE)

Module 2: Peak Deconvoluter (NEXT)
- Goal: Take the `y_signal` and decompose into mathematical functions 
- Library: `scipy.signal.find_peaks` and `lmfit` to optimise the fit
- Tasks: 
	1. Peak Finding 
	2. Model building 
	3. Optimisation 
	4. Output 

Module 3: Quantifier (Calibration)
- Goal: Convert 'Height' to concentration

Module 4: The pipeline (Integration)