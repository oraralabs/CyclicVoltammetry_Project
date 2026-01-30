Stopped working on deconvolution for abit to make sure the code was robust, 
Trying to implent the CV generator from here: https://github.com/kiranvad/pyMECSim
Will use it to test our code and upgrade it. 
![[Pasted image 20260126121153.png]]
Starting point 


**source venv/bin/activate**
- do on start up of the program always to ensure packages work properly.
***
1. **Keep our Physics Simulator:** It generates the "peaks" you actually need.
2. **Adopt the Paper’s "NEG/POS/FULL" Splitting:** Train one AI for the Oxidation peaks and one for the Reduction peaks. This is what the researchers did to reach 
    R2>0.98R2>0.98
3. **Adopt the "Segmented Reduction":** Continue using your 1000-point interpolation to keep data "clean" for the Random Forest.
***

![[Pasted image 20260126155252.png]]

****
The objective remains the development of a medical-grade, automated pipeline for detecting ion concentrations from Cyclic Voltammetry (CV) data.
***
### 1. Signal Ingestion and In-Silico Simulation

**Processes Attempted:**

- **Gaussian Modeling:** Initially used simple symmetric Gaussians to represent chemical peaks.
    
- **pyMECSim Integration:** Attempted to use a professional Fortran-based library for physical realism.
    
- **Custom Randles-Sevcik Simulator:** After the pyMECSim binary failure on macOS, we built a pure Python simulator using a skewed Gumbel-wave approximation to mimic diffusion-limited kinetics (sharp rise, long decay tail).
    

**Critical Lessons & Errors:**

- **The "Jump" Error:** Modeling capacitance as a hard "box" created artificial vertical jumps at switching potentials. Corrected by implementing a sine-based hysteresis that ensures continuous current at the ends of the scan.
    
- **The Physicality of "Tails":** Simple Gaussians are insufficient for training. Real electrochemical peaks are asymmetric; the slow decay of the "diffusion tail" is what often obscures secondary peaks (shoulders), making deconvolution necessary.
    

---

### 2. Baseline Subtraction Methodologies

This phase was the most iterative, as the baseline defines the accuracy of all subsequent calculations.

**Processes Attempted:**

- **3rd Order Polynomials:** Failed due to "ballooning" and inverted peaks at scan edges (Runge’s Phenomenon).
    
- **Symmetric Mean Subtraction:** We calculated the center-line of the hysteresis loop 
    
            $(Top+Bottom)/2(Top+Bottom)/2$
          
    
    . This was rejected because it forces the Oxidation and Reduction signals to be mathematical reflections of one another, which is physically impossible and hides unique chemical data.
    
- **Fixed Endpoint Anchors (15%/85%):** Failed on highly curved water-based CV data by "cutting corners," leaving a parabolic arch in the extracted signal.
    

**The Current "Gold Standard":**

- **Independent Linear 1%-99% Anchors:** We process the forward and reverse scans as completely separate datasets. By connecting the extreme 1% and 99% points with a linear fit, we remove the resistive tilt while preserving the unique asymmetric shape of the chemical peaks on each scan.
    

---

### 3. Hyperparameter Optimization & Machine Learning (The "Scout")

To automate scipy.find_peaks, we needed a way to set the prominence threshold without human intervention.

**Processes Attempted:**

- **MIMO (Multi-Input Multi-Output) Prediction:** Attempted to predict both prominence and widthsimultaneously. This failed (
    
            $R^2≈0.0$
          
    
    ) because the two parameters are "coupled" (inverse relationship), confusing the model.
    
- **Empirical Target Labeling:** Used a grid-search "solver" to find the best prominence for synthetic data. This failed due to "Target Jitter," where random noise spikes caused the target to jump wildly, preventing the ML from finding a pattern.
    

**Current Successful Logic:**

- **Deterministic Noise-Floor Targeting:** We shifted the AI's goal. Instead of "guessing" what works, the model is trained to predict the **Maximum Noise Ceiling** 

            $Max_Noise×Safety_Factor$
    
- **Features:** We use Median Absolute Deviation (
    
            $MAD$
          
    
    ), Standard Deviation, and Max Amplitude.
    
- **Result:** This achieved an 
    
            $R2>0.95$
          
    
    and an error of 
    
            $0.1μA<0.1μA$
          
    
    . The AI effectively acts as a "scout" that assesses data quality and tells the detector exactly how sensitive it needs to be.
    

---

### 4. Advanced Deconvolution (Resolving Overlaps)

This phase addressed the problem of "shoulders"—peaks buried inside larger peaks.

**Processes Attempted:**

- **Iterative lmfit:** Tried fitting a Gaussian, checking residuals, and adding more Gaussians. This proved unstable as it was too dependent on the initial find_peaks guess.
    
- **Ridge Regression (Tikhonov):** Inspired by the "DRTtools" paper. Resulted in "blurry" mountains. While the math was stable, it lacked the precision needed to identify specific ion locations.
    
- **Lasso Regression (L1 Sparsity):** We attempted to force the math to explain the curve using the fewest possible sharp spikes.
    

**Critical Failures:**

- **The "Forest of Spikes":** Early Lasso attempts used a basis width that was too narrow, resulting in 130+ tiny spikes fitting the noise.
    
- **The L-Curve Corner:** We attempted to automate the selection of the regularization parameter (by finding the "elbow" of the L-curve. This is mathematically sound but computationally expensive.)
---

### 5. Summary of Current Architecture

We currently have a stable, three-stage pipeline:

1. **Ingestion:** Robust CSV parsing with 1%-99% independent linear baseline subtraction.
    
2. **Scouting (ML):** A Random Forest assesses the noise level and sets a conservative prominence threshold.
    
3. **Detection:** A simple, high-sensitivity peak finder identifies major chemical features.
    

**Next Strategic Move:**  
We must finalize the **Deconvolution** logic. The current bottleneck is that we can see the "shoulder" peak visually in the extracted signal, but the simple detector still misses it or groups it with the main peak. The goal is to refine the Lasso method to consolidate "spike clusters" into a single, high-accuracy coordinate for each chemical species.


****
# Deconvolution Attempt 
![[Pasted image 20260126165619.png]]
![[Pasted image 20260126165634.png]]

***
# Validating Results
Finding the parameters for the Lasso regression and basis width. 

- Generate 50 different sunthetic CVs, where we know where the peak is
- Then try 100 different combinations to find optimal parameters 

![[Pasted image 20260126174445.png]]

 Tested: Width=0.05, AlphaMult=0.01 -> Score: 1345.8
  Tested: Width=0.05, AlphaMult=0.05 -> Score: 1481.9
  Tested: Width=0.05, AlphaMult=0.1 -> Score: 1457.6
  Tested: Width=0.05, AlphaMult=0.2 -> Score: 1341.6
  Tested: Width=0.05, AlphaMult=0.5 -> Score: 600.0
  Tested: Width=0.08, AlphaMult=0.01 -> Score: 901.2
  Tested: Width=0.08, AlphaMult=0.05 -> Score: 658.6
  Tested: Width=0.08, AlphaMult=0.1 -> Score: 626.9
  Tested: Width=0.08, AlphaMult=0.2 -> Score: 822.0
  Tested: Width=0.08, AlphaMult=0.5 -> Score: 141.3
  Tested: Width=0.1, AlphaMult=0.01 -> Score: 736.4
  Tested: Width=0.1, AlphaMult=0.05 -> Score: 661.0
  Tested: Width=0.1, AlphaMult=0.1 -> Score: 511.6
  Tested: Width=0.1, AlphaMult=0.2 -> Score: 517.9
  Tested: Width=0.1, AlphaMult=0.5 -> Score: 298.9
  Tested: Width=0.12, AlphaMult=0.01 -> Score: 671.7
  Tested: Width=0.12, AlphaMult=0.05 -> Score: 438.3
  Tested: Width=0.12, AlphaMult=0.1 -> Score: 451.9
  Tested: Width=0.12, AlphaMult=0.2 -> Score: 392.2
  Tested: Width=0.12, AlphaMult=0.5 -> Score: 474.7
  Tested: Width=0.15, AlphaMult=0.01 -> Score: 522.3
  Tested: Width=0.15, AlphaMult=0.05 -> Score: 455.7
  Tested: Width=0.15, AlphaMult=0.1 -> Score: 373.9
  Tested: Width=0.15, AlphaMult=0.2 -> Score: 327.2
  Tested: Width=0.15, AlphaMult=0.5 -> Score: 393.7
--- WINNING SETTINGS FOUND ---
Optimal Basis Width:    0.08
Optimal Alpha Mult:     0.5
Resulting Avg Error:    41.31 mV