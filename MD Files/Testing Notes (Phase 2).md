***
Block 1: 

Generated synthetic spectrum with noise. Known truths aswell 

![[Pasted image 20260121143321.png]]

***
Block 2: 

![[Pasted image 20260121143335.png]]




****
BLOCK 3

- Fro white noise, width is the most dominant parameter, not prominence 

width == 
prominenace == 

![[Pasted image 20260121143359.png]]
To maintain a cost of 0, 100% accuracy, the prominenance must scale linearly with the noise level.
- We need an AI that looks at the data, measures the noise, to predict the prominenance to allow for accurate peak fitting
- Above graph was with the removal of width as a parameter. 


*** 
BLOCK 4

![[Pasted image 20260121144434.png]]
***
BLOCK 5

![[Pasted image 20260121144452.png]]


***
BLOCK 6

![[Pasted image 20260122120647.png]]
Left: Strict prominenance (2.0)
- Found 10 peaks 
- Successfully found main peaks but then hallucinated false positives

Right: AI method (3.05)
- Reduced fale positives to 3
- Raised the prominence when it saw that noise was high 
- With an adjust of the width it should be able to remove the false positives

***
**NOTES**

![[Pasted image 20260122160613.png]]

*Why not use width?*
- When predicting both prominence and width, the system failed. 
- When fixing width, removing a variable, made the system significantly stable. 
- Allows the model to focus on the Noise

*Noise Estimator*
The model is currently acting as a noise estimator. 
$$\text{Target Prominenance}=\text{Maximum Noise Spike}\times1.2$$
Set the threshold to 20% higher than the loudest noise spike

Input: Ai looks at raw signal, calculates Median Absolute Deviation
Logic: AI learns, if line is shaking by X amount, highest spike is likely Y amount
Output: sets prominence to Y\*1.2
