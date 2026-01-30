***
**Polynomial Overshoot**
- Runge's Phenomenon 

- Background current must be lower than the total signal. 
- Baseline was higher than the signal, leading to negative peaks. 
![[Pasted image 20260119130149.png]]

Must constrain the model by switching to a linear curve. 


| 1. Cubic Fit  | `polyfit(order=3)` | Math was too flexible, giant arc above data |
| ------------- | ------------------ | ------------------------------------------- |
| 2. Linear Fit | `polyfit(order=1)` | Too stiff. Cut through data                 |
| 3. Mean line  | `(top+bot)/2`      | Created symmetrical visual                  |
## Solution 
Abandoned `polyfit` used a geometric tool. 

1. Geometric Anchor
	- `get_endpoint_baseline`
	- Draws a straight line between the start and end point 

2. Edge Clamping 
	- `clean_edges`
	- Force the first and last 5% to be 0, avoids negative artifacts from switching 

3. Result
	1. Normalised Data - starts/ends at 0 
	2. Independant - top and bottom proccessed independantly
	3. Clean - no mathematical artifacts

