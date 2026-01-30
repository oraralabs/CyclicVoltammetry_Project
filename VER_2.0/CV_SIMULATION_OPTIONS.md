# CV Simulation Options Analysis

## Repositories Examined

### 1. pyMECSim (https://github.com/kiranvad/pyMECSim)
- **Type:** Python wrapper around MECSim (C++/Fortran)
- **Pros:** Very accurate mechanistic simulations
- **Cons:** 
  - Requires compilation of C++ executable
  - Complex dependency chain
  - Not easily portable

### 2. Crypton (https://github.com/WYAlChEMIST/Crypton)
- **Type:** Compiled MATLAB executable
- **Pros:** Fast execution
- **Cons:**
  - Not accessible as Python code
  - Requires MATLAB Runtime
  - Black box (can't modify)

---

## Problem with Current Simulator

Your observation is correct: the reduction reaction in our `MechanisticSimulator` is just a mirrored/flipped oxidation peak:

```python
# Current code (WRONG):
i_faradaic[self.points:] += self._get_norm_current(rev_v, e0 - 0.06, 1) * -h * 0.95
```

This creates unrealistic CV curves that won't match real data.

---

## Proposed Solution: Electrochemical CV Generator

Implement from first principles using established equations:

### Option A: Randles-Ševčík (Simple, Fast)
For **reversible** systems:
```
i_p = 0.4463 * n * F * A * C * sqrt(n * F * D * v / (R * T))
```
- Gives correct peak current
- Proper peak-to-peak separation (ΔEp = 59/n mV)
- Fast to compute

### Option B: Butler-Volmer + Finite Difference (Accurate)
For **quasi-reversible** systems:
- Numerically solve diffusion equation
- Apply Butler-Volmer kinetics at interface
- Realistic peak shapes with tails

### Option C: Try pyMECSim Installation
- Attempt to compile MECSim executable
- May take significant effort
- Highest accuracy but complex

---

## Recommendation

**Start with Option A** (Randles-Ševčík):
1. Fast implementation (~1 day)
2. Mathematically correct oxidation/reduction
3. Good enough for breast milk ions (mostly reversible)
4. Can upgrade to Option B later if needed

**Timeline:**
- Day 1: Implement Randles-Ševčík generator
- Day 2: Generate 20k training samples
- Day 3: Retrain model, compare to current approach

---

## Next Step Decision Needed

Which approach should I pursue?
- [ ] Option A: Randles-Ševčík (fast, simple, correct)
- [ ] Option B: Butler-Volmer + diffusion (slower, more accurate)
- [ ] Option C: Install pyMECSim (time-consuming, most accurate)
