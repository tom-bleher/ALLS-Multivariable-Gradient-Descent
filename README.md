#### Description
This is a test for the gradient descent algorithm relying on a definition of an arbitrary function to verify the code correctly finds the maximum. Let's start with testing the code on a convex function, and proceed to test it on more complex functions with local maxima. 

#### Convex parabolic function
For the purpose of testing the `count` was was simulated by the function - no images were processed.

$$C(\phi_{2},\phi_{3},f)=-((\phi_{2}-42)^2 + (\phi_{3}-70)^2 + (f+972)^{2}+3^6)$$
$$\text{Algorithm should arrive at:} \; \phi_{2}=42,\; \phi_{3}=70,\; f=-972$$

in the code this is expressed by the `count_function`:

```python
    def count_function(self, new_focus, new_second_dispersion, new_third_dispersion):
        count_func = -1*(((new_second_dispersion - 42) ** 2) + ((new_third_dispersion - 70) ** 2) + ((new_focus + 972) ** 2)) +3e6
        return count_func
```

where the partial derivatives are calculated:

```python
    def calc_derivatives(self):
        self.count_focus_der = -2*(self.new_focus+972)
        self.count_second_dispersion_der = -2*(self.new_second_dispersion-42)
        self.count_third_dispersion_der = -2*(self.new_third_dispersion-69)
        self.focus_der_history.append(self.count_focus_der)      self.second_dispersion_der_history.append(self.count_second_dispersion_der)
     self.third_dispersion_der_history.append(self.count_third_dispersion_der)
        self.total_gradient = (self.focus_der_history[-1] + self.second_dispersion_der_history[-1] + self.third_dispersion_der_history[-1])
        self.total_gradient_history.append(self.total_gradient)
        self.der_iteration_data.append(self.dir_run_count)
        return {"focus":self.count_focus_der,"second_dispersion":self.count_second_dispersion_der,"third_dispersion":self.count_third_dispersion_der}
```

After running the code the console suggests:
```
initial directions are: focus -1, second_dispersion -1, third_dispersion -1
-------------
convergence achieved
function_value 2053973.0, current values are: focus -197, second_dispersion 6, third_dispersion 13
-------------       
convergence achieved
function_value 2394830.0, current values are: focus -352, second_dispersion 13, third_dispersion 24
-------------       
convergence achieved
function_value 2612643.0, current values are: focus -476, second_dispersion 19, third_dispersion 33
-------------       
convergence achieved
function_value 2752086.0, current values are: focus -575, second_dispersion 24, third_dispersion 40
-------------
convergence achieved
function_value 2982964.0, current values are: focus -868, second_dispersion 37, third_dispersion 62

* * *

-------------
function_value 2999914.0, current values are: focus -967, second_dispersion 37, third_dispersion 64
```

The final line shows we approached `focus -967, second_dispersion 37, third_dispersion 64`. This is due to rounding errors, resulting in the optimized values not being exact, but we get very close. **The optimization algorithm works!**
