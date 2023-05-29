# Fast, Smooth and Safe - LCSS + CDC submission

This code is the companion to results in https://sites.google.com/view/fsslcss/home .
Here, we construct a CBF up to second-order in real-time using DDP optimization. Then, the online quadratic program is solved to obtain the filtered safety control. Without interfering constraints, the filtered controls display better smoothness properties. With multiple constraints from yaw and road boundaries, controls are jerkier but behavior of trajectories is safer and smoother. We use JAX to obtain acceleration while running on a CPU.

Further, the directory numpy_dubins contains our numpy testing framework on the 3D dubins car.

Python dependencies for running the code is available inside 'requirements.txt'.
