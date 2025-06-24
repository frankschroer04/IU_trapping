This code existed in our lab in the form of a Mathematica script. Ilyoung Jung and myself translated it to python and expanded upon it. 

This code takes in imported COMSOL files. You will have three files that correspond to x,y, and z for each independently controlled electrode. It will generate the electrostatic plots and secular frequencies for a given electrode geometry and voltage set. Additionally, it can calculate the equilibrium positions and normal mode spectrum. This should be able to handle whatever ion geometry you can dream up using your electrodes that have been imported into COMSOL and simulated.

As I mentioned on the primary README for this repo, I'm just OK at coding. I plan on condensing this down by implementing classes! This is a messy one that needs some work.
