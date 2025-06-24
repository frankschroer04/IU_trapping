
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
import sympy as sp
import qutip as qt
from scipy.constants import elementary_charge
from scipy.constants import epsilon_0
from scipy.constants import hbar

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.optimize import minimize, curve_fit
from numba import jit
import time

#%% FUNCTIONS


# @title Functions
def equil_positions(N):

  """
  Calculates equilibrium positions for N ions under harmonic trapping potentials \
  along all spatial axis.

  Parameters:
  N (int): The number of ions in the system.

  Returns:
  chopped_solution (array): Array of equilibrium positions.

  """
  def coupled_equations(u):
      equations = []
      for m in range(N):
          sum1 = sum(1 / (u[m] - u[n])**2 for n in range(m))
          sum2 = sum(1 / (u[m] - u[n])**2 for n in range(m+1, N))
          equations.append(u[m] - sum1 + sum2)
      return equations

  # Generate initial conditions
  initial_guess = [m / 10 for m in range(1, N+1)]

  # Solve the coupled equations
  solution = fsolve(coupled_equations, initial_guess)

  # Chop (close to zero precision) and return the solution
  chopped_solution = np.round(solution, decimals=9)

  return chopped_solution


def scaled_positions(u,lc):
  """
  Calculates the true equilibrium positions of ions in meters.

  Parameters:
  u (array): Array of equilibrium positions.
  lc (float): The lattice constant in meters.

  Returns:
  u*lc (array): Array of equilibrium positions in meters.

  """
  pos = equil_positions(N)

  return pos*lc

def eigensystem(frad,fax,N):
  """
  Calculates the mode spectrum for a given set of ions at specific trap frequencies.

  Parameters:
  frad (float): Radial Trap frequency in Hz.
  fax (float): Axial trap frequency in Hz.
  num_ions (int): The number of ions in the system.
  lc (float): The lattice constant in meters.

  Returns:
  eigenvalues (array): Normal mode frequencies in MHz. Need to be
  multiplied by 10^6 to get in Hz.

  eigenvectors (array): Array of eigenvectors.
  """
  u = equil_positions(N)


  Anm = np.empty((N,N))

  for n in range(N):
    for m in range(N):
      if n == m:
        sum1 = 0.0
        for p in range(N):
          if(p!=m):
            sum1 += 1/abs(u[m] - u[p])**3
        Anm[n][n]= (frad/fax)**2-sum1
      elif n!=m:
        sum2 = 1/ abs(u[m] - u[n])**3
        Anm[n][m]= sum2


  eigenvalues, eigenvectors = np.linalg.eig(Anm)
  eigenvalues = np.sqrt(eigenvalues)*fz
  eigenvectors = eigenvectors.T

  return eigenvalues, eigenvectors



def J(Omegas,bmk,wmk,mode_detune,det,recoil,num_ions):
  """
  Calculates the J matrix that dictate ion-ion coupling. Requires the inputs of
  frequencies to be in Hz.

  Note: we could modify for in the future to include position calculations.

  Parameters:

  Omegas (array): Array of rabi frequencies in Hz.
  bmk (array): Array of eigenvectors.
  wmk (array): Array of eigenvalues.
  mode_detune (float): The mode being detuned from in Hz. e.g. COM.
  det (float): Detuning from mode in Hz.
  recoil (float): Recoil frequency in Hz.
  num_ions (int): Number of ions.

  Returns:
  J (array): The J matrix.
  """
  mu = mode_detune + det
  J = np.zeros((num_ions,num_ions),dtype=float)

  for i in range(num_ions):
    for j in range(num_ions):
      if i!=j:
        s = sum( (bmk[k][i]*bmk[k][j])/ ((mu)**2-wmk[k]**2) \
                for k in range(num_ions))
        J[i][j] = recoil*Omegas[i]*Omegas[j] *s / (2*np.pi*10**3)

  return J



#ground state generation of N particles
def grnd_state(N):
  """
  Constructs ground state for an N-particle system.

  Parameters:
  N (int): The number of particles in the system.

  Returns:
  qt.Qobj: The ground state of the system.

  """
  state = []
  for i in range(N):
    state.append(qt.basis(2,0))
  grnd_state = qt.tensor(state)
  return grnd_state

#operator generation for two-body XX interaction
def XX(i,j,N):
  """
  Constructs a two-body XX interaction for an N-particle system.

  Parameters:
  i (int): The index of the first particle.
  j (int): The index of the second particle.
  N (int): The total number of particles in the system.

  Returns:
  qt.Qobj: X on i and j, identity everywhere else.
  """
  operators = []

  for k in range(N):
    if i == k:
      operators.append(qt.sigmax())
    elif j ==  k:
      operators.append(qt.sigmax())
    else:
      operators.append(qt.qeye(2))

  return qt.tensor(operators)

#operator generation for two-body YY interaction
def YY(i,j,N):
  """
  Constructs a two-body XX interaction for an N-particle system.

  Parameters:
  i (int): The index of the first particle.
  j (int): The index of the second particle.
  N (int): The total number of particles in the system.

  Returns:
  qt.Qobj: Y on i and j, identity everywhere else.
  """
  operators = []

  for k in range(N):
    if i == k:
      operators.append(qt.sigmay())
    elif j ==  k:
      operators.append(qt.sigmay())
    else:
      operators.append(qt.qeye(2))

  return qt.tensor(operators)

#operator generation for two-body ZZ interaction
def ZZ(i,j,N):
  """
  Constructs a two-body ZZ interaction for an N-particle system.

  Parameters:
  i (int): The index of the first particle.
  j (int): The index of the second particle.
  N (int): The total number of particles in the system.

  Returns:
  qt.Qobj: Z on i and j, identity everywhere else.
  """
  operators = []

  for k in range(N):
    if i == k:
      operators.append(qt.sigmaz())
    elif j ==  k:
      operators.append(qt.sigmaz())
    else:
      operators.append(qt.qeye(2))

  return qt.tensor(operators)


def HamiltonianXX(J,N):

  """
  Calculates the overall Hamiltonian for an N-particle system.

  Parameters:
  J (array): The J matrix.
  N (int): The number of particles in the system.

  Returns:
  H (array): The Hamiltonian.
  """
  H = []
  for i in range(N):
    for j in range(N):
      if i < j:
        val = 2*np.pi*(J[i][j] * XX(i,j,N))
        H.append(val)
  return sum(H)


def KroneckerDelta(i,j):
  """
  Calculates the Kronecker delta function.

  Parameters:
  i (int): The first index.
  j (int): The second index.

  Returns:
  int: The Kronecker delta function.
  """
  if i == j:
    return 1
  else:
    return 0

def JijPowerLaw(Jo,N):
  """
  Calculates the J matrix for power law approximation.

  Parameters:
  Jo (array): Constant coupling Jo.
  N (int): The number of particles in the system.

  Returns:
  J (array): Coupling matrix for power law coupling, J.
  """

  J = np.zeros((N,N),dtype=float)
  for i in range(N):
    for j in range(N):
      if i != j:
        J[i][j] = Jo / abs(i-j)**3
  return J


def chop(array, tol=1e-10):
    """
    Replace small values in the array with zero.

    Parameters:
    array: list or numpy array
        The array or list to process.
    tol: float, optional
        The tolerance level below which values are considered zero (default: 1e-10).

    Returns:
    numpy array
        The array with small values replaced by zero.
    """
    arr = np.array(array)
    return np.where(np.abs(arr) < tol, 0, arr)

def GHZ(N):
    """
    Constructs GHZ state for an N-particle system.

    Parameters:
    N (int): The number of particles in the system.

    Returns:
    qt.Qobj: The ground state of the system.

    """
    state0 = []
    state1 = []
    for i in range(N):
      state0.append(qt.basis(2,0))
      state1.append(qt.basis(2,1))
    state00 = qt.tensor(state0)
    state11 = qt.tensor(state1)
    ghz_state = (1/np.sqrt(2))*(state00 + state11)
    return ghz_state


#equilibrium positions
def find_positions(N,wx,wy,wz):
        
    #@jit(nopython=True, fastmath=False)
    def potential_energy(positions):
        # wx, wy, wz, N, m, e, epsilon_0 = params
        xs = positions[:N]
        ys = positions[N:2*N]
        zs = positions[2*N:3*N]
    
        # harmonic energy
        harm_energy = np.sum(1/2 * m * (wx**2 * xs**2 + wy**2 * ys**2 + wz**2 * zs**2))
    
        # electronic interaction
        interaction = 0
        for i in range(N):
            for j in range(N):
                if j != i:
                    interaction += 1/np.sqrt((xs[i]-xs[j])**2 + (ys[i]-ys[j])**2 + (zs[i]-zs[j])**2)
        interaction = q_e**2/4/np.pi/epsilon_0 * interaction
    
        return harm_energy + interaction
    
    
    xs_0 = np.random.choice(range(-N,N), N) * 1e-6
    #xs_0 = np.arange(0, N) * 1e-6
    ys_0 = np.random.choice(range(-N,N), N) * 1e-6
    #ys_0 = np.array([0] * N) * 1e-6
    #zs_0 = np.array([0] * N) * 1e-6
    zs_0 = np.random.choice(range(-N,N), N) * 1e-6
    
    pos = np.append(np.append(xs_0, ys_0), zs_0)
    
    plt.plot(xs_0 * 1e6, ys_0 * 1e6, '.', color='royalblue', markersize=12, markeredgecolor='darkblue')
    plt.xlabel('x distance (micron)')
    plt.ylabel('y distance (micron)')
    plt.title('Initial ion position guess')
    plt.grid()
    plt.show()
    
    params = [wx, wy, wz, N, m, q_e, epsilon_0]
    # pot_en = potential_energy(pos, *params)
    pot_en = potential_energy(pos)
    print('Potential Energy', pot_en)
    
    time_start = time.time()
        
        
    # run bad guess through optimization with on a few iterations
    pos_better = minimize(potential_energy, pos, method='COBYLA', options={'tol':1e-30, 'maxiter':500})
    
    # Get fine results with better initial best and many iterations
    res = minimize(potential_energy, pos_better.x, method='COBYLA', options={'tol':1e-30, 'maxiter':80000})
    
    time_end = time.time()
    
    
    time_total =  time_end - time_start
    print('Total time', time_total)
    
        
    xs_f = res.x[:N]
    ys_f = res.x[N:2*N]
    zs_f = res.x[2*N:3*N]
    
    
    pos = np.array([xs_f,ys_f,zs_f])
    pos = chop(pos)
    rad_pos = np.array([pos[1],pos[2]]).T
    rad_pos = rad_pos[np.argsort(rad_pos[:, 0])]
    rad_pos = rad_pos.T
    
    xs_f = chop(xs_f)
    ys_f = rad_pos[0]
    zs_f = rad_pos[1]
        
    r_lim = 5.0
    plt.plot(xs_f * 1e6, zs_f * 1e6, '.', markersize=16, color='royalblue', markeredgecolor='k')
    plt.title('Ion equilibrium position')
    plt.xlabel('x distance (micron)')
    plt.ylabel('z distance (micron)')
    plt.ylim(-r_lim, r_lim)
    plt.xlim(-r_lim, r_lim)
    plt.grid()
    plt.gca().set_aspect('equal')
    plt.show()
    
    plt.plot(xs_f * 1e6, ys_f * 1e6, '.', markersize=16, color='royalblue', markeredgecolor='k')
    plt.title('Ion equilibrium position')
    plt.xlabel('x distance (micron)')
    plt.ylabel('y distance (micron)')
    plt.ylim(-r_lim, r_lim)
    plt.xlim(-r_lim, r_lim)
    plt.grid()
    plt.gca().set_aspect('equal')
    plt.show()
    
    
    plt.plot(ys_f * 1e6, zs_f * 1e6, '.', markersize=16, color='royalblue', markeredgecolor='k')
    plt.title('Ion equilibrium position')
    plt.xlabel('y distance (micron)')
    plt.ylabel('z distance (micron)')
    plt.ylim(-r_lim, r_lim)
    plt.xlim(-r_lim, r_lim)
    plt.grid()
    plt.gca().set_aspect('equal')
    plt.show()
    
    return xs_f,ys_f,zs_f

#eigenspectrum
def TwoDmodes(N,x,y,z,w_axial):
    Anm = np.zeros((N, N))
    xs = x / lc
    ys = y / lc
    zs = z / lc
    for i in range(N):
        for j in range(N):
            if i != j:
                dx = xs[i]-xs[j]
                dy = ys[i]-ys[j]
                dz = zs[i]-zs[j]
                print(dx,dy,dz)
                Anm[i, j] = 1 / (dx**2 + dy**2 + dz**2)**(3/2)
            else:
                Anm[i, i] = 1 - sum(
                    1 / ( (xs[k] - xs[j])**2 +(ys[k] - ys[j])**2 +
                           (zs[k] - zs[j])**2 )**(3/2)
                    for k in range(N) if k != j)
    evals, evects = np.linalg.eig(Anm)
    evals = evals * w_axial / (2 * np.pi) * 1e-6

    idx = np.argsort(evals)
    evals_sorted = evals[idx]
    evects_sorted = evects[:, idx]
    
        
    #plot
    for j in range(N):
        positions = chop(np.array([y,z])).T
        
        values = evects_sorted.T[j]
        values = np.round(values, decimals=3)  # or even decimals=3
        print(np.round(evals_sorted[j],10),np.round(evects_sorted.T[j],5))
        norm = plt.Normalize(min(values), max(values))
        
        # Create a colormap
        cmap = plt.cm.bwr  #viridis
        
        # Create figure and axis
        fig, ax = plt.subplots()
        
        # Plot each circle with color based on value
        for i, (pos, value) in enumerate(zip(positions, values)):
            color = cmap(norm(value))
            circle = plt.Circle(pos, radius=1.75e-6, color=color, ec='black')
            ax.add_patch(circle)
            # Add a label (e.g., index or custom text)
            ax.text(pos[0], pos[1], f'{i+1}', color='white', ha='center', va='center', fontsize=8)
        # Add colorbar
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])  # Required only for older Matplotlib versions
        cbar = fig.colorbar(sm, ax=ax, orientation='vertical')
        cbar.set_label('Eigenvector Component', fontsize=10)
        ax.set_aspect('equal')
        ax.set_xlim(-30-6, 30e-6)
        ax.set_ylim(-30e-6, 30e-6)
        plt.title(evals_sorted[j])
        plt.axis('off')
        plt.show()
        
    return evals_sorted, evects_sorted


def exp_parameters(N,wx,wy,wz):
    
    x,y,z = find_positions(N, wx, wy, wz)
    pos = np.array([x,y,z])
    
    evals, evects = TwoDmodes(N, x, y, z, wx)
    
    return pos, evals,evects.T
    
    

def J_mat(Omegas,bmk,wmk,mode_detune,det,recoil,N):

  mu = mode_detune + det
  J = np.zeros((N,N),dtype=float)

  for i in range(N):
    for j in range(N):
      if i!=j:
        s = sum( (bmk[k][i]*bmk[k][j])/ ((mu)**2-wmk[k]**2) \
                for k in range(N))
        J[i][j] = recoil*Omegas[i]*Omegas[j] *s / (2*np.pi*10**3)

  return J


x = qt.sigmax()
y = qt.sigmay()
z = qt.sigmaz()
i2 = qt.qeye(2)


#%%

m = 170.936323 * 1.66054e-27  # ion mass
q_e = elementary_charge
eps_o = epsilon_0
M_Yb = 171*1.67262192e-27
delta_k = np.sqrt(2)*2*np.pi/(355e-9)

wx = 2 * np.pi * 1000*1e3
wy = 2 * np.pi * 500*1e3
wz = 2 * np.pi * 500*1e3



lc = (q_e**2/(4*np.pi*eps_o*M_Yb*wx**2))**(1/3)
recoil = (hbar*(delta_k)**2 ) / (2*M_Yb)


#%% TWO IONS
N = 3
Omegas = 200*np.ones(N)
Omegas = .5*Omegas #split between beams 

radial_condition = (wx/wz) > (2.264*N)**(.25)
positions, eigvals, eigvects = exp_parameters(N,wx,wy,wz)

#%%

#setup Jij parameters
COM_freq = np.max(eigvals)
evects_sorted = eigvects.T

detuning = 25*1e3 

Jij = J_mat(Omegas*1e3,evects_sorted,eigvals*1e6,COM_freq*1e6,detuning,recoil,N)
H  = HamiltonianXX(Jij, N)
#%%
print(Jij)
#setup dynamics calulcation
times = np.linspace(0,2.5,1000)

state_00 = grnd_state(N)

expzi = qt.tensor(z,i2)
expiz = qt.tensor(i2,z)
mag_00 = expzi + expiz 
exps_00 = [expzi,expiz,mag_00]
labels_00 = ['ZI','IZ', "ZI+IZ"] 

two_ion_results = qt.mesolve(H,state_00,times,[],exps_00)


#plot results 

rows = 3
cols = 2

fig, axes = plt.subplots(rows,cols,figsize=(8,8))

axes = axes.flatten()

for i in range(len(labels_00)):
    y = two_ion_results.expect[i]
    axes[i].plot(times,y,label=labels_00[i])
    axes[i].legend()
    axes[i].set_title(r'$\langle$'+labels_00[i] + r'$\rangle$')

    #axes[i].plot(pulse_times,mean_spin_data[i])


# Adjust the layout to prevent overlapping
plt.tight_layout()

# Show the plot
plt.show()
