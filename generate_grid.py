# run this code to generate a grid of points (x,,y,z) which cover the cylinder with a stationary mesh
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy import io as sio
import os

# load the numpy data
data_folder_abs_path = '/mnt/LaCie/RDE/SHRED_data/AFRL_RDRE/1309'
mat_file = 'data_plt00130028'
data = sio.loadmat(f'{data_folder_abs_path}/{mat_file}.mat')
print('Keys in the .mat fileL = ',data.keys())

# get the x,y,z coordinates of the points and print the ranges
x = data['x'].flatten()
y = data['y'].flatten()
z = data['z'].flatten()

print('x range = ',x.min(), 'to', x.max())
print('y range = ',y.min(), 'to', y.max())
print('z range = ',z.min(), 'to', z.max())


# transform the data into cylindrical coordinates and plot the ranges of phi, r, h:
def cart2cyl(x, y, z):
    """Convert Cartesian coordinates to cylindrical coordinates."""
    r = np.sqrt(x**2 + z**2)
    phi = np.arctan2(z, x)
    h = y
    return r, phi, h    
def cyl2cart(r, phi, h):
    """Convert cylindrical coordinates to Cartesian coordinates."""
    x = r * np.cos(phi)
    z = r * np.sin(phi)
    y = h
    return x, y, z

r, phi, h = cart2cyl(x, y, z)
print('phi range = ',phi.min(), 'to', phi.max())
print('r range = ',r.min(), 'to', r.max())
print('h range = ',h.min(), 'to', h.max())

# now generate a grid of points in cylindrical coordinates that cover the cylinder
n_r = 3#3
n_phi = 100 #100
n_h = 100 #100

n_grid = n_r * n_phi * n_h
# so we get a total of n_r * n_phi * n_h points
print('Generating a grid with', n_grid, 'points')

r_grid = np.linspace(r.min(), r.max(), n_r)
phi_grid = np.linspace(phi.min(), phi.max(), n_phi)

grid_type =  "uniform"  # "two_parts"  # "uniform", "two_parts"
if grid_type == "uniform":
    h_grid = np.linspace(h.min(), h.max(), n_h)
elif grid_type == "two_parts":
    h_grid1 = np.linspace(h.min(), h.max()/2, int(n_h * 0.7))
    h_grid2 = np.linspace(h.max()/2, h.max(), int(n_h * 0.3))
    h_grid = np.concatenate((h_grid1, h_grid2))
elif grid_type == "arithmetic_progression":
    h_start = h.min()
    h_end = h.max()
    d = (h_end - h_start) / (n_h * (n_h + 1) / 2)  # common difference
    h_grid = np.array([h_start + d * (i * (i + 1)) / 2 for i in range(n_h)])
elif grid_type == "geometric_progression":
    h_start = h.min()
    h_end = h.max()
    # avoid naming the scalar 'r' which would shadow the array r (cylindrical radii)
    # also guard against h_start being zero or extremely small
    h_start_safe = h_start if h_start > 0 else max(h_start, 1e-12)
    common_ratio = (h_end / h_start_safe) ** (1 / (n_h - 1))  # common ratio
    h_grid = np.array([h_start_safe * (common_ratio ** i) for i in range(n_h)])
else:
    raise ValueError("Invalid grid type")

R, PHI, H = np.meshgrid(r_grid, phi_grid, h_grid, indexing='ij')


X, Y, Z = cyl2cart(R, PHI, H)
print('Grid shape = ',X.shape)  

def plotting():
    global x, y, z, r, phi, h
    # plot what h looks like
    plot_h = True
    if plot_h:
        plt.figure()
        plt.plot(h_grid, 'o-')
        plt.title('Grid in h')
        plt.xlabel('Index')
        plt.ylabel('h value')
        plt.grid()
        plt.savefig('progression_h_grid.png', dpi=300)
        plt.close()
        print('Plot saved to progression_h_grid.png')
    # generate the plot with four subplots: cartesian and cylindrical coordinates for both data and grid
    n_to_plot = 4_000_000

    seed = 42
    # gen t_to_plot random indices from 0 to len(x)-1
    np.random.seed(seed)
    indices = np.random.choice(len(x), size=n_to_plot, replace=False)
    x_subset = x[indices]
    y_subset = y[indices]
    z_subset = z[indices]
    r_subset = r[indices]
    phi_subset = phi[indices]
    h_subset = h[indices]
    fig = plt.figure(figsize=(12, 12))

    def set_axes_equal(ax):
        """Set 3D plot axes to equal scale."""
        x_limits = ax.get_xlim3d()
        y_limits = ax.get_ylim3d()
        z_limits = ax.get_zlim3d()
        x_range = abs(x_limits[1] - x_limits[0])
        y_range = abs(y_limits[1] - y_limits[0])
        z_range = abs(z_limits[1] - z_limits[0])
        max_range = max([x_range, y_range, z_range])
        x_middle = np.mean(x_limits)
        y_middle = np.mean(y_limits)
        z_middle = np.mean(z_limits)
        ax.set_xlim3d([x_middle - max_range/2, x_middle + max_range/2])
        ax.set_ylim3d([y_middle - max_range/2, y_middle + max_range/2])
        ax.set_zlim3d([z_middle - max_range/2, z_middle + max_range/2])

    # 1. Original data in Cartesian coordinates
    ax1 = fig.add_subplot(221, projection='3d')
    ax1.scatter(x_subset, y_subset, z_subset, s=1)
    ax1.set_title('Original Data: Cartesian coordinates')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    set_axes_equal(ax1)

    # 2. Original data in Cylindrical coordinates
    ax2 = fig.add_subplot(222, projection='3d')
    ax2.scatter(r_subset, phi_subset, h_subset, s=1)
    ax2.set_title('Original Data: Cylindrical coordinates')
    ax2.set_xlabel('R')
    ax2.set_ylabel('Phi')
    ax2.set_zlabel('H')

    # 3. Grid in Cartesian coordinates
    ax3 = fig.add_subplot(223, projection='3d')
    # Create indices for the grid plotting 
    grid_indices = np.random.choice(len(X.flatten()), size=min(n_to_plot, len(X.flatten())), replace=False)
    ax3.scatter(X.flatten()[grid_indices], Y.flatten()[grid_indices], Z.flatten()[grid_indices], s=1, c=np.arange(len(grid_indices)), cmap='viridis')
    ax3.set_title('Grid: Cartesian coordinates')
    ax3.set_xlabel('X')
    ax3.set_ylabel('Y')
    ax3.set_zlabel('Z')
    set_axes_equal(ax3)

    # 4. Grid in Cylindrical coordinates
    ax4 = fig.add_subplot(224, projection='3d')
    # Use same grid indices for cylindrical coordinates  
    ax4.scatter(R.flatten()[grid_indices], PHI.flatten()[grid_indices], H.flatten()[grid_indices], s=1)
    ax4.set_title('Grid: Cylindrical coordinates')
    ax4.set_xlabel('R')
    ax4.set_ylabel('Phi')
    ax4.set_zlabel('H')

    plt.tight_layout()
    plt.show()
    fig.savefig('all_cartesian_cylindrical_grids.png', dpi=300)
    plt.close(fig)
    print('Plot saved to all_cartesian_cylindrical_grids.png')
    
# plotting()

# save the grid points to a .npy file. Include both the flattened coordinates and the original axes
grid_data = {
    'X_grid': X.flatten(), 
    'Y_grid': Y.flatten(), 
    'Z_grid': Z.flatten(),
    'r_axis': r_grid,      # Original r axis
    'phi_axis': phi_grid,  # Original phi axis  
    'h_axis': h_grid       # Original h axis (with arithmetic progression)
}
np.save('cylindrical_grid.npy', grid_data)
print('Grid data saved to cylindrical_grid.npy with axes included')