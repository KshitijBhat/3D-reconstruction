import numpy as np
from sympy import symbols, solve
import matplotlib.pyplot as plt
from utils import displayEpipolarF, toHomogenous, refineF, _singularize, calc_epi_error

def eightpoint(pts1, pts2, M):
    """
    Input:  pts1, Nx2 Matrix
            pts2, Nx2 Matrix
            M, a scalar parameter computed as max (imwidth, imheight)
    Output: F, the fundamental matrix
    """

    # Normalize the input pts1 and pts2 using the matrix T.
    T = np.diag([1./M, 1./M])
    pts1 = pts1 @ T
    pts2 = pts2 @ T
    x_dash = pts1[:, 0]
    y_dash = pts1[:, 1]
    x = pts2[:, 0]
    y = pts2[:, 1]

    # Setup the eight point algorithm's equation.
    A = np.vstack((x * x_dash, x * y_dash, x, y * x_dash,  y * y_dash, y, x_dash, y_dash, np.ones_like(x))).T
    
    # Solve for the least square solution using SVD.
    U, S, Vt = np.linalg.svd(A)
    F = Vt[-1].reshape(3, 3)

    # Use the function `_singularize` (provided) to enforce the singularity condition.
    F = _singularize(F)

    # Use the function `refineF` (provided) to refine the computed fundamental matrix.
    F = refineF(F, pts1, pts2)

    # Unscale the fundamental matrix
    T = np.diag([1./M, 1./M, 1])
    F = ((T.T @ F) @ T)/F[2, 2]

    return F

def fivepoint(pts1, pts2, M, K1, K2):
    """
    Input:  pts1, Nx2 Matrix
            pts2, Nx2 Matrix
            M, a scalar parameter computed as max (imwidth, imheight)
    Output: F, the fundamental matrix
    """

    # Normalize the input pts1 and pts2 using the matrix T.
    T = np.diag([1./M, 1./M])
    pts1 = pts1 @ T
    pts2 = pts2 @ T
    x_dash = pts1[:, 0]
    y_dash = pts1[:, 1]
    x = pts2[:, 0]
    y = pts2[:, 1]

    # Setup the eight point algorithm's equation.
    A = np.vstack((x * x_dash, x * y_dash, x, y * x_dash,  y * y_dash, y, x_dash, y_dash, np.ones_like(x))).T
    
    # Solve for the least square solution using SVD.
    U, S, Vt = np.linalg.svd(A)
    F = Vt[-1].reshape(3, 3)

    # Use the function `_singularize` (provided) to enforce the singularity condition.
    F = _singularize(F)

    # Use the function `refineF` (provided) to refine the computed fundamental matrix.
    F = refineF(F, pts1, pts2)

    # Unscale the fundamental matrix
    T = np.diag([1./M, 1./M, 1])
    F = ((T.T @ F) @ T)/F[2, 2]

    return F

def sevenpoint(pts1, pts2, M):
    """
    Input:  pts1, Nx2 Matrix
            pts2, Nx2 Matrix
            M, a scalar parameter computed as max (imwidth, imheight)
    Output: F, the fundamental matrix
    """

    # Normalize the input pts1 and pts2 scale paramter M.
    pts1 = pts1/M
    pts2 = pts2/M
    x_dash = pts1[:, 0]
    y_dash = pts1[:, 1]
    x = pts2[:, 0]
    y = pts2[:, 1]

    # Setup the seven point algorithm's equation.
    A = np.vstack((x * x_dash, x * y_dash, x, y * x_dash,  y * y_dash, y, x_dash, y_dash, np.ones_like(x))).T
    
    # Solve for the least square solution using SVD.
    U, S, Vt = np.linalg.svd(A)

    # Last two columns of Vt
    f1, f2 = Vt[-1].reshape(3, 3), Vt[-2].reshape(3, 3)

    # Use the singularity constraint to solve for the cubic polynomial equation
    def det_constraint(a):
        # Compute the determinant of the matrix
        return np.linalg.det(a*f1 + (1-a)*f2)

    coeff0 = det_constraint(0)
    coeff1 = 2*(det_constraint(1)-det_constraint(-1))/3 - (det_constraint(2)-det_constraint(-2))/12 
    coeff2 = (det_constraint(1)+det_constraint(-1))/2.0 - coeff0
    coeff3 = (det_constraint(1)+det_constraint(-1))/2.0 - coeff1

    polynomial = np.array([coeff3,coeff2,coeff1,coeff0])
    solutions = np.roots(polynomial)

    print("Solutions:", solutions)

    # Use sympy to solve the cubic polynomial equation
    a = symbols('a')
    polynomial = coeff3 * a**3 + coeff2 * a**2 + coeff1 * a + coeff0
    solutionsq = solve(polynomial, a)
    print("Solutions:", solutionsq)
    # Discard solutions with very small imaginary part
    solutions = [complex(sol).real for sol in solutionsq if abs(complex(sol).imag) < 1e-12]

    Farray = []
    # Unscale the fundamental matrixes and return as Farray
    T = np.diag([1./M, 1./M, 1])
    for sol in solutions:
        if isinstance(sol, complex):
            continue
        F = sol*f1 + (1-sol)*f2
        F = refineF(F, pts1, pts2)
        F = ((T.T @ F) @ T)/F[2, 2]
        Farray.append(F)

    return Farray

if __name__ == "__main__":
    correspondence = np.load("data/some_corresp.npz")  # Loading correspondences
    intrinsics = np.load("data/intrinsics.npz")  # Loading the intrinscis of the camera
    K1, K2 = intrinsics["K1"], intrinsics["K2"]
    pts1, pts2 = correspondence["pts1"], correspondence["pts2"]
    im1 = plt.imread("data/im1.png")
    im2 = plt.imread("data/im2.png")

    # F = eightpoint(pts1, pts2, M=np.max([*im1.shape, *im2.shape]))
    # print(F)

    # displayEpipolarF(im1, im2, F)


    # # Simple Tests to verify your implementation:
    # pts1_homogenous, pts2_homogenous = toHomogenous(pts1), toHomogenous(pts2)

    indices = np.array([82, 19, 56, 84, 54, 24, 18])

    M = np.max([*im1.shape, *im2.shape])

    Farray = sevenpoint(pts1[indices, :], pts2[indices, :], M)

    print(Farray)

    F = Farray[0]
    print(F)

    # fundamental matrix must have rank 2!
    # assert(np.linalg.matrix_rank(F) == 2)
    displayEpipolarF(im1, im2, F)

    # Simple Tests to verify your implementation:
    # Test out the seven-point algorithm by randomly sampling 7 points and finding the best solution.
    np.random.seed(1)  # Added for testing, can be commented out

    pts1_homogenous, pts2_homogenous = toHomogenous(pts1), toHomogenous(pts2)

    max_iter = 500
    pts1_homo = np.hstack((pts1, np.ones((pts1.shape[0], 1))))
    pts2_homo = np.hstack((pts2, np.ones((pts2.shape[0], 1))))

    ress = []
    F_res = []
    choices = []
    M = np.max([*im1.shape, *im2.shape])
    for i in range(max_iter):
        choice = np.random.choice(range(pts1.shape[0]), 7)
        pts1_choice = pts1[choice, :]
        pts2_choice = pts2[choice, :]
        Fs = sevenpoint(pts1_choice, pts2_choice, M)
        for F in Fs:
            choices.append(choice)
            res = calc_epi_error(pts1_homo, pts2_homo, F)
            F_res.append(F)
            ress.append(np.mean(res))

    min_idx = np.argmin(np.abs(np.array(ress)))
    F = F_res[min_idx]
    print("Error:", ress[min_idx])

    assert F.shape == (3, 3)
    assert F[2, 2] == 1
    assert np.linalg.matrix_rank(F) == 2
    assert np.mean(calc_epi_error(pts1_homogenous, pts2_homogenous, F)) < 1