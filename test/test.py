import numpy as np
from cfdtool import Solve

def test_cfdSolveAlgebraicSystem():
    # Test case 1: Smoother = 'DILU'
    theCoefficients = {
        'ac': [1.0, 2.0, 3.0],
        'anb': [np.array([0.1, 0.2]), np.array([0.3, 0.4]), np.array([0.5, 0.6])],
        'bc': [0.5, 0.7, 0.9],
        'theCConn': [[1], [0, 2], [1]],
        'dphi': [0.0, 0.0, 0.0],
        'NumberOfElements': 3
    }
    maxIter = 20
    tolerance = 1e-6
    relTol = 0.1
    iComponent = -1

    initialResidual, finalResidual = Solve.cfdSolveAlgebraicSystem(1, 'Equation 1', theCoefficients, smoother='DILU', maxIter=maxIter, tolerance=tolerance, relTol=relTol, iComponent=iComponent)

    assert initialResidual == 0.0
    assert finalResidual == 0.0

    # Test case 2: Smoother = 'SOR'
    theCoefficients = {
        'ac': [1.0, 2.0, 3.0],
        'anb': [np.array([0.1, 0.2]), np.array([0.3, 0.4]), np.array([0.5, 0.6])],
        'bc': [0.5, 0.7, 0.9],
        'theCConn': [[1], [0, 2], [1]],
        'dphi': [0.0, 0.0, 0.0],
        'NumberOfElements': 3
    }
    maxIter = 20
    tolerance = 1e-6
    relTol = 0.1
    iComponent = -1

    initialResidual, finalResidual = Solve.cfdSolveAlgebraicSystem(1, 'Equation 1', theCoefficients, smoother='SOR', maxIter=maxIter, tolerance=tolerance, relTol=relTol, iComponent=iComponent)

    assert initialResidual == 0.0
    assert finalResidual == 0.0

    # Test case 3: Smoother = 'GaussSeidel'
    theCoefficients = {
        'ac': [1.0, 2.0, 3.0],
        'anb': [np.array([0.1, 0.2]), np.array([0.3, 0.4]), np.array([0.5, 0.6])],
        'bc': [0.5, 0.7, 0.9],
        'theCConn': [[1], [0, 2], [1]],
        'dphi': [0.0, 0.0, 0.0],
        'NumberOfElements': 3
    }
    maxIter = 20
    tolerance = 1e-6
    relTol = 0.1
    iComponent = -1

    initialResidual, finalResidual = Solve.cfdSolveAlgebraicSystem(1, 'Equation 1', theCoefficients, smoother='GaussSeidel', maxIter=maxIter, tolerance=tolerance, relTol=relTol, iComponent=iComponent)

    assert initialResidual == 0.0
    assert finalResidual == 0.0

test_cfdSolveAlgebraicSystem()