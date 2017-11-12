import vtk
import numpy as np

# A mesh class
class Mesh(object):
    def __init__(self,**kwds):
        self.__dict__.update(kwds)

# A function to import a bcm-rep from a VTK file
def read_bcmrep(file):
    reader = vtk.vtkPolyDataReader()
    reader.SetFileName(file)
    reader.Update()
    pd = reader.GetOutput()
    n = pd.GetNumberOfPoints()
    
    # Read the point coordinates
    X=np.zeros((n,3))
    for i in range(n):
        for j in range(3):
            X[i,j] = pd.GetPoint(i)[j]
            
    # Read the cells
    m=pd.GetNumberOfCells()
    T=np.zeros((m,3),'int64')
    for i in range(m):
        for j in range(3):
            T[i,j] = pd.GetCell(i).GetPointId(j)
            
    # Read the medial index point array
    mi_arr = pd.GetPointData().GetArray('MedialIndex')
    MI=np.zeros((n), 'int64')
    for i in range(n):
        MI[i] = mi_arr.GetTuple1(i)

    # Create a triangle-wise medial index point array
    triMI = np.unique(np.sort(MI[T],1),axis=0,return_inverse=True)[1]

    # Create a result
    return Mesh(X=X,T=T,MI=MI,triMI=triMI)
