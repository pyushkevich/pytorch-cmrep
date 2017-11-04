import torch
from torch.autograd import Variable

class PointSetHamiltonianSystem:

    def __init__(self, q0, sigma, N):
        """
        Geodesic shooting for point sets implemented with PyTorch
        Parameters:
            q0:      M x 3 tensor of point coordinates
            sigma:   Standard deviation of smoothing kernel
            N:       Number of time steps
        """

        # Store the parameters
        self.q0, self.sigma, self.N = q0, sigma, N

        # Allocate the intermediate tensors
        self.m, self.d = q0.size()
        self.D = torch.zeros(self.m, self.m)
        self.G = torch.zeros(self.m, self.m)
        self.P = torch.zeros(self.m, self.m)

        # Allocate the output tensors
        self.qt = list()
        self.pt = list()
        for t in range(self.N):
            self.qt.append(torch.zeros(self.m, self.d))
            self.pt.append(torch.zeros(self.m, self.d))

    def dist2_mat(self,q):
        B = torch.mm(q, q.t())
        A = torch.sum(q**2,1).expand_as(B)
        return A+A.t()-2*B

    def hamiltonian_jet(self, q, p):
        """
        Compute the kinetic energy and its derivatives in position and momentum
        Parameters:
            q:      M x 2 tensor of coordinates
            p:      M x 2 tensor of momenta
        """

        # Gaussian factor
        f = -0.5 / self.sigma ** 2

        # Compute the distance matrix between elements of q
        A = torch.sum(q**2,1).expand_as(self.D)
        torch.addmm(f, A+A.t(), -2 * f, q, q.t(), out=self.D)
        self.D.exp_() 

        # At this time, D holds the gaussian matrix
        Hp = self.D.mm(p)

        # Now multiply by pi_pj outer product
        torch.mm(p, p.t(), out = self.P)
        torch.mul(self.D, self.P, out = self.D)

        # Hamiltonian value
        H = 0.5 * self.D.sum()

        # Hq
        z = sum(self.D, 0).unsqueeze(1).expand_as(q)
        Hq = 2 * f * (z * q - self.D.mm(q))

        return H, Hq, Hp

    def flow(self, p0):

        # Initialize the flow
        self.qt[0] = self.q0
        self.pt[0] = p0

        # Compute the hamiltonian and its derivatives
        dt = 1.0 / (self.N - 1)
        for t in range(self.N):
            H,Hq,Hp = self.hamiltonian_jet(self.qt[-1], self.pt[-1])
            self.qt[t] = self.qt[t-1] + dt * Hp
            self.pt[t] = self.pt[t-1] - dt * Hq

        # Return final state
        return (self.qt[-1], self.pt[-1])
        
        



