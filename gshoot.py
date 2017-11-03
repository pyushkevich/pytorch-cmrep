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

    def dist2_mat(self,q):
        A = torch.sum(q**2,1).repeat(q.size(0),1)
        B = torch.mm(q, q.t())
        return A+A.t()-2*B

    def hamiltonian_jet(self, q, p):
        """
        Compute the kinetic energy and its derivatives in position and momentum
        Parameters:
            q:      M x 2 tensor of coordinates
            p:      M x 2 tensor of momenta
        """

        # Number of points
        m,d = q.size()

        # Gaussian factor
        f = -0.5 / self.sigma ** 2

        # Compute the distance matrix between elements of q
        pi_pj = torch.mm(p, p.t())

        # Hamiltonian value
        dm = self.dist2_mat(q)
        G = torch.exp(f * dm)
        pi_pj_gqq = pi_pj * G
        H = 0.5 * pi_pj_gqq.sum()

        # Hp
        Hp = torch.mm(G, p)

        # Hq
        Z=2 * f * pi_pj_gqq;
        Hq = q * sum(Z,0).view(-1,1).repeat(1,d) - torch.mm(Z,q)

        return H, Hq, Hp

    def flow(self, p0):

        # Initialize the flow
        self.qt = list()
        self.pt = list()
        
        self.qt.append(self.q0)
        self.pt.append(p0)

        # Compute the hamiltonian and its derivatives
        dt = 1.0 / (self.N - 1)
        for t in range(self.N):
            H,Hq,Hp = self.hamiltonian_jet(self.qt[-1], self.pt[-1])
            self.qt.append(self.qt[-1] + dt * Hp)
            self.pt.append(self.pt[-1] - dt * Hq)

        # Return final state
        return (self.qt[-1], self.pt[-1])
        
        



