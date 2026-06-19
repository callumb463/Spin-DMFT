#include <iostream>
#include <vector>
#include <complex>
#include <stdexcept>
#include <Eigen/Dense>
#include <unsupported/Eigen/MatrixFunctions>
#include <random>
#include <cassert>
#include <thread>
using cd = std::complex<double>;

static const Eigen::Matrix2cd Sx = (Eigen::Matrix2cd() <<
    cd(0.0, 0.0), cd(0.5, 0.0),
    cd(0.5, 0.0), cd(0.0, 0.0)
).finished();
static const Eigen::Matrix2cd Sy = (Eigen::Matrix2cd() <<
    cd(0.0, 0.0), cd(0.0, -0.5),
    cd(0.0, 0.5), cd(0.0, 0.0)
).finished();
static const Eigen::Matrix2cd Sz = (Eigen::Matrix2cd() <<
    cd(0.5, 0.0),  cd(0.0, 0.0),
    cd(0.0, 0.0), cd(-0.5, 0.0)
).finished();

template <typename MatrixT>
class MatrixSequence {
public:
    MatrixSequence(std::size_t count, const MatrixT& fillValue=MatrixT::Zero())
        : blocks_(count, fillValue) {}

    MatrixT& operator[](std::size_t idx) { return blocks_[idx]; }
    const MatrixT& operator[](std::size_t idx) const { return blocks_[idx]; }

    std::size_t size() const { return blocks_.size(); }

private:
    std::vector<MatrixT> blocks_;
};

class GaussianSampler {
public:
    explicit GaussianSampler(unsigned int seed)
        : rng_(seed), dist_(0.0, 1.0) {}

    double draw(double stdev) { return dist_(rng_) * stdev; }

private:
    std::mt19937 rng_;
    std::normal_distribution<double> dist_;
};

//Hamiltonian
Eigen::Matrix2cd hamiltonianEval(const Eigen::Matrix3Xd& trajectory, const int index){
    static const int gamma = 1, B = 1;

    Eigen::Matrix2cd H = Sx*trajectory(0,index) + Sy*trajectory(1,index) + Sz*trajectory(2,index) + gamma*B*Sz;
    return H;
}

//Matrix buildCovarMatrix()
Eigen::MatrixXd buildCovarMatrix(const MatrixSequence<Eigen::Matrix3d>& meanFieldMoments, int L){
    Eigen::MatrixXd covarMatrix(3*(L+1), 3*(L+1));
    for (int alpha=0; alpha<3; ++alpha){
        for (int beta=0; beta<3; ++beta){
            for (int t1=0; t1<=L; ++t1){
                for (int t2=0; t2<=L; ++t2){
                    int row = alpha*(L+1) + t1;
                    int col = beta*(L+1) + t2;
                    int dl = std::abs(t1 - t2);
                    double value = (t1 >= t2) ? meanFieldMoments[dl](alpha, beta)
                                            : meanFieldMoments[dl](beta, alpha);
                    covarMatrix(row, col) = value;
                }
            }
        }
    }
    return covarMatrix;
}

//Type is a placeholder maybe may change to another format but this should be fine I think
Eigen::Matrix3Xd sample(const Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd>& solver, GaussianSampler& sampler) {
    const Eigen::MatrixXd& O = solver.eigenvectors();
    int dim = O.rows();
    assert(dim % 3 == 0 && "covariance matrix dimension must be a multiple of 3");
    int L = dim / 3 - 1;

    Eigen::VectorXd sigmas = solver.eigenvalues().cwiseMax(0.0).cwiseSqrt();
    Eigen::VectorXd R(dim);
    for (int k = 0; k < dim; ++k)
        R[k] = sampler.draw(sigmas[k]);

    Eigen::VectorXd Vflat = O * R;   // row-major: [x(t0..tL), y(t0..tL), z(t0..tL)]
    Eigen::Map<const Eigen::Matrix<double, 3, Eigen::Dynamic, Eigen::RowMajor>>
        view(Vflat.data(), 3, L + 1);

    Eigen::Matrix3Xd trajectory = view;   // copy out of the Map into storage
    return trajectory;
}


MatrixSequence<Eigen::Matrix2cd> computePropagator(Eigen::Matrix3Xd trajectory, std::size_t L, double dt){
    MatrixSequence<Eigen::Matrix2cd> U(L+1);
    Eigen::Matrix2cd newH;
    Eigen::Matrix2cd oldH = hamiltonianEval(trajectory,0);
    Eigen::Matrix2cd A;
    U[0] = Eigen::Matrix2cd::Identity();
    for (int j=1; j<=L; j++){
        double sign = (j % 2 == 0) ? 1.0 : -1.0;
        double coefficient = (2*j-1)*(dt/2.0);
        newH = hamiltonianEval(trajectory, j);
        A = cd(0.0,coefficient)*(newH-(sign*oldH));
        U[j] = A.exp()*U[j-1];
    }
    return U;
}

double computeCorrelation(const Eigen::Matrix2cd& Ut, const Eigen::Matrix2cd& Sa, const Eigen::Matrix2cd& Sb){
    cd trace = (Ut.adjoint() * Sa * Ut * Sb).trace();
    return trace.real()*0.5;
}

Eigen::VectorXd correlationMatrix(const MatrixSequence<Eigen::Matrix2cd>& U, std::size_t L){
    Eigen::VectorXd  correlation(3*(L+1));
    for (std::size_t i=0; i<=L; ++i) {
            double gxx = computeCorrelation(U[i], Sx, Sx);
            double gxy = computeCorrelation(U[i], Sx, Sy);
            double gzz = computeCorrelation(U[i], Sz, Sz);
            correlation[3*i] = gxx;
            correlation[3*i + 1] = gxy;
            correlation[3*i + 2] = gzz;
    }
    return correlation;
}

//??? average
//I'll implement this once I get multithreading going
MatrixSequence<Eigen::Matrix3d> constructMeanFieldMoments(Eigen::VectorXd correlation, std::size_t L){
    static const double J2 = 1;
    MatrixSequence<Eigen::Matrix3d> meanFieldMoments(L+1);
    for (std::size_t i=0; i<=L; ++i){
        Eigen::Matrix3d formattedMoment;
        formattedMoment << correlation[3*i], correlation[3*i+1],       0,
                         -1*correlation[3*i+1], correlation[3*i],       0,
                                    0,                  0,      correlation[3*i+2];
        meanFieldMoments[i] = J2*J2 * formattedMoment;
    }
}
//MeanFieldMoments selfConsistency()

int main() {
    //Initialize
    //NEED TO CHANGE THESE BEFORE RUNNING
    std::size_t L=1;
    double dt = 0.01;
    const int numTraj = 100000;
    int iterations = 5;
    int threads = 0;

    unsigned int numThreads = std::thread::hardware_concurrency();
    if (numThreads == 0) numThreads = 4;

    //MAKE RANDOM MEAN FIELD MOMENTS
    MatrixSequence<Eigen::Matrix3d> meanFieldMoments(L+1, Eigen::Matrix3d::Zero());

    for (int n=0; n<iterations; n++) {
        //Make Covariance Matrix
        Eigen::MatrixXd M = buildCovarMatrix(meanFieldMoments, L);

        //Diagonalize Covariance Matrix
        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(M);
        if (solver.info() != Eigen::Success) {
            throw std::runtime_error("Eigen decomposition failed");
        }
        
        std::vector<Eigen::VectorXd> threadSums(numThreads, Eigen::VectorXd::Zero(3*(L+1)));
        std::vector<std::thread> threads;

        int baseCount = numTraj / numThreads;
        int remainder = numTraj % numThreads;

        for (unsigned int t=0; t<numThreads; ++t) {
            int countForThread = baseCount + (t < remainder ? 1 : 0);
            threads.emplace_back([&, t, countForThread](){
                GaussianSampler localSampler(42 + n*1000 + t);
                Eigen::VectorXd localSum = Eigen::VectorXd::Zero(3*(L+1));

                for (int i=0; i<countForThread; ++i) {
                    Eigen::Matrix3Xd trajectory = sample(solver, localSampler);
                    MatrixSequence<Eigen::Matrix2cd> U = computePropagator(trajectory, L, dt);
                    Eigen::VectorXd corr = correlationMatrix(U, L);
                    localSum += corr;
                }
                threadSums[t] = localSum; 
            });
        }
        for (auto& th : threads) th.join();

        Eigen::VectorXd totalSum = Eigen::VectorXd::Zero(3*(L+1));
        for (const auto& s : threadSums) totalSum += s;
        Eigen::VectorXd avgCorrelation = totalSum / static_cast<double>(numTraj);

        MatrixSequence<Eigen::Matrix3d> newMoments = constructMeanFieldMoments(avgCorrelation, L);

    }
}