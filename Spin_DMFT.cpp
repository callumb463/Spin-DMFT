#include <iostream>
#include <fstream>
#include <vector>
#include <complex>
#include <stdexcept>
#include <cmath>
#include <Eigen/Dense>
#include <unsupported/Eigen/MatrixFunctions>
#include <random>
#include <cassert>
#include <thread>
#include <chrono>
using cd = std::complex<double>;
using Clock = std::chrono::steady_clock;

static double elapsed(Clock::time_point start) {
    return std::chrono::duration<double>(Clock::now() - start).count();
}

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

// Hamiltonian
Eigen::Matrix2cd hamiltonianEval(const Eigen::Matrix3Xd& trajectory, const int index, double gammaB){
    Eigen::Matrix2cd H = Sx*trajectory(0,index) + Sy*trajectory(1,index) + Sz*trajectory(2,index) + gammaB*Sz;
    return H;
}

// Initial Mean Field Moments — exponential decay with rate J2
MatrixSequence<Eigen::Matrix3d> initialMeanFields(const double J2, const std::size_t L, const double dt){
    MatrixSequence<Eigen::Matrix3d> initialMFM(L+1);
    for (int dl = 0; dl < L+1; ++dl){
        for (int alpha = 0; alpha < 3; ++alpha){
            initialMFM[dl](alpha, alpha) = J2 * std::exp(-J2 * dl * dt);  // FIX: negative exponent
        }
    }
    return initialMFM;
}

// Generate Covariance Matrix from Moments
Eigen::MatrixXd buildCovarMatrix(const MatrixSequence<Eigen::Matrix3d>& meanFieldMoments, const std::size_t L){
    Eigen::MatrixXd covarMatrix(3*(L+1), 3*(L+1));
    for (int alpha = 0; alpha < 3; ++alpha){
        for (int beta = 0; beta < 3; ++beta){
            for (int t1 = 0; t1 <= (int)L; ++t1){
                for (int t2 = 0; t2 <= (int)L; ++t2){
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

// B = eigenvectors * diag(sqrt(eigenvalues)), precomputed once per DMFT iteration
Eigen::MatrixXd buildSampleMatrix(const Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd>& solver) {
    Eigen::VectorXd sigmas = solver.eigenvalues().cwiseMax(0.0).cwiseSqrt();
    return solver.eigenvectors() * sigmas.asDiagonal();
}

Eigen::Matrix3Xd sample(const Eigen::MatrixXd& B, int L, GaussianSampler& sampler) {
    int dim = B.rows();
    Eigen::VectorXd z(dim);
    for (int k = 0; k < dim; ++k)
        z[k] = sampler.draw(1.0);

    Eigen::VectorXd Vflat = B * z;
    Eigen::Map<const Eigen::Matrix<double, 3, Eigen::Dynamic, Eigen::RowMajor>>
        view(Vflat.data(), 3, L + 1);

    return Eigen::Matrix3Xd(view);
}

// Analytic exp for traceless skew-Hermitian 2x2: exp(A) = cos(θ)I + sinc(θ)A, θ = sqrt(-det(A))
inline Eigen::Matrix2cd matexp2(const Eigen::Matrix2cd& A) {
    double theta = std::sqrt(std::max(0.0, -(A(0,0)*A(1,1) - A(0,1)*A(1,0)).real()));
    double c = std::cos(theta);
    double s = (theta > 1e-10) ? std::sin(theta) / theta : 1.0 - theta*theta / 6.0;
    return c * Eigen::Matrix2cd::Identity() + s * A;
}

// 2nd-order CFET propagator — fills pre-allocated buffer U
void computePropagator(const Eigen::Matrix3Xd& trajectory, std::size_t L, double dt, double gammaB,
                       MatrixSequence<Eigen::Matrix2cd>& U){
    Eigen::Matrix2cd oldH = hamiltonianEval(trajectory, 0, gammaB);
    Eigen::Matrix2cd newH;
    U[0] = Eigen::Matrix2cd::Identity();
    for (std::size_t j = 1; j <= L; j++){
        newH = hamiltonianEval(trajectory, j, gammaB);
        Eigen::Matrix2cd A = cd(0.0, -dt / 2.0) * (newH + oldH);
        U[j] = matexp2(A) * U[j-1];
        oldH = newH;
    }
}

double computeCorrelation(const Eigen::Matrix2cd& Ut, const Eigen::Matrix2cd& Sa, const Eigen::Matrix2cd& Sb){
    cd trace = (Ut.adjoint() * Sa * Ut * Sb).trace();
    return trace.real() * 0.5;
}

Eigen::VectorXd correlationMatrix(const MatrixSequence<Eigen::Matrix2cd>& U, std::size_t L){
    Eigen::VectorXd correlation(3*(L+1));
    for (std::size_t i = 0; i <= L; ++i) {
        double gxx = computeCorrelation(U[i], Sx, Sx);
        double gxy = computeCorrelation(U[i], Sx, Sy);
        double gzz = computeCorrelation(U[i], Sz, Sz);
        correlation[3*i]     = gxx;
        correlation[3*i + 1] = gxy;
        correlation[3*i + 2] = gzz;
    }
    return correlation;
}

MatrixSequence<Eigen::Matrix3d> constructMeanFieldMoments(Eigen::VectorXd correlation, std::size_t L, const double J2){
    MatrixSequence<Eigen::Matrix3d> meanFieldMoments(L+1);
    for (std::size_t i = 0; i <= L; ++i){
        Eigen::Matrix3d formattedMoment;
        formattedMoment <<  correlation[3*i],      correlation[3*i+1], 0,
                           -correlation[3*i+1],    correlation[3*i],   0,
                            0,                     0,                  correlation[3*i+2];
        meanFieldMoments[i] = J2*J2 * formattedMoment;
    }
    return meanFieldMoments;
}

int main() {
    double J2      = 1.0;   // sets energy/time scale
    double gammaB  = 0.0;   // 0 for Fig 2 (zero field), 5.0*J2 for Fig 3
    double dt      = 0.02;  // time step in units of 1/J2
    std::size_t L  = 300;   // t_max = L*dt = 6/J2
    const int numTraj  = 10000;
    int iterations = 5;

    unsigned int numThreads = std::thread::hardware_concurrency();
    if (numThreads == 0) numThreads = 4;

    MatrixSequence<Eigen::Matrix3d> meanFieldMoments = initialMeanFields(J2, L, dt);

    auto totalStart = Clock::now();
    for (int n = 0; n < iterations; n++) {
        auto iterStart = Clock::now();
        Eigen::MatrixXd M = buildCovarMatrix(meanFieldMoments, L);

        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(M);
        if (solver.info() != Eigen::Success) {
            throw std::runtime_error("Eigen decomposition failed");
        }

        Eigen::MatrixXd B = buildSampleMatrix(solver);

        std::vector<Eigen::VectorXd> threadSums(numThreads, Eigen::VectorXd::Zero(3*(L+1)));
        std::vector<std::thread> threads;

        int baseCount = numTraj / numThreads;
        int remainder = numTraj % numThreads;

        for (unsigned int t = 0; t < numThreads; ++t) {
            int countForThread = baseCount + (t < (unsigned int)remainder ? 1 : 0);
            threads.emplace_back([&, t, countForThread](){
                GaussianSampler localSampler(42 + n*1000 + t);
                Eigen::VectorXd localSum = Eigen::VectorXd::Zero(3*(L+1));
                MatrixSequence<Eigen::Matrix2cd> U(L+1);

                for (int i = 0; i < countForThread; ++i) {
                    Eigen::Matrix3Xd trajectory = sample(B, (int)L, localSampler);
                    computePropagator(trajectory, L, dt, gammaB, U);
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

        meanFieldMoments = constructMeanFieldMoments(avgCorrelation, L, J2);
        std::cout << "iteration " << n+1 << "/" << iterations
                  << "  (" << elapsed(iterStart) << " s)\n";
    }
    std::cout << "total: " << elapsed(totalStart) << " s\n";

    std::ofstream csv("correlations.csv");
    csv << "tJ2,gxx,gxy,gzz\n";  // tJ2 = t in units of 1/J2
    for (std::size_t i = 0; i <= L; ++i) {
        double tJ2 = i * dt * J2;
        double gxx = meanFieldMoments[i](0, 0) / (J2 * J2);
        double gxy = meanFieldMoments[i](0, 1) / (J2 * J2);
        double gzz = meanFieldMoments[i](2, 2) / (J2 * J2);
        csv << tJ2 << "," << gxx << "," << gxy << "," << gzz << "\n";
    }
    csv.close();
    std::cout << "Exported correlations.csv (" << L+1 << " rows)\n";
}