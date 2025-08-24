#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <numeric> // For std::accumulate
# include <iomanip>
#include <Eigen/Dense>

// M_PI定数を使用するために必要
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// using namespace宣言
using namespace std;
using namespace Eigen;

int main() {
    // ... set calculation conditions.
    const int n_batches = 1000;
    const int n_particles = 10000;
    const int n_skip = 20;

    // ... set sphere radius.
    const double radius = 8.0;

    // ... set group constants.
    const int NG = 1;   // number of energy groups
    const int NMat = 1; // number of materials
    MatrixXd SigT(NMat, NG);
    MatrixXd SigA(NMat, NG);
    MatrixXd SigP(NMat, NG);
    MatrixXd SigF(NMat, NG);
    vector<MatrixXd> SigS(NMat, MatrixXd(NG, NG));
    MatrixXd Chi(NMat, NG);
    MatrixXd D(NMat, NG);
    MatrixXd NuTot(NMat, NG);

    SigA(0, 0) = 0.5;
    SigP(0, 0) = 0.6;
    NuTot(0, 0) = 2.5;
    SigF = SigP.array() / NuTot.array();
    Chi(0, 0) = 1.0;
    SigS[0](0, 0) = 0.5;
    SigT(0, 0) = SigS[0].sum() + SigA(0, 0);
    D(0, 0) = 1.0 / (3.0 * SigT(0, 0));

    // ... prepare probabilities.
    MatrixXd prob_fis = SigF.array() / SigA.array();
    MatrixXd prob_abs = SigA.array() / SigT.array();

    // ... C++11 random number generator
    mt19937 gen(1);
    uniform_real_distribution<double> dist_uni(0.0, 1.0);

    // ... prepare containers for bank.
    vector<Vector4d> bank;       // unnormalized bank
    vector<Vector4d> bank_init;  // normalized bank (initial bank to start each batch)

    // ... prepare containers for tally.
    vector<Vector2d> tally_k;

    // ... prepare initial source.
    for (int i = 0; i < n_particles; ++i) {
        double rrr = radius * pow(dist_uni(gen), 1.0 / 3.0);
        double costh = 2.0 * dist_uni(gen) - 1.0;
        double phi = 2.0 * M_PI * dist_uni(gen);
        double sinth = sqrt(1.0 - costh * costh);
        double x0 = rrr * sinth * cos(phi);
        double y0 = rrr * sinth * sin(phi);
        double z0 = rrr * costh;
        bank_init.push_back(Vector4d(x0, y0, z0, 1.0));
    }

    // ... start loop for batch.
    for (int j = 0; j < n_batches; ++j) {
        cout << "batch " << j << flush;

        // ... normalize fission source.
        if (j != 0) {
            for (const auto& p_info : bank) {
                double weight = p_info(3);
                while (weight >= 1.0) {
                    bank_init.push_back(Vector4d(p_info(0), p_info(1), p_info(2), 1.0));
                    weight -= 1.0;
                }
                if (weight >= dist_uni(gen)) {
                    bank_init.push_back(Vector4d(p_info(0), p_info(1), p_info(2), 1.0));
                }
            }

            if (bank_init.empty()) {
                cout << "\nXXX No fission neutrons." << endl;
                exit(1);
            }

            // Adjust particle count to n_particles
            size_t current_size = bank_init.size();
            if (current_size < n_particles) {
                for (size_t i = 0; i < n_particles - current_size; ++i) {
                    bank_init.push_back(bank_init[i % current_size]);
                }
            }
            bank_init.resize(n_particles);
        }
        bank.clear();

        // ... start each particle.
        for (int i = 0; i < n_particles; ++i) {
            double x0 = bank_init[i](0);
            double y0 = bank_init[i](1);
            double z0 = bank_init[i](2);

            // ... determine initial flight direction.
            double costh = 2.0 * dist_uni(gen) - 1.0;
            double phi = 2.0 * M_PI * dist_uni(gen);
            double sinth = sqrt(1.0 - costh * costh);
            double u = sinth * cos(phi);
            double v = sinth * sin(phi);
            double w = costh;

            int grp = 0; // ... determine initial energy group.

            // ... loop for collisions.
            while (true) {
                // ... determine flight distance.
                double xstot = SigT(0, grp);
                double dist = -log(dist_uni(gen)) / xstot;

                // ... calculate position after flight & airline distance.
                double x1 = x0 + dist * u;
                double y1 = y0 + dist * v;
                double z1 = z0 + dist * w;
                double airl_dist = sqrt(x1 * x1 + y1 * y1 + z1 * z1);

                // ... check whether leaks or not.
                if (airl_dist > radius) {
                    break;  // ... leaked.
                }

                // ... analyze collision.
                if (dist_uni(gen) < prob_abs(0, grp)) {
                    if (dist_uni(gen) < prob_fis(0, grp)) {
                        // ... store fission neutron info.
                        bank.push_back(Vector4d(x1, y1, z1, NuTot(0, grp)));
                    }
                    break;  // ... absorbed.
                }

                // ... scattering: determine new flight direction.
                costh = 2.0 * dist_uni(gen) - 1.0;
                phi = 2.0 * M_PI * dist_uni(gen);
                sinth = sqrt(1.0 - costh * costh);
                u = sinth * cos(phi);
                v = sinth * sin(phi);
                w = costh;

                // ... update position.
                x0 = x1; y0 = y1; z0 = z1;
            }
        }

        // ... tally k-score.
        double sum_fis_neu = 0.0;
        for (const auto& p_info : bank) {
            sum_fis_neu += p_info(3);
        }
        Vector2d counter_k;
        counter_k(0) = sum_fis_neu / static_cast<double>(n_particles);
        counter_k(1) = counter_k(0) * counter_k(0);
        cout << "  k = " << fixed << setprecision(5) << counter_k(0) << endl;
        tally_k.push_back(counter_k);

        // ... clear fission bank for next batch.
        bank_init.clear();
    }

    // ... process statistics.
    tally_k.erase(tally_k.begin(), tally_k.begin() + n_skip);
    double n_active = static_cast<double>(n_batches - n_skip);

    Vector2d sum_vec = accumulate(tally_k.begin(), tally_k.end(), Vector2d::Zero().eval());

    double k_average = sum_vec(0) / n_active;
    double k2_average = sum_vec(1) / n_active;
    double k_variance = (k2_average - k_average * k_average) / n_active;
    double k_stdev = sqrt(k_variance);
    double k_fsd = k_stdev / k_average;

    // ... output results.
    cout << "\n***** final results *****" << endl;
    long n_total = static_cast<long>(n_particles) * n_batches;
    cout << "Number of total histories : " << n_total << endl;
    cout << "k value              : " << fixed << setprecision(5) << k_average << endl;
    cout << "standard deviation   : " << fixed << setprecision(5) << k_stdev << endl;
    cout << "fractional std. dev. : " << fixed << setprecision(5) << k_fsd << endl;

    return 0;
}