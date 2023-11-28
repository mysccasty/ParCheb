#include "ParCheb.h"

double norm(double* vector, int len) {
    int i;
    double sum = 0;
#pragma omp parallel for private(i) reduction(+:sum)
    for (i = 0; i < len; i++) {
        sum += vector[i] * vector[i];
    }
    return sqrt(sum);
}


void nevyazka(double* a0, double* a1, double* a2, double* a3, double* a4, double* f, double* u, double*& R, int nx, int ny) {

    int i;
#pragma omp parallel for shared(a0, a1, a2, a3, a4, f, u, R) private(i)
    for (i = 0; i < nx * ny; i++) {
        if (i == 0) {
            R[i] = f[i]-(a0[i] * u[i] + a1[i] * u[i + 1] + a3[i] * u[i + nx]);
        }
        else if (i < nx) {
            R[i] = f[i] - (a2[i - 1] * u[i - 1] + a0[i] * u[i] + a1[i] * u[i + 1] + a3[i] * u[i + nx]);
        }
        else if (i == nx * ny - 1) {
            R[i] = f[i] - (a4[i - nx] * u[i - nx] + a2[i - 1] * u[i - 1] + a0[i] * u[i]);
        }
        else if (i >= nx * (ny - 1)) {
            R[i] = f[i] - (a4[i - nx] * u[i - nx] + a2[i - 1] * u[i - 1] + a0[i] * u[i] + a1[i] * u[i + 1]);

        }
        else {
            R[i] = f[i] - (a4[i - nx] * u[i - nx] + a2[i - 1] * u[i - 1] + a0[i] * u[i] + a1[i] * u[i + 1] + a3[i] * u[i + nx]);
        }
    }
}
//Решение СЛАУ
void cheb(int nx, int ny, int N, double* a0, double* a1, double* a2, double* a3, double* a4,
    double* f, double tau, ofstream& out) {

    double* r = (double*)malloc((nx * ny) * sizeof(double));
    double* x = (double*)malloc((nx * ny) * sizeof(double));
    double epsilon = 1e-10;
    out << "Nx = " << nx << endl;
    out << "Ny = " << ny << endl;
    double fNorm;
    for (int i = 0; i < nx * ny; i++) {
        r[i] = f[i];
        x[i] = 0;
    }


    auto start = std::chrono::high_resolution_clock::now();
    fNorm = norm(f, nx * ny);
    nevyazka(a0, a1, a2, a3, a4, f, x, r, nx, ny);
    out << endl;
    for (int iter = 1; iter < N; iter++) {

        if (norm(r, nx * ny) < epsilon * fNorm) {
            out << iter << " iterations need to get " << epsilon << " error" << endl;
            break;
        }
        for (int i = 0; i < nx * ny; i++) {
            x[i] += tau * r[i];
        }
        nevyazka(a0, a1, a2, a3, a4, f, x, r, nx, ny);


    }
    for (int i = 0; i < nx * ny; i++) {
        //out << std::fixed;
        //out << std::setprecision(15);
        out << x[i] << "\t";
    }
    out << endl;
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration = end - start;
    out << "Time: " << duration.count() << " ms" << endl;

}
void SLAE(double* s0, double* s1, double* s2, double* s3, double* s4, double* f, int nx, int ny) {
    int N = 128;

    double* a0 = s0;
    double* a1 = s3;
    double* a2 = s1;
    double* a3 = s4;
    double* a4 = s2;

    double* a = (double*)malloc((N) * sizeof(double));
    double* b = (double*)malloc((N) * sizeof(double));
    double R = 0;

    double fNorm;
    double lam_max = -DBL_MAX;
    double lam_min = DBL_MAX;
    for (int i = 0; i < nx * ny; i++) {
        if (i == 0) {
            R = abs(a1[i]) + abs(a3[i]);
        }
        else if (i < nx) {
            R = abs(a1[i]) + abs(a2[i - 1]) + abs(a3[i]);
        }
        else if (i == (nx * ny - 1)) {
            R = abs(a2[i - 1]) + abs(a4[i - nx]);
        }
        else if (i >= nx * (ny - 1)) {
            R = abs(a1[i]) + abs(a2[i - 1]) + abs(a4[i - nx]);
        }
        else {
            R = abs(a1[i]) + abs(a2[i - 1]) + abs(a4[i - nx]) + abs(a3[i]);
        }
        if (lam_max < a0[i] + R) {
            lam_max = a0[i] + R;
        }
        if (lam_min > a0[i]-R) {
            lam_min = a0[i]-R;
        }
    }
    double tau = 2 / (lam_min + lam_max);
    omp_set_dynamic(0);
    omp_set_num_threads(12);
    std::ofstream result;
    result.open("result.txt");
    cheb(nx, ny, N, a0, a1, a2, a3, a4, f, tau, result);
    result.close();
}