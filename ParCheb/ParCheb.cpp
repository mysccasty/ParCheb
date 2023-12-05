#include "ParCheb.h"
void generateMatr(double*& a0, double*& a1, double*& a2, double*& a3, double*& a4, double*& f, int nx, int ny) {
    for (int i = 0; i < nx * ny; i++) {
        a0[i] = 2.93333;
        if (i == 0 | i == nx - 1 | i == nx * (ny - 1) | i == nx * ny - 1) {
            f[i] = 0.133333;
        }
        else if ((i > 0 & i < nx - 1) | ((i + 1) % nx == 0) | (i % nx == 0) | (i > nx * (ny - 1) & i < nx * ny - 1)) {
            f[i] = 0.133333;
        }
        else {
            f[i] = 0;
        }
    }
    for (int i = 0; i < nx * ny - 1; i++) {
        if (((i + 1) % nx == 0) & (i != 0)) {
            a1[i] = 0;
            a2[i] = 0;

        }
        else {
            a1[i] = -0.0666667;
            a2[i] = -0.0666667;
        }
    }
    for (int i = 0; i < nx * (ny - 1); i++) {
        a3[i] = -0.0666667;
        a4[i] = -0.0666667;
    }
}
double norm(double* vector, int len) {
    int i;
    double sum = 0;
//#pragma omp parallel for private(i) reduction(+:sum)
    for (i = 0; i < len; i++) {
        sum += vector[i] * vector[i];
    }
    return sqrt(sum);
}


void nevyazka(double* a0, double* a1, double* a2, double* a3, double* a4, double* f, double* u, double*& R, int nx, int ny) {

    int i;
//#pragma omp parallel for shared(a0, a1, a2, a3, a4, f, u, R) private(i)
    for (i = 0; i < nx * ny; i++) {
        if (i == 0) {
            R[i] = f[i] - (a0[i] * u[i] + a1[i] * u[i + 1] + a3[i] * u[i + nx]);
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
    double* f, double* x, double tau) {

    double* r = (double*)malloc((nx * ny) * sizeof(double));
    double epsilon = 1e-15;
    double fNorm;
    for (int i = 0; i < nx * ny; i++) {
        r[i] = f[i];
    }
    fNorm = norm(f, nx * ny);
    for (int iter = 1; iter < N; iter++) {

        if (norm(r, nx * ny) < epsilon) {
            break;
        }
        for (int i = 0; i < nx * ny; i++) {
            x[i] += tau * r[i];
        }
        nevyazka(a0, a1, a2, a3, a4, f, x, r, nx, ny);


    }
}
void SLAE(double* s0, double* s1, double* s2, double* s3, double* s4, double* x, double* f, int nx, int ny) {
    int N = 1024;

    double* a0 = s0;
    double* a1 = s3;
    double* a2 = s1;
    double* a3 = s4;
    double* a4 = s2;
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
        if (lam_min > a0[i] - R) {
            lam_min = a0[i] - R;
        }
    }
    double tau = 2 / (lam_min + lam_max);
    omp_set_dynamic(0);
    omp_set_num_threads(12);
    cheb(nx, ny, N, a0, a1, a2, a3, a4, f, x, tau);
}
int main() {
    int nx = 2, ny = 2;
    double* a0 = (double*)malloc((nx * ny) * sizeof(double));
    double* a1 = (double*)malloc((nx * ny-1) * sizeof(double));
    double* a2 = (double*)malloc((nx * ny-1) * sizeof(double));
    double* a3 = (double*)malloc((nx * (ny-1)) * sizeof(double));
    double* a4 = (double*)malloc((nx * (ny-1)) * sizeof(double));
    double* f = (double*)malloc((nx * ny) * sizeof(double));
    generateMatr(a0, a1, a2, a3, a4, f, nx, ny);
    double* x = (double*)malloc((nx * ny) * sizeof(double));
    for (int i = 0; i < nx * ny; i++) {
        x[i] = 0;
    }
    std::ofstream result;
    result.open("result.txt");
    SLAE(a0, a2, a4, a1, a3, x, f, nx, ny);
    for (int i = 0; i < nx * ny; i++) {
        result << x[i] << endl;
    }
    result.close();

    
    /*for (int i = 0; i < nx * ny; i++) {
        if (i == 0) {
            cout << a0[i] * x[i] + a1[i] * x[i + 1] + a3[i] * x[i + nx] << "\t";
        }
        else if (i < nx) {
            cout << a2[i - 1] * x[i - 1] + a0[i] * x[i] + a1[i] * x[i + 1] + a3[i] * x[i + nx] << "\t";
        }
        else if (i == nx * ny - 1) {
            cout << a4[i - nx] * x[i - nx] + a2[i - 1] * x[i - 1] + a0[i] * x[i] << "\t";
        }
        else if (i >= nx * (ny - 1)) {
            cout << a4[i - nx] * x[i - nx] + a2[i - 1] * x[i - 1] + a0[i] * x[i] + a1[i] * x[i + 1] << "\t";

        }
        else {
            cout << a4[i - nx] * x[i - nx] + a2[i - 1] * x[i - 1] + a0[i] * x[i] + a1[i] * x[i + 1] + a3[i] * x[i + nx] << "\t";
        }
    }*/
    return 0;
}