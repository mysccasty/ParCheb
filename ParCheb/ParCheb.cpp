#include "ParCheb.h"
void generateMatr(double*& a0, double*& a1, double*& a2, double*& a3, double*& a4, double*& f, int nx, int ny) {
    for (int i = 0; i < nx * ny; i++) {
        a0[i] = 2.9282;
        if (i == 0 | i == nx - 1 | i == nx * (ny - 1) | i == nx * ny - 1) {
            f[i] = 4.8917e-09;
        }
        else if ((i > 0 & i < nx - 1) | ((i + 1) % nx == 0) | (i % nx == 0) | (i > nx * (ny - 1) & i < nx * ny - 1)) {
            f[i] = 2.44585e-09;
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
            a1[i] = 2.44585e-09;
            a2[i] = 2.44585e-09;
        }
    }
    for (int i = 0; i < nx * (ny - 1); i++) {
        a3[i] = 2.44585e-09;
        a4[i] = 2.44585e-09;
    }
}

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


void nevyazka(double* a0, double* a1, double* a2, double* a3, double* a4, double* f, double* u, double*& R, int nx, int ny, double alpha) {

    int i;
#pragma omp parallel for shared(a0, a1, a2, a3, a4, f, u, R) private(i)
    for (i = 0; i < nx * ny; i++) {
        if (i == 0) {
            R[i] -= alpha * (a0[i] * u[i] + a1[i] * u[i + 1] + a3[i] * u[i + nx]);
        }
        else if (i < nx) {
            R[i] -= alpha * (a2[i - 1] * u[i - 1] + a0[i] * u[i] + a1[i] * u[i + 1] + a3[i] * u[i + nx]);
        }
        else if (i == nx * ny - 1) {
            R[i] -= alpha * (a4[i - nx] * u[i - nx] + a2[i - 1] * u[i - 1] + a0[i] * u[i]);
        }
        else if (i >= nx * (ny - 1)) {
            R[i] -= alpha * (a4[i - nx] * u[i - nx] + a2[i - 1] * u[i - 1] + a0[i] * u[i] + a1[i] * u[i + 1]);

        }
        else {
            R[i] -= alpha * (a4[i - nx] * u[i - nx] + a2[i - 1] * u[i - 1] + a0[i] * u[i] + a1[i] * u[i + 1] + a3[i] * u[i + nx]);
        }
    }
}
//Решение СЛАУ
double* cheb(int nx, int ny, int N, double* a0, double* a1, double* a2, double* a3, double* a4,
    double* f, double* b, double* a, double* g) {

    double* p = (double*)malloc((nx * ny) * sizeof(double));
    double* r = (double*)malloc((nx * ny) * sizeof(double));
    double* x = (double*)malloc((nx * ny) * sizeof(double));
    double epsilon = 1e-7;
    double fNorm;
    for (int i = 0; i < nx * ny; i++) {
        r[i] = f[i];
        x[i] = 0;
    }


    auto start = std::chrono::high_resolution_clock::now();
    fNorm = norm(f, nx * ny);
    for (int i = 0; i < nx * ny; i++) {
        p[i] = r[i];
    }
    for (int iter = 1; iter < N; iter++) {

        if (norm(r, nx * ny) < epsilon * fNorm) {
            cout << iter << " iterations need to get " << epsilon << " error" << endl;
            break;
        }
        for (int i = 0; i < nx * ny; i++) {
            x[i] += a[iter - 1] * p[i];
        }
        nevyazka(a0, a1, a2, a3, a4, f, p, r, nx, ny, a[iter - 1]);
        for (int i = 0; i < nx * ny; i++) {
            p[i] = r[i] + b[iter] * p[i];
        }

    }
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration = end - start;
    cout << "Time: " << duration.count() << " ms" << endl;
    /*for (int i = 0; i < nx * ny; i++) {
        cout << x[i] << "\t";
    }*/
    return x;

}
double* SLAE(double* s0, double* s1, double* s2, double* s3, double* s4, double* f, int nx, int ny) {
    int N = 1024;
    double* a0 = s0;
    double* a1 = s3;
    double* a2 = s1;
    double* a3 = s4;
    double* a4 = s2;

    double* a = (double*)malloc((N) * sizeof(double));
    double* b = (double*)malloc((N) * sizeof(double));
    double* g = (double*)malloc((N) * sizeof(double));
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
        if (lam_min > a0[i]-R){
            lam_min = a0[i] - R;
        }
    }
    double gamma = (lam_max + lam_min) / (lam_max - lam_min);
    double d0 = 1 / gamma, d1 = 1 / (2 * gamma - d0);
    g[0] = 0;
    a[0] = 2 / (lam_max + lam_min);
    b[0] = 0;
    for (int i = 1; i < N; i++) {
        g[i] = d0 * d1;
        d0 = d1;
        d1 = 1 / (2 * gamma - d0);
        a[i] = 2 * (1 + g[i]) / (lam_max + lam_min);
        b[i] = g[i] * a[i - 1] * a[i];
    }


    omp_set_dynamic(0);
    omp_set_num_threads(12);
    double* x = cheb(nx, ny, N, a0, a1, a2, a3, a4, f, b, a, g);
    return x;
}
//Вычисление обратной
void inverse(double* s0, double* s1, double* s2, double* s3, double* s4, int nx, int ny) {

    double fNorm;
    double* f = (double*)malloc((nx * ny) * sizeof(double));
    double* x;



    double epsilon = 1e-7;
    double** AInv = (double**)malloc((nx * ny) * sizeof(double*));

    for (int i = 0; i < nx * ny; i++) {
        AInv[i] = (double*)malloc((nx * ny) * sizeof(double));
    }
    int k;
    auto start = std::chrono::high_resolution_clock::now();
//#pragma omp parallel for shared(AInv) private(k, x, f)
    for (k = 0; k < nx * ny; k++) {

        f = (double*)malloc((nx * ny) * sizeof(double));
        for (int i = 0; i < nx * ny; i++) {
            f[i] = 0;
        }
        f[k] = 1;

        x = SLAE(s0, s1, s2, s3, s4, f, nx, ny);
        for (int i = 0; i < nx * ny; i++) {
            AInv[k][i] = x[i];
        }
        free(f);
        free(x);

    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration = end - start;
    cout << "Time: " << duration.count() << " ms" << std::endl;
    for (int i = 0; i < nx * nx; i++) {
        for (int j = 0; j < nx * ny; j++) {
            cout << std::fixed;
            cout << std::setprecision(5);
            cout << AInv[i][j];
            cout << "\t";
        }
        cout << endl;
        free(AInv[i]);
    }
    free(AInv);
}
void main() {
    int nx = 3; 
    int ny = 3;
    double* s0 = (double*)malloc((nx * ny) * sizeof(double));
    double* s1 = (double*)malloc((nx * ny-1) * sizeof(double));
    double* s2 = (double*)malloc((nx * (ny-1)) * sizeof(double));
    double* s3 = (double*)malloc((nx * ny-1) * sizeof(double));
    double* s4 = (double*)malloc((nx * (ny-1)) * sizeof(double));
    double* f = (double*)malloc((nx * ny) * sizeof(double));

    generateMatr(s0, s1, s3, s2, s4, f, nx, ny);
    inverse(s0, s1, s2, s3, s4, nx, ny);

    /*double* x = SLAE(s0, s1, s2, s3, s4, f, nx, ny);
    for (int i = 0; i < nx * ny; i++) {
        cout << x[i] << "\t";
    }*/
}