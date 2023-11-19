#include <omp.h>
#include <iostream>
#include <fstream>
#include <chrono>
#include <iomanip>
#define _USE_MATH_DEFINES
using namespace std;

typedef struct MyChebStructCircly
{
    double lambda_max;
    double lambda_min;
    double* T_n;
    int N;

} MyChebStructCircly;
#define PI 3.1415926535897932384626433832795

void generateMatr(double*& a0, double*& a1, double*& a2, double*& a3, double*& a4, double*& f, int nx, int ny) {
    for (int i = 0; i < nx * ny; i++) {
        a0[i] = 4;
        if (i == 0 | i == nx - 1 | i == nx * (ny - 1) | i == nx * ny - 1) {
            f[i] = 2;
        }
        else if ((i > 0 & i < nx - 1) | ((i + 1) % nx == 0) | (i % nx == 0) | (i > nx * (ny - 1) & i < nx * ny - 1)) {
            f[i] = 1;
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
            a1[i] = -1;
            a2[i] = -1;
        }
    }
    for (int i = 0; i < nx * (ny - 1); i++) {
        a3[i] = -1;
        a4[i] = -1;
    }
}
double norm(double* vector, int len) {
    int i;
    double sum = 0;
#pragma omp parallel for private(i) reduction(+:sum)
    for (i = 0; i < len; i++) {
        sum += vector[i] * vector[i];
    }
    return sqrt(sum);
}


void nevyazka(double*& a0, double*& a1, double*& a2, double*& a3, double*& a4, double*& f, double*& u, double*& R, int nx, int ny) {

    int i;
#pragma omp parallel for shared(a0, a1, a2, a3, a4, f, u, R) private(i)
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
void coef(int* a, int* b, int fn, int ln) {

    int j = 1;
    for (int i = 1; i < fn + 1; i++) {
        b[j] = a[i];
        b[j + 1] = fn * 4 - a[i];
        j = j + 2;
    }
    fn = fn * 2;
    for (int i = 1; i < fn + 1; i++) {
        a[i] = b[i];
    }
    if (fn == ln) {
        return;
    }
    coef(a, b, fn, ln);
}
int* giveParam(int N) {
    int* a = (int*)malloc((N + 1) * sizeof(int));
    int* b = (int*)malloc((N + 1) * sizeof(int));
    int firstN = 2;
    a[1] = 1;
    a[2] = 3;
    coef(a, b, firstN, N);
    return a;
}


void ConstrChebCircly(MyChebStructCircly* my_cheb) {
    int n;

    double* index = my_cheb->T_n;
    int N = my_cheb->N;
    double lmax = my_cheb->lambda_max;
    double lmin = my_cheb->lambda_min;
    int* Tn = giveParam(N);
    for (n = 1; n <= N; n++) {

        index[n] = 2.0 / (lmax + lmin - (lmax - lmin) * (cos((2 * (double)Tn[n] - 1) * (double)PI / (2 * (double)N))));
    }
    free(Tn);
}
//Решение СЛАУ
void cheb(int nx, int ny, int N, double* a0, double* a1, double* a2, double* a3, double* a4, double* x, double* r, double* f, double* T_n, ofstream& out) {

    double epsilon = 1e-10;
    out << "Nx = " << nx << endl;
    out << "Ny = " << ny << endl;
    double fNorm;

    auto start = std::chrono::high_resolution_clock::now();
    fNorm = norm(f, nx*ny);

    for (int iter = 1; iter < N; iter++) {
        nevyazka(a0, a1, a2, a3, a4, f, x, r, nx, ny);

        if (norm(r, nx*ny) < epsilon * fNorm) {
            out << iter << " iterations need to get " << epsilon << " error" << endl;
            break;
        }
        for (int i = 0; i < nx * ny; i++) {
            x[i] += T_n[iter] * r[i];
        }


    }
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration = end - start;
    out << "Time: " << duration.count() << " ms" << endl;
}
//Вычисление обратной
void inverse(int nx, int ny, int N, double* a0, double* a1, double* a2, double* a3, double* a4, double* x, double* r, double* f, double* T_n, ofstream& out) {
    out << "Nx = " << nx << endl;
    out << "Ny = " << ny << endl;

    double epsilon = 1e-10;
    double** AInv = (double**)malloc((nx * ny) * sizeof(double*));

    for (int i = 0; i < nx * ny; i++) {
        AInv[i] = (double*)malloc((nx * ny) * sizeof(double));
    }
    int k;
    auto start = std::chrono::high_resolution_clock::now();
#pragma omp parallel for shared(AInv) private(k, x, r, f)
    for (k = 0; k < nx * ny; k++) {
        x = (double*)malloc((nx * ny) * sizeof(double));
        for (int i = 0; i < nx * ny; i++) {
            x[i] = 2;
        }
        r = (double*)malloc((nx * ny) * sizeof(double));
        f = (double*)malloc((nx * ny) * sizeof(double));
        for (int i = 0; i < nx * ny; i++) {
            f[i] = 0;
        }
        f[k] = 1;



        for (int j = 1; j < N; j++) {
            nevyazka(a0, a1, a2, a3, a4, f, x, r, nx, ny);

            if (norm(r, nx*ny) < epsilon) {
                break;
            }
            for (int i = 0; i < nx * ny; i++) {
                x[i] += T_n[j] * r[i];
            }


        }
        for (int i = 0; i < nx * ny; i++) {
            AInv[k][i] = x[i];
        }
        free(f);
        free(x);
        free(r);

    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration = end - start;
    out << "Time: " << duration.count() << " ms" << std::endl;
    for (int i = 0; i < nx * nx; i++) {
        /*for (int j = 0; j < nx * ny; j++) {
            out << std::fixed;
            out << std::setprecision(5);
            out << AInv[i][j];
            out << "\t";
        }
        out << endl;*/
        free(AInv[i]);
    }
    free(AInv);
}
int main() {
    int nx = 3, N = 16 * 2048;
    int ny = nx;

    double* T_n = (double*)malloc((N) * sizeof(double));

    double fNorm;
    MyChebStructCircly MyCheb;


    MyCheb.N = N;
    MyCheb.T_n = T_n;
    double* a0 = (double*)malloc((nx * ny) * sizeof(double));
    double* a1 = (double*)malloc((nx * ny - 1) * sizeof(double));
    double* a2 = (double*)malloc((nx * ny - 1) * sizeof(double));
    double* a3 = (double*)malloc((nx * (ny - 1)) * sizeof(double));
    double* a4 = (double*)malloc((nx * (ny - 1)) * sizeof(double));
    double* f = (double*)malloc((nx * ny) * sizeof(double));
    double* x = (double*)malloc((nx * ny) * sizeof(double));;
    double* r = (double*)malloc((nx * ny) * sizeof(double));;
    generateMatr(a0, a1, a2, a3, a4, f, nx, ny);
    MyCheb.lambda_max = 8.0 * sin(((double)nx + 1) * PI / (2 * ((double)nx + 1) + 2)) * sin(((double)nx + 1) * PI / (2 * ((double)nx + 1) + 2));
    MyCheb.lambda_min = 8.0 * sin(PI / (2 * ((double)nx + 1) + 2)) * sin(PI / (2 * ((double)nx + 1) + 2));
    ConstrChebCircly(&MyCheb);

    omp_set_dynamic(0);
    omp_set_num_threads(12);
    std::ofstream result;
    result.open("result.txt");
    cheb(nx, ny, N, a0, a1, a2, a3, a4, x, r, f, T_n, result);
    result.close();
    std::ofstream invMatr;
    invMatr.open("InverseMatrix.txt");
    inverse(nx, ny, N, a0, a1, a2, a3, a4, x, r, f, T_n, invMatr);
    invMatr.close();
}