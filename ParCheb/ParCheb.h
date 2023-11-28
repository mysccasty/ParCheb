#ifndef PARCHEB_H
#define PARCHEB_H

#include <iostream>
#include <fstream>
#include <chrono>
#include <iomanip>
#include <omp.h>
#include <cmath>
#include <fstream>
using namespace std;

double norm(double* vector, int len);
void nevyazka(double* a0, double* a1, double* a2, double* a3, double* a4, double* f, double* u, double*& R, int nx, int ny, double alpha);
void cheb(int nx, int ny, int N, double* a0, double* a1, double* a2, double* a3, double* a4, double* f, double* beta, double* alpha, ofstream& out);
void SLAU(double* s0, double* a1, double* a2, double* a3, double* a4, double* f, int nx, int ny);

#endif // PARCHEB_H