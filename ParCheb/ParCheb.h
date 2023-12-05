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
void nevyazka(double* a0, double* a1, double* a2, double* a3, double* a4, double* f, double* u, double*& R, int nx, int ny);
void cheb(int nx, int ny, int N, double* a0, double* a1, double* a2, double* a3, double* a4, double* f, double* x, double tau);
void SLAE(double* s0, double* s1, double* s2, double* s3, double* s4, double* x, double* f, int nx, int ny);

#endif // PARCHEB_H