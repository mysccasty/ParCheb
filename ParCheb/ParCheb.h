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
double* cheb(int nx, int ny, int N, double* a0, double* a1, double* a2, double* a3, double* a4, double* f, double* b, double* a, double* g);
double* SLAE(double* s0, double* s1, double* s2, double* s3, double* s4, double* f, int nx, int ny);
void inverse(double* s0, double* s1, double* s2, double* s3, double* s4, int nx, int ny);

#endif // PARCHEB_H