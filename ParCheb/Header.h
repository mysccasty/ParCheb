#pragma once

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
typedef struct MyDiagMatrStruct
{
    int nx;
    int ny;
    double* a0;
    double* a1;
    double* a2;
    double* a3;
    double* a4;
    double* f;
} MyDiagMatrStruct;
typedef struct MyVectorStruct
{
    int nx;
    int ny;
    double* u;
} MyVectorStruct;