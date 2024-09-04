#ifndef CONSTITUITIVEDD_H_
#define CONSTITUITIVEDD_H_

#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
// #include <Eigen/Dense>
// #include <Eigen/Eigenvalues>
#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Eigenvalues>
#include <nanoflann.hpp>

using namespace Eigen;
using namespace nanoflann;


#define DIM 3

  class DDElasticMaterial
  {
   public:
    int nPtsDB;
    int nItems = 2;
    int dim;
    int size;
    double beta;
    double rad;

double mu; double bulk;

    // std::vector<double> properties; 
    std::vector<std::vector<double>> valuestrain, valuestress;
    MatrixXd data; 
    MatrixXd datastress;
    MatrixXd sigma;
    MatrixXd epsBar;
    MatrixXd C;

    DDElasticMaterial();
    DDElasticMaterial(int d, double mu_, double bulk_, double beta_);
    

    void computeStress(const MatrixXd &epsilon);
    MatrixXd computeTangent(const MatrixXd &epsilon);
    MatrixXd computeSensitivitySigma(const MatrixXd &epsilon);
    MatrixXd computeSensitivityEps(const MatrixXd &epsilon);

    void read_input_file(const std::string &filename2);
    void getNextDataLine(FILE *const filePtr, char *nextLinePtr, int const maxSize, int *const endOfFileFlag);

    private:
  };


  #endif
