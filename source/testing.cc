#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
// #include <Eigen/Dense>
// #include <Eigen/Eigenvalues>
#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Eigenvalues>
#include <nanoflann.hpp>
#include "ConstituitiveDD.h"


int main() {
    // Your code here
    MatrixXd strain = MatrixXd::Zero(3,3);
    std::vector<double> props; 
    props = {1.0, 2.0, 3.0, 4.0, 5.0};

    DDElasticMaterial material(3, props, 0.5);
    material.computeStress(strain, true);

    std::cout<<material.sigma<<std::endl;

}