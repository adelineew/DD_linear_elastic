#ifndef CONSTITUITIVEDD_CC_
#define CONSTITUITIVEDD_CC_

#include "ConstituitiveDD.h"

#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
// #include <Eigen/Dense>
// #include <Eigen/Eigenvalues>
#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Eigenvalues>
#include <nanoflann.hpp>

#include <cstdlib>
#include <ctime>
#include "utils.h"

#include <vector>
#include <map>
#include <filesystem>
#include <fstream>


using namespace Eigen;
using namespace nanoflann;

DDElasticMaterial::DDElasticMaterial() {}
DDElasticMaterial::DDElasticMaterial(int d, double mu_, double bulk_, double beta_) : dim(d), mu(mu_), bulk(bulk_), size(d * d), beta(beta_) 
{
    sigma = MatrixXd::Zero(1, size);  // row vector 1x9
    epsBar = MatrixXd::Zero(1, size); // row vector 1x9
    double beta2 = 50000000000000000.0/1000000.0;
    rad =  0.0000005*10000.0;  //std::sqrt(14.0 / beta2); //0.0000005*200.0*100.0;                     // std::sqrt(14.0 / beta);

    // std::vector<std::vector<std::vector<std::vector<double>>>> Ctemp;
    // Ctemp = std::vector<std::vector<std::vector<std::vector<double>>>>(dim, std::vector<std::vector<std::vector<double>>>(dim, std::vector<std::vector<double>>(dim, std::vector<double>(dim, 0.0))));
    C = MatrixXd::Zero(size, size);
    std::vector<std::vector<std::vector<std::vector<double>>>> Ctemp(dim, std::vector<std::vector<std::vector<double>>>(dim, std::vector<std::vector<double>>(dim, std::vector<double>(dim, 0.0))));


      for (unsigned int i = 0; i < dim; i ++){
        for (unsigned int j = 0; j < dim; j ++){
          for (unsigned int k = 0; k < dim; k ++){
            for (unsigned int l = 0; l < dim; l ++){
            //   Ctemp[i][j][k][l] = ((i == j) && (k ==l) ? (bulk-2.0*mu/3.0) : 0.0) +
            //     ((i == k) && (j ==l) ? mu : 0.0) + ((i == l) && (j ==k) ? mu : 0.0);
            //     C(i*dim+j,k*dim+l) = Ctemp[i][j][k][l];
            Ctemp[i][j][k][l] = ((i == j) && (k ==l) ? (bulk-2.0*mu/3.0) : 0.0) +
                ((i == k) && (j ==l) ? mu : 0.0) + ((i == l) && (j ==k) ? mu : 0.0);  
                // C(i*dim+j,k*dim+l)  = ((i == j) && (k ==l) ? (bulk-2.0*mu/3.0) : 0.0) +
                // ((i == k) && (j ==l) ? mu : 0.0) + ((i == l) && (j ==k) ? mu : 0.0);
            }}}}

                Ctemp[0][2][0][2] = Ctemp[0][2][0][2] + Ctemp[0][2][2][0]; 
                Ctemp[2][0][2][0] = Ctemp[0][2][0][2] + Ctemp[0][2][2][0];
                Ctemp[0][2][2][0] = 0.0;
                Ctemp[2][0][0][2] = 0.0;
                Ctemp[1][2][1][2] = Ctemp[1][2][1][2] + Ctemp[1][2][2][1];
                Ctemp[2][1][2][1] = Ctemp[1][2][1][2] + Ctemp[1][2][2][1];
                Ctemp[1][2][2][1] = 0.0; 
                Ctemp[2][1][1][2] = 0.0;


        for (unsigned int i = 0; i < dim; i ++){
        for (unsigned int j = 0; j < dim; j ++){
          for (unsigned int k = 0; k < dim; k ++){
            for (unsigned int l = 0; l < dim; l ++){              
                 C(i*dim+j,k*dim+l)  = Ctemp[i][j][k][l];
            }}}}


}

void DDElasticMaterial::computeStress(const MatrixXd& epsilon) {
    MatrixXd eps = epsilon;
    eps.resize(1, size);

    double query_pt[size];
    for (int i = 0; i < size; ++i)
    {
        query_pt[i] = eps(i);
    }

    // double query_pt[size] = {-0.008, -0.015, 0.0, -0.015, -0.005, 0.0, 0.0, 0.0, 0.0};

    // Loop through the array and print each element
    //                 for (int i = 0; i < size; ++i) {
    //                     std::cout << "query_pt[" << i << "] = " << query_pt[i] << std::endl;
    //                 }
    // std::cout<<"query_pt \n"<<query_pt<<std::endl;

    typedef KDTreeEigenMatrixAdaptor<MatrixXd> eps_tree;
    const int max_leaf = 10;
    eps_tree mat_index(size, data, max_leaf);
    mat_index.index->buildIndex();

    // Find nearest neighbors in radius
    std::vector<std::pair<std::ptrdiff_t, double>> matches;

    // Search parameters
    nanoflann::SearchParams params;

    // Perform the radius search
    const size_t nMatches = mat_index.index->radiusSearch(&query_pt[0], rad * rad, matches, params);

    // std::cout << "RadiusSearch(): radius = " << rad << " -> "
    //           << nMatches << " matches" << std::endl;
 /*    for (size_t i = 0; i < nMatches; i++)
        std::cout << "Idx[" << i << "] = " << matches[i].first
                  << " dist[" << i << "] = " << matches[i].second << std::endl;
    std::cout << std::endl; */

    int count = nMatches; //matches.size();
    VectorXd p(count);
    double Z = 0.0;
    for (int k = 0; k < count; ++k)
    {
        p[k] = std::exp(-0.5 * beta * matches[k].second * matches[k].second);
        Z += p[k]; 
            }
            p /= Z;

            for (int k = 0; k < count; ++k) {
                sigma += p[k] * datastress.row(matches[k].first).segment(0, size);
                epsBar += p[k] * data.row(matches[k].first).segment(0, size);
            }
    }

MatrixXd DDElasticMaterial::computeTangent(const MatrixXd &epsilon) {

            MatrixXd eps = epsilon;
            eps.resize(1, size); //CHECK!

            double query_pt[size];
            for (int i = 0; i < size; ++i)
            {
                query_pt[i] = eps(i);
            }

            typedef KDTreeEigenMatrixAdaptor<MatrixXd> eps_tree;
            const int max_leaf = 10;
            eps_tree mat_index(size, data, max_leaf);
            mat_index.index->buildIndex();

            // Find nearest neighbors in radius
            std::vector<std::pair<std::ptrdiff_t, double>> matches;
            nanoflann::SearchParams params;
            const size_t nMatches = mat_index.index->radiusSearch(query_pt, rad * rad, matches, params);

            int count = nMatches;
            VectorXd p(count);
            double Z = 0.0;
            for (int k = 0; k < count; ++k)
            {
                p[k] = std::exp(-0.5 * beta * matches[k].second*matches[k].second);
                Z += p[k];
            }
            p /= Z;
            
            MatrixXd D = MatrixXd::Identity(size, size);
            MatrixXd M0 = MatrixXd::Zero(size, size);
            std::vector<std::vector<std::vector<std::vector<double>>>> Mtemp(dim, std::vector<std::vector<std::vector<double>>>(dim, std::vector<std::vector<double>>(dim, std::vector<double>(dim, 0.0))));
            for (int k = 0; k < count; ++k)
            {
                M0 += beta * p[k] * ((datastress.row(matches[k].first).segment(0, size)).transpose() * ( C *  (data.row(matches[k].first).segment(0, size) - epsBar).transpose()   ).transpose());
            }

            // Populate the 4D std::vector with values from the flattened array
            std::vector<std::vector<std::vector<std::vector<double>>>> M0_tensor(3, std::vector<std::vector<std::vector<double>>>(3, std::vector<std::vector<double>>(3, std::vector<double>(3))));

            for (int i = 0; i < 3; ++i)
            {
              for (int j = 0; j < 3; ++j)
              {
                for (int k = 0; k < 3; ++k)
                {
                  for (int l = 0; l < 3; ++l)
                  {
                    M0_tensor[i][j][k][l] = M0(i*dim+j,k*dim+l);
                  }
                }
              }
            }

            // Perform operations on the tensor
            M0_tensor[0][2][0][2] = M0_tensor[0][2][0][2] + M0_tensor[0][2][2][0];
            M0_tensor[2][0][2][0] = M0_tensor[0][2][0][2] + M0_tensor[0][2][2][0];
            M0_tensor[0][2][2][0] = 0.0;
            M0_tensor[2][0][0][2] = 0.0;
            M0_tensor[1][2][1][2] = M0_tensor[1][2][1][2] + M0_tensor[1][2][2][1];
            M0_tensor[2][1][2][1] = M0_tensor[1][2][1][2] + M0_tensor[1][2][2][1];
            M0_tensor[1][2][2][1] = 0.0;
            M0_tensor[2][1][1][2] = 0.0;

            // Convert the tensor back to a 9x9 matrix
            Eigen::MatrixXd M0_reshaped(9, 9);
            for (int i = 0; i < 3; ++i)
            {
              for (int j = 0; j < 3; ++j)
              {
                for (int k = 0; k < 3; ++k)
                {
                  for (int l = 0; l < 3; ++l)
                  {
                    M0_reshaped(i*dim+j,k*dim+l) = M0_tensor[i][j][k][l];
                    // std::cout<<M0_tensor[i][j][k][l]<<"     ";
                  }
                }
              }
            }

            // std::cout << std::endl << M0_reshaped(2 * dim + 0, 0 * dim + 2) << std::endl;


            return M0_reshaped;
}



MatrixXd DDElasticMaterial::computeSensitivitySigma(const MatrixXd &epsilon) {

            MatrixXd eps = epsilon;
            eps.resize(1, size);

            double query_pt[size];
            for (int i = 0; i < size; ++i) {
                query_pt[i] = eps(i);
            }

            typedef KDTreeEigenMatrixAdaptor<MatrixXd> eps_tree;
            const int max_leaf = 10;
            eps_tree mat_index(size, data, max_leaf);
            mat_index.index->buildIndex();

            // Find nearest neighbors in radius
            std::vector<std::pair<std::ptrdiff_t, double>> matches;
            nanoflann::SearchParams params;
            const size_t nMatches = mat_index.index->radiusSearch(query_pt, rad * rad, matches, params);
            int count = nMatches;
            VectorXd p(count);
            double Z = 0.0;
            for (int k = 0; k < count; ++k) {
                p[k] = std::exp(-0.5 * beta * matches[k].second*matches[k].second);
                Z += p[k];
            }
            p /= Z;

            for (int k = 0; k < count; ++k) {
            std::cout<<"p , k \n"<<k<<" "<<p[k] <<std::endl;
            }

            MatrixXd D = MatrixXd::Identity(size, size);
            MatrixXd SS = MatrixXd::Zero(size, size);
            MatrixXd SSALL = MatrixXd::Zero(count, size * size);
            for (int k = 0; k < count; ++k) {
                SS = p[k] * D;
                SSALL.row(k) = Map<RowVectorXd>(SS.data(), size * size);
            }
            return SSALL;
        }

MatrixXd DDElasticMaterial::computeSensitivityEps(const MatrixXd &epsilon) {
            MatrixXd eps = epsilon;
            eps.resize(1, size);

            double query_pt[size];
            for (int i = 0; i < size; ++i) {
                query_pt[i] = eps(i);
            }

            typedef KDTreeEigenMatrixAdaptor<MatrixXd> eps_tree;
            const int max_leaf = 10;
            eps_tree mat_index(size, data, max_leaf);
            mat_index.index->buildIndex();

            // Find nearest neighbors in radius
            std::vector<std::pair<std::ptrdiff_t, double>> matches;
            nanoflann::SearchParams params;
            const size_t nMatches = mat_index.index->radiusSearch(query_pt, rad * rad, matches, params);
            int count = nMatches;
            VectorXd p(count);
            double Z = 0.0;
            for (int k = 0; k < count; ++k) {
                p[k] = std::exp(-0.5 * beta * matches[k].second*matches[k].second);
                Z += p[k];
            }
            p /= Z;

            MatrixXd SEALL = MatrixXd::Zero(count, size * size);
            MatrixXd SE = MatrixXd::Zero(size, size);
            for (int k = 0; k < count; ++k) {
                SE = beta * p[k] * (sigma - datastress.row(matches[k].first).segment(0, size) * 
                                            ( C * (data.row(matches[k].first).segment(0, size) - eps).transpose()  ).transpose());
                SEALL.row(k) = Map<RowVectorXd>(SE.data(), SE.size());
            }
            return SEALL;
        }

//if SE were to be [1 2 3 4; 4 5 6 7; 8 9 8 9; 10 11 12 13], what would MatrixXd SEALL = MatrixXd::Zero(count, size * size); return
//SE << 1, 2, 3, 4,
 //     4, 5, 6, 7,
 //     8, 9, 8, 9,
 //     10, 11, 12, 13;
// RESULTING SE_vector is 1 2 3 4 4 5 6 7 8 9 8 9 10 11 12 13
// MatrixXd SEALL = MatrixXd::Zero(4, 16);
// for (int i = 0; i < count; ++i) {
//     SEALL.row(i) = SE_vector;
// }
// 1  2  3  4  4  5  6  7  8  9  8  9  10 11 12 13
// 1  2  3  4  4  5  6  7  8  9  8  9  10 11 12 13
// 1  2  3  4  4  5  6  7  8  9  8  9  10 11 12 13
// 1  2  3  4  4  5  6  7  8  9  8  9  10 11 12 13

void DDElasticMaterial::read_input_file(const std::string &filename2)
  {
    unsigned int maxread = 1033770;
    unsigned int MAXLINE = 1024;
    unsigned int indexmax;

      valuestrain.resize(maxread, std::vector<double>(size, 0.0));
      valuestress.resize(maxread, std::vector<double>(size, 0.0));

      FILE *fid;
      int endOfFileFlag;
      char nextLine[MAXLINE];

      int valuesWritten;
      bool fileReadErrorFlag = false;


      fid = std::fopen(filename2.c_str(), "r");
      if (fid == NULL)
      {
        if (std::filesystem::exists(filename2))
        {
          std::ifstream file(filename2);
          if (!file)
          {
            std::cout << "Unable to open file DIC \"" << filename2 << "\". File exists but could not be opened." << std::endl;
            perror("Error opening file");
          }
          else
          {
            std::cout << "File DIC \"" << filename2 << "\" opened successfully." << std::endl;
          }
        }
        else
        {
          std::cout << "Unable to open file DIC \"" << filename2 << "\". File does not exist." << std::endl;
        }
        fileReadErrorFlag = true;
      }

      else
      {
         for(unsigned int index = 0; index < maxread; ++index){ 
          getNextDataLine(fid, nextLine, MAXLINE, &endOfFileFlag);
          if (endOfFileFlag == 0){
          valuesWritten = sscanf(nextLine, "%lg\t%lg\t%lg\t%lg\t%lg\t%lg\t%lg\t%lg\t%lg\t%lg\t%lg\t%lg\t%lg\t%lg\t%lg\t%lg\t%lg\t%lg ", 
          &valuestrain[index][0], 
          &valuestrain[index][1],
          &valuestrain[index][2],
          &valuestrain[index][3],
          &valuestrain[index][4],
          &valuestrain[index][5],
          &valuestrain[index][6],
          &valuestrain[index][7],
          &valuestrain[index][8],
            &valuestress[index][0],
            &valuestress[index][1],
            &valuestress[index][2],
            &valuestress[index][3],
            &valuestress[index][4],
            &valuestress[index][5],
            &valuestress[index][6],
            &valuestress[index][7],
            &valuestress[index][8]);

          }
          else{
            indexmax = index;
            break;
          }
        } 

      }

      if (fileReadErrorFlag)
      {
        std::cout << "Error reading input file DIC, Exiting.\n"
                  << std::endl;
        exit(1);
      }

      fclose(fid);

        nPtsDB = indexmax+1; 
        std::cout<<"nPtsDB "<<nPtsDB<<std::endl;

        valuestrain.resize(nPtsDB);
        valuestress.resize(nPtsDB);

        std::cout << "eps database out very last index: "<<valuestrain[indexmax-1][0] << " " << valuestrain[indexmax][1] << " " << valuestrain[indexmax][2] << " " << valuestrain[indexmax][3] << " " << valuestrain[indexmax][4] << " " << valuestrain[indexmax][5] << " " << valuestrain[indexmax][6] << " " << valuestrain[indexmax][7] << " " << valuestrain[indexmax][8] << std::endl;
        std::cout << "sigma database out very last index: "<<valuestress[indexmax-1][0] << " " << valuestress[indexmax][1] << " " << valuestress[indexmax][2] << " " << valuestress[indexmax][3] << " " << valuestress[indexmax][4] << " " << valuestress[indexmax][5] << " " << valuestress[indexmax][6] << " " << valuestress[indexmax][7] << " " << valuestress[indexmax][8] << std::endl;

        unsigned int d = dim;

        data.resize(nPtsDB * 1, d * d);
        for (int n = 0; n < nPtsDB; ++n)
        {
            for (int i = 0; i < d * d; ++i)
            {
                data(n, i) = valuestrain[n][i];
            }
        }

        datastress.resize(nPtsDB * 1, d * d);
        for (int n = 0; n < nPtsDB; ++n)
        {
            for (int i = 0; i < d * d; ++i)
            {
                datastress(n, i) = valuestress[n][i];
            }
        }

        std::cout << " read done" << std::endl;
  }

void DDElasticMaterial::getNextDataLine(FILE *const filePtr, char *nextLinePtr, int const maxSize, int *const endOfFileFlag)
  {
    *endOfFileFlag = 0;
    do
    {
      if (fgets(nextLinePtr, maxSize, filePtr) == NULL)
      {
        *endOfFileFlag = 1;
        break;
      }
      while ((nextLinePtr[0] == ' ' || nextLinePtr[0] == '\t') ||
             (nextLinePtr[0] == '\n' || nextLinePtr[0] == '\r'))
      {
        nextLinePtr = (nextLinePtr + 1);
      }
    } while ((strncmp("#", nextLinePtr, 1) == 0) || (strlen(nextLinePtr) == 0));
  }



#endif
