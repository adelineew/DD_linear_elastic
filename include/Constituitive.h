#ifndef CONSTITUITIVE_H_
#define CONSTITUITIVE_H_

#include <iostream>
#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/tensor.h>
#include <deal.II/base/symmetric_tensor.h>

#define Dims 3

using namespace dealii;


  class LinearElastic
  {
    public:

	  LinearElastic(){ mu = 0; lambda = 0; }; //default constructor
    virtual ~LinearElastic(){};


    LinearElastic(double E, double nu) //nondefault constructor
    {
      mu = E/(2.0*(1 + nu));
      lambda = E*nu/((1 + nu)*(1 - 2.0*nu));

      for (unsigned int i = 0; i < Dims; i ++)
        for (unsigned int j = 0; j < Dims; j ++)
          for (unsigned int k = 0; k < Dims; k ++)
            for (unsigned int l = 0; l < Dims; l ++)
              C[i][j][k][l] = ((i == j) && (k ==l) ? lambda : 0.0) +
                ((i == k) && (j ==l) ? mu : 0.0) + ((i == l) && (j ==k) ? mu : 0.0);
    };

    void init(double E, double nu)
    {
      mu = E/(2.0*(1 + nu));
      lambda = E*nu/((1 + nu)*(1 - 2.0*nu));

      for (unsigned int i = 0; i < Dims; i ++)
        for (unsigned int j = 0; j < Dims; j ++)
          for (unsigned int k = 0; k < Dims; k ++)
            for (unsigned int l = 0; l < Dims; l ++)
              C[i][j][k][l] = ((i == j) && (k ==l) ? lambda : 0.0) +
                ((i == k) && (j ==l) ? mu : 0.0) + ((i == l) && (j ==k) ? mu : 0.0);
    };

    double get_energy(Tensor<2, Dims> &grad_u);
    void get_sigma(Tensor<2, Dims> &grad_u, Tensor<2, Dims> &sigma);

    Tensor<4, Dims> C;

    private:
      double lambda;
      double mu;
  };

  #endif
