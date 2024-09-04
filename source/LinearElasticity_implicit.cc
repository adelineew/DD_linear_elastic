/* ---------------------------------------------------------------------
 *
 *
 *
 *
 * May 22, 2023.
 * Adeline WIhardja
 * For incompressible LCE, doing implicit dynamics first.
 * ---------------------------------------------------------------------
 *
 *
 */

#ifndef LINEAR_ELASTIC_CC_
#define LINEAR_ELASTIC_CC_
#include <fstream>
#include <iostream>
#include <string>
#include <cmath>
#include <stdlib.h>

#include <sys/types.h>
#include <sys/stat.h>
#include <time.h>
#include <bits/stdc++.h>
#include "../include/LinearElasticity_implicit.h"
#include <math.h>

#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Eigenvalues>
#include <nanoflann.hpp>

#define DIM 3

namespace lce_elastic
{
	using namespace dealii;
  using dealii::Vector;
  using dealii::SparseMatrix;
  using dealii::FullMatrix;
  using dealii::Point;
  using dealii::Tensor;
  using dealii::SymmetricTensor;

  using namespace Eigen;
  using Eigen::MatrixXd;
  using Eigen::VectorXd;


	ElasticProblem::ElasticProblem()
		: dof_handler(triangulation),
		  fe(FESystem<DIM>(FE_Q<DIM>(1), DIM), 1),
		  time_step(0.2),
		  time(time_step),
		  timestep_number(1)
      	//LE(mu, nu)
	{
	}

	ElasticProblem::~ElasticProblem()
	{
		dof_handler.clear();
	}

void ElasticProblem::sigma_convert_to_tensor(const Tensor<2,DIM> &tensorin, Tensor<2, DIM> &tensorout)
{

	//input strain tensor and get out stress tensor from DD constitutive
      SymmetricTensor<2, DIM> tensorin2; tensorin2 = symmetrize(tensorin);
      MatrixXd epstemp, pktemp;
      epstemp = MatrixXd::Zero(1, DIM * DIM);
      pktemp = MatrixXd::Zero(1, DIM * DIM);
      for (unsigned int i = 0; i < DIM; ++i)
      {
        for (unsigned int j = 0; j < DIM; ++j)
        {
          epstemp(i * DIM + j) = tensorin2[i][j];
        }      }
      DD1->computeStress(epstemp);
      pktemp = DD1->sigma;
    // std::cout<<"output sigma: "<<DD1->sigma<<std::endl;
    // std::cout<<"output eps bar: "<<DD1->epsBar<<std::endl;

      int k = 0;
      for (unsigned int i = 0; i < DIM; ++i)
      {
        for (unsigned int j = 0; j < DIM; ++j)
        {
          tensorout[i][j] = pktemp(i * DIM + j);
          k = k + 1;
        }      }
}

void ElasticProblem::tangent_convert_to_tensor(const Tensor<2,DIM> &tensorin, Tensor<4, DIM> &tensorout)
{
	// input strain tensor and get out tangent tensor from DD constitutive
  //the tangent matrix from DD has dimension size*size
      SymmetricTensor<2, DIM> tensorin2; tensorin2 = symmetrize(tensorin);
      MatrixXd epstemp, tatemp;
      epstemp = MatrixXd::Zero(1, DIM * DIM);
      tatemp = MatrixXd::Zero(DIM * DIM, DIM * DIM);
      for (unsigned int i = 0; i < DIM; ++i)
      {
        for (unsigned int j = 0; j < DIM; ++j)
        {
              epstemp(i * DIM + j) = tensorin2[i][j];
            }          }       

      tatemp = DD1->computeTangent(epstemp);

      for (unsigned int i = 0; i < DIM; ++i)
      {
        for (unsigned int j = 0; j < DIM; ++j)
        {
          for (unsigned int k = 0; k < DIM; ++k)
          {
            for (unsigned int l = 0; l < DIM; ++l)
            {
              tensorout[i][j][k][l] = tatemp( i * DIM + j, k * DIM + l ); //tatemp(i * DIM * DIM * DIM + j * DIM * DIM + k * DIM + l);//// check thistodooooo
            }      }        }      }

	// std::cout<<"output begin tangent"<<std::endl;
    //     for (int i = 0; i < tatemp.rows(); ++i) {
    //     for (int j = 0; j < tatemp.cols(); ++j) {
    //         std::cout << tatemp(i, j) << " ";
    //     }
    // } 

}

void ElasticProblem::sens_convert_to_tensor(const MatrixXd &tensorin, std::vector<Tensor<4, DIM>> &tensorout)
{
    std::vector<Tensor<2, DIM*DIM>> sensSigmaalltemp;
    sensSigmaalltemp.resize(tensorin.rows());

        for (int i = 0; i < tensorin.rows(); ++i) {
        for (int j = 0; j < DIM*DIM; ++j) {
          for(int k = 0; k < DIM*DIM; ++k){
            sensSigmaalltemp[i][j][k] = tensorin(i, j*DIM*DIM + k);
            std::cout << tensorin(i, j*DIM*DIM + k) << " ";
          }
        }
        std::cout << std::endl;
    }

    for(unsigned int pp = 0; pp < tensorin.rows(); ++pp){
      for (unsigned int i = 0; i < DIM; ++i)
      {
        for (unsigned int j = 0; j < DIM; ++j)
        {
          for (unsigned int k = 0; k < DIM; ++k)
          {
            for (unsigned int l = 0; l < DIM; ++l)
            {
              tensorout.at(pp)[i][j][k][l] = sensSigmaalltemp.at(pp)[ i * DIM + j][ k * DIM + l ]; 
            }      }        }      }      }
}

void ElasticProblem::testrun_DD(){
    
    std::cout<<"Running DD"<<std::endl;
    int dimension = 3;

    MatrixXd strain = MatrixXd::Zero(3,3);
    strain.resize(1,dimension*dimension);
 
    std::vector<double> props; 
    props.resize(2);
        props[0] = mu; props[1] = bulk;
    // strain(0) = -0.008;
    // strain(1) = -0.015;
    // strain(3) = -0.015;
    // strain(4)= -0.005; 

	strain(0) = 0.00019813; //0.000201739;
	strain(1) = 0.0; //1.21175e-05;
	strain(2) = 0.0; //6.22642e-11;
	strain(3) = 0.0; //1.21175e-05;
	strain(4)= -5.92353e-05 ; //-6.25808e-05;
	strain(5) = 0.0; //-6.81703e-07;
	strain(6) =  0.0; //6.22642e-11;
	strain(7) =  0.0; //-6.81703e-07;
	strain(8) =  -5.92353e-05 ; //-5.92561e-05 ;
    DD1 = new DDElasticMaterial(DIM, mu, bulk, 5000); 
    //DD1->read_input_file("dataddtemp.out"); 
	DD1->read_input_file("DBs_stress_strain_averagerectdomain_4ME_0.3nu_speckle10e_steptime0.5_end1000_refine3x.out");

////////////////////////////////////////////////////////    
   DD1->computeStress(strain);

    std::cout<<"values values \n \n "<<std::endl;
    std::cout<<DD1->sigma<<std::endl;
    std::cout<<DD1->epsBar<<std::endl;
    std::cout<<"done \n \n "<<std::endl;

    // for (int i = 0; i < 10; ++i) {
    //     for (int j = 0; j < DD1->data.cols(); ++j) {
    //         std::cout << DD1->data(i, j) << " ";
    //     }
    //     std::cout << std::endl;
    // }

////////////////////////////////////////////////////////
    std::cout<<"begin tangent"<<std::endl;
    MatrixXd tangent;
    tangent = DD1->computeTangent(strain);

        for (int i = 0; i < tangent.rows(); ++i) {
        for (int j = 0; j < tangent.cols(); ++j) {
            std::cout << tangent(i, j) << " ";
        }
        std::cout << std::endl;
    } 

    std::cout<<"begin C"<<std::endl;
    for (int i = 0; i < DD1->C.rows(); ++i) {
        for (int j = 0; j < DD1->C.cols(); ++j) {
            std::cout << DD1->C(i, j) << " ";
        }
        std::cout << std::endl;
    } 

///////////////////////////////////////////////////////////
/*      std::cout<<"begin sensitivity sigma"<<std::endl;
    MatrixXd sensSigma;
    sensSigma = DD1->computeSensitivitySigma(strain);

    std::vector<Tensor<2, DIM*DIM>> sensSigmaall;
    sensSigmaall.resize(sensSigma.rows());

        for (int i = 0; i < sensSigma.rows(); ++i) {
        for (int j = 0; j < DIM*DIM; ++j) {
          for(int k = 0; k < DIM*DIM; ++k){
            sensSigmaall.at(i)[j][k] = sensSigma(i, j*DIM*DIM + k);
            //std::cout << sensSigma(i, j*DIM*DIM + k) << " ";
          }
        }
        std::cout << std::endl;
    }

            std::cout << sensSigmaall[0][0][0] << " "<< sensSigmaall[0][1][1] << " "
            << sensSigmaall[0][2][2] << " "
            << sensSigmaall[0][3][3] << " "
            << sensSigmaall[0][4][4] << " "
            << sensSigmaall[0][5][5] << " "
            << sensSigmaall[0][6][6] << " "
            << sensSigmaall[0][7][7] << " "
            << sensSigmaall[0][8][8] << " "
            << sensSigmaall[1][0][0] << " ";

///////////////////////////////////////////////////////////
     std::cout<<"begin sensitivity eps"<<std::endl;
    MatrixXd sensEps;
    sensEps = DD1->computeSensitivityEps(strain);

    //reshaping
    std::vector<Tensor<2, DIM*DIM>> sensEpsall;
    sensEpsall.resize(sensEps.rows());

        for (int i = 0; i < sensEps.rows(); ++i) {
        for (int j = 0; j < DIM*DIM; ++j) {
          for(int k = 0; k < DIM*DIM; ++k){
            sensEpsall[i][j][k] = sensEps(i, j*DIM*DIM + k);
            //std::cout << sensEps(i, j*DIM*DIM + k) << " ";
          }
        }
        std::cout << std::endl;
    } */

}


 void ElasticProblem::create_mesh()
	{
		Point<DIM> corner1, corner2;
		corner1(0) = 0.0;
		corner1(1) = 0.0;
		corner1(2) = 0.0;
		corner2(0) = 0.036; //domain_dimensions[0]; // domain_dimensions[0];
		corner2(1) = 0.012; //domain_dimensions[1]; // domain_dimensions[1];
		corner2(2) = 0.003; //domain_dimensions[2]; // domain_dimensions[2];

		grid_dimensions[0] = 6; //12; //24; //50; //40; //25; // 40;
		grid_dimensions[1] = 2; //4; //8; //20; //15; //7; //15;
		grid_dimensions[2] = 1; //2; //5; //7; //5; // 7; 

		yheight = 0.012;
		zheight = 0.003;
		GridGenerator::subdivided_hyper_rectangle(triangulation, grid_dimensions, corner1, corner2, true);

		//triangulation.refine_global(3); //TODOOOOOOOOOOOOO


		std::cout << "Number of active cells: " << triangulation.n_active_cells()
				  << std::endl;

		for (auto &face : triangulation.active_face_iterators())
		{
			if (std::fabs(face->center()(2) - zheight) < 1e-12)
				face->set_boundary_id(11);
		}

		if(get_data == 1){
		cell_indices.resize(triangulation.n_active_cells()); 
	
 		unsigned int o = 0;
		for (auto &cell2 : triangulation.active_cell_iterators())
		{
			for (auto &face2 : cell2->face_iterators())
			{
				if (std::fabs(face2->center()(2) - zheight) < 1e-12){
					cell_indices[cell2->active_cell_index()] = o;
					o=o+1;
				}
			}
		}
		strain_dbs.resize(o);
		stress_dbs.resize(o);
		std::cout<<"size "<<o<<std::endl; 
		}


	}

  void ElasticProblem::create_mesh_hole()
	{
		zheight = 0.005;
		yheight = 0.012;

		Point<DIM> center; 

		//symmetric hole:
    	center(0) = 0.0275-0.0095; 
    	center(1) = 0.00175+0.00375+0.0005; 
    	center(2) = 0.0;		
		GridGenerator::plate_with_a_hole(triangulation,	0.0015, 0.003, 0.003, 0.003, 0.015, 0.015, center, 0, 1, zheight, 2, true);
		triangulation.refine_global(1);

		std::cout << "Number of active cells: " << triangulation.n_active_cells()
				  << std::endl;
		cell_indices.resize(triangulation.n_active_cells()); 

		for (auto &face : triangulation.active_face_iterators())
		{
			if (std::fabs(face->center()(2) - (zheight/2.0)) < 1e-12)
				face->set_boundary_id(11);
				
		}

		if(get_data == 1){
 		unsigned int o = 0;
		for (auto &cell2 : triangulation.active_cell_iterators())
		{
			for (auto &face2 : cell2->face_iterators())
			{
				if (std::fabs(face2->center()(2) - (zheight / 2.0)) < 1e-12){
					cell_indices[cell2->active_cell_index()] = o;
					o=o+1;
				}
			}
		}
		strain_dbs.resize(o);
		stress_dbs.resize(o);
		std::cout<<"size "<<o<<std::endl; 
		}

	}
 
 
	void ElasticProblem::setup_system(const bool initial_step)
	{
		if (initial_step)
		{
			if(ifLE == 1)
			LE.init(mu, nu);

			if(ifDD == 1){
      		DD1 = new DDElasticMaterial(DIM, mu, bulk, 0.7);
			DD1->read_input_file("DBs_stress_strain_averagerectdomain_4ME_0.3nu_speckle10e_steptime0.5_end1000_refine3x.out");}
			//read_input_file("DBs_stress_strain_averageHOLErectdomain_4ME_0.3nu_speckle10e_steptime0.5_end1000.out");}

			dof_handler.distribute_dofs(fe);
			  DoFRenumbering::Cuthill_McKee(dof_handler);

			current_solution.reinit(dof_handler.n_dofs());
			current_solution = 0.0; // ICs
			solution_u.reinit(dof_handler.n_dofs());
			solution_u = 0.0;
			solution_g.reinit(dof_handler.n_dofs());
			solution_g = 0.0;
			residual.reinit(dof_handler.n_dofs());
			residual = 0.0;


			std::vector<bool> sidex_components = {true, false, false};
			ComponentMask sidex_mask(sidex_components);
			DoFTools::extract_boundary_dofs(dof_handler,
											sidex_mask,
											selected_dofs_x,
											{1});

			support_points.resize(dof_handler.n_dofs());
			DoFTools::map_dofs_to_support_points(MappingQ1<DIM>(), dof_handler, support_points, sidex_mask);// to figure out location of the dofs
			int n = 0;
			

			read_input_file_DIC_exp_speckle();
			dofindex.resize(dof_handler.n_dofs());

			if (HOLE == 1)
			{
				for (int i = 0; i < dof_handler.n_dofs(); ++i)
				{
					if (std::fabs(support_points[i](2) - (zheight / 2.0)) < 1e-12)
					{
						dofindex[n] = i; // i is the index of the dof we are interested in 
						solution_g[i] = speckle[n]; // from read_input_file_DIC_exp_speckle
						n = n + 1;
					}
				}
				dofindex.resize(n);

				for (int p = 0; p < dofindex.size(); ++p)
				{
					solution_g[dofindex[p]] = speckle[p];
				}
			}
			specklepixels_all.resize(endsteptime);

			support_points_all_dofs.resize(dof_handler.n_dofs());
			DoFTools::map_dofs_to_support_points(MappingQ1<DIM>(), dof_handler, support_points_all_dofs);// to figure out location of the dofs


			setup_system_constraints();

			std::cout << "Number of degrees of freedom: " << dof_handler.n_dofs()
					  << std::endl
					  << std::endl;

		}

		newton_update.reinit(dof_handler.n_dofs());
		newton_update = 0.0;

		DynamicSparsityPattern dsp(dof_handler.n_dofs(), dof_handler.n_dofs());
		DoFTools::make_sparsity_pattern(dof_handler,
										dsp,
										constraints,
										true);

		sparsity_pattern.copy_from(dsp);
		system_matrix.reinit(sparsity_pattern);
		system_rhs.reinit(dof_handler.n_dofs());

	}

	void ElasticProblem::setup_system_constraints()
	{

		constraints.clear();

		constraints.close();

		AffineConstraints<double> hanging_node_constraints;
		hanging_node_constraints.clear();
		DoFTools::make_hanging_node_constraints(dof_handler, hanging_node_constraints);
		hanging_node_constraints.close();

		constraints.merge(hanging_node_constraints, AffineConstraints<double>::MergeConflictBehavior::right_object_wins);
	}


void ElasticProblem::assemble_system_rhs()
	{
		
		system_rhs = 0.0;
		system_matrix = 0.0;
		/* if(counter == 1){
		total_force_exp = 0.0; total_force_exp2 = 0.0;  } */
		
		QGauss<1> quad_x(qx);
		QGauss<1> quad_y(qy);
		QGauss<1> quad_z(qz);
		QAnisotropic<DIM> quadrature_formula(quad_x, quad_y, quad_z);
		FEValues<DIM> fe_values(fe, quadrature_formula,
								update_values | update_gradients |
									update_quadrature_points | update_JxW_values);

		// FACE QUAD:::
		QGauss<DIM - 1> face_quadrature_formula(fe.degree + 1);
		const unsigned int n_face_q_points = face_quadrature_formula.size();
		FEFaceValues<DIM> fe_face_values(fe,
										 face_quadrature_formula,
										 //update_values | 
										 //update_gradients | 
										 //update_quadrature_points | 
										 update_normal_vectors |
											 update_JxW_values);

		unsigned int n_q_points = quadrature_formula.size();
		const unsigned int dofs_per_cell = fe.dofs_per_cell;

		FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
		Vector<double> R_local(dofs_per_cell); ////////////////////////////

		std::vector<Tensor<2, DIM>> old_solution_gradients(n_q_points);
		// std::vector<Tensor<1, DIM>> old_solution_values_face(n_face_q_points);
		// std::vector<Tensor<2, DIM>> old_solution_gradients_face(n_face_q_points);
		// std::vector<Tensor<2, DIM>> old_solution_gradients_u(n_q_points);
    	// std::vector<Tensor<2, DIM>> old_solution_grad_face(n_face_q_points);


		std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);
		// std::vector<Tensor<1,DIM>> f(n_q_points);


		const FEValuesExtractors::Vector u(0);

		typename DoFHandler<DIM>::active_cell_iterator cell = dof_handler.begin_active(),
													   endc = dof_handler.end();

		unsigned int w = 0;
		unsigned int at_cell = 0;
		for (; cell != endc; ++cell)
		{
//FOR TESTING: CELL VS FACE VALUES
         Tensor<2, DIM> F_face; F_face = 0.0; //need cell average for face values
         Tensor<4, DIM> dWdF2_face; dWdF2_face = 0.0; //need cell average for face values
         Tensor<2,DIM> dWdF_face; dWdF_face = 0.0; //need cell average for face values 


			cell_matrix = 0.0;
			R_local = 0.0;

			at_cell = cell->active_cell_index();

			fe_values.reinit(cell);
			fe_values[u].get_function_gradients(current_solution, old_solution_gradients); 
			//fe_values[u].get_function_gradients(solution_u, old_solution_gradients_u); 
			
			//	   if ((std::fabs(cell->center()(2) < upperlimit))  && (std::fabs(cell->center()(0) > lowerlimit)) )
			//	   manycell += 1;

			// Tensor<2, DIM> PKPK;
			// for(unsigned int i=0; i<DIM; ++i){
			// 	for(unsigned int j=0;j<DIM;++j){
			// 	PKPK[i][j] = 0.0;}}

			for (unsigned int q_point = 0; q_point < n_q_points; ++q_point)
			{

				Tensor<2, DIM> temptemp;
				temptemp = symmetrize(old_solution_gradients[q_point]);
				Tensor<2, DIM> PK;
				Tensor<4, DIM> hessian;
				// 1) DD OPTION:
				if (ifDD == 1)
				{
					sigma_convert_to_tensor(temptemp, PK);
					tangent_convert_to_tensor(temptemp, hessian);
						//TODO DD

					// for (unsigned int i = 0; i < DIM; ++i)
					// {
					// 	for (unsigned int j = 0; j < DIM; ++j)
					// 	{
					// 		for (unsigned int k = 0; k < DIM; ++k)
					// 		{
					// 			for (unsigned int l = 0; l < DIM; ++l)
					// 			{
					// 				hessian[i][j][k][l] = DD1->C(i * DIM + j, k * DIM + l); // tatemp(i * DIM * DIM * DIM + j * DIM * DIM + k * DIM + l);//// check thistodooooo
					// 			}
					// 		}
					// 	}
					// }
				}

				// 2) LE OPTION:
				if (ifLE == 1)
				{
					LE.get_sigma(old_solution_gradients[q_point], PK);
					hessian = LE.C;
				}


				// FOR TESTING: CELL VS FACE VALUES
				/* 				Tensor<4, DIM> hessian2;
								LE.get_dWdF2(old_solution_gradients_u[q_point], hessian2);
								Tensor<2, DIM> F2;
								F2 = 0.0;
								LE.get_F(old_solution_gradients_u[q_point], F2); */

				// FOR TESTING: CEKK VS FACE VALUES
				/* 	         for (unsigned int i = 0; i < DIM; i++)
								{
								for (unsigned int j = 0; j < DIM; j++)
								{

									F_face[i][j] += F2[i][j]/n_q_points;
									dWdF_face[i][j] += PK2[i][j]/n_q_points;
									for (unsigned int k = 0; k < DIM; k++)
									{
									for (unsigned int l = 0; l < DIM; l++)
									{
										//testing:
										dWdF2_face[i][j][k][l] += hessian2[i][j][k][l]/n_q_points;
									}}
								}}  */

				for (unsigned int n = 0; n < dofs_per_cell; ++n)
				{
					for (unsigned int m = 0; m < dofs_per_cell; ++m)
					{

						for (unsigned int i = 0; i < DIM; i++)
						{
							for (unsigned int j = 0; j < DIM; j++)
							{
								for (unsigned int k = 0; k < DIM; k++)
								{
									for (unsigned int l = 0; l < DIM; l++)
									{

										cell_matrix(n, m) -= hessian[i][j][k][l] * fe_values[u].symmetric_gradient(n, q_point)[i][j] *
															 fe_values[u].symmetric_gradient(m, q_point)[k][l] *
															 fe_values.JxW(q_point);
									}
								}
							}
						}

					} // end of m shape fun (cell matrix)

					for (unsigned int k = 0; k < DIM; k++)
					{
						for (unsigned int l = 0; l < DIM; l++)
						{
							R_local(n) -= PK[k][l] * fe_values[u].symmetric_gradient(n, q_point)[k][l] * fe_values.JxW(q_point);
						}
					}

				} // end of n shape fun

			} // end of q point iteration

			/* if (counter == 1)
			{
				for (const auto &face : cell->face_iterators())
				{
					if (face->at_boundary() && (face->boundary_id() == 11))
					{
					}

					if (face->at_boundary() && (face->boundary_id() == 1))
					{
						fe_face_values.reinit(cell, face);
						//fe_face_values[u].get_function_gradients(solution_u, old_solution_gradients_face);

						for (unsigned int q_point = 0; q_point < n_face_q_points; ++q_point)
						{
							//Tensor<2, DIM> sigma_face;
							//LE.get_dWdF(old_solution_gradients_face[q_point], sigma_face);
							Tensor<1, DIM> normalvector;
							normalvector = fe_face_values.normal_vector(q_point);
							

							for (unsigned int i = 0; i < DIM; i++)
							{
								for (unsigned int j = 0; j < DIM; j++)
								{
									//total_force_exp += (i == 0) ? sigma_face[i][j] * normalvector[j] * fe_face_values.JxW(q_point) : 0.0;
									total_force_exp2 += (i == 0) ? PKPK[i][j] * normalvector[j] * fe_face_values.JxW(q_point) : 0.0;

									//if(at_cell==11430 && i==0 && j==0){
									//std::cout<<"face "<<sigma_face[i][j]<<" cell "<<PKPK[i][j]<<std::endl;}
								}
							}
						}
					}

				}
			} */
			
			cell->get_dof_indices(local_dof_indices);

			for (unsigned int n = 0; n < dofs_per_cell; ++n)
			{
				for (unsigned int m = 0; m < dofs_per_cell; ++m)
				{
					system_matrix.add(local_dof_indices[n],
									  local_dof_indices[m],
									  cell_matrix(n, m));
				}
				system_rhs(local_dof_indices[n]) -= R_local(n);
			}

		} // end of cell iteration

		// std::cout <<" cell forces exp "<< total_force_exp2 << std::endl;

	} // end of assemble system



  void ElasticProblem::assemble_system_rhs_parallel()
  {	
    QGauss<1> quad_x(qx);
    QGauss<1> quad_y(qy);
    QGauss<1> quad_z(qz);
    QAnisotropic<DIM> quadrature_formula(quad_x, quad_y, quad_z);

    system_rhs = 0.0; system_matrix = 0.0;
	  WorkStream::run(dof_handler.begin_active(),
			  dof_handler.end(),
			  *this,
			  &ElasticProblem::local_assemble_system_rhs,
			  &ElasticProblem::copy_local_to_global,
			  AssemblyScratchData(fe, quadrature_formula),
			  AssemblyCopyData());
  }// end of assemble system

  ElasticProblem::AssemblyScratchData::
  AssemblyScratchData (const FiniteElement<DIM> &fe, Quadrature<DIM> &quad)
    :
    fe_values (fe, quad,
               update_gradients |
               update_JxW_values)
  {}

  ElasticProblem::AssemblyScratchData::
  AssemblyScratchData (const AssemblyScratchData &scratch_data)
    :
    fe_values (scratch_data.fe_values.get_fe(),
               scratch_data.fe_values.get_quadrature(),
               update_gradients |
               update_JxW_values)
  {}

void ElasticProblem::local_assemble_system_rhs(
	const typename DoFHandler<DIM>::active_cell_iterator &cell,
	AssemblyScratchData                                  &scratch,
	AssemblyCopyData                                     &copy_data)
  {

	    const unsigned int dofs_per_cell   = fe.dofs_per_cell;
	    const unsigned int n_q_points      = scratch.fe_values.get_quadrature().size();

      FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
      Vector<double> R_local(dofs_per_cell); 
      cell_matrix = 0.0;
      R_local = 0.0;

	    scratch.fe_values.reinit (cell);

	    const FEValuesExtractors::Vector u(0);
      std::vector<Tensor<2, DIM>> old_solution_gradients(n_q_points);
      scratch.fe_values[u].get_function_gradients(current_solution, old_solution_gradients); 


	    unsigned int  at_cell = cell -> active_cell_index();
// BEGIN QUADRATURE ITERATION FOR THE CELL: --------------------------------------------------------------------------------------------------

	  for (unsigned int q_point=0; q_point<n_q_points; ++q_point)
      {
		Tensor<2, DIM> temptemp; temptemp = symmetrize(old_solution_gradients[q_point]);
		Tensor<2, DIM> PK; Tensor<4, DIM> hessian;
		// 1) DD OPTION:
		if(ifDD == 1){
        sigma_convert_to_tensor(temptemp, PK);
        tangent_convert_to_tensor(temptemp, hessian); }

		// 2) LE OPTION:
		if(ifLE == 1){
        LE.get_sigma(old_solution_gradients[q_point], PK);
        hessian = LE.C; }


//output database stress strain;
if(get_data == 1){
		if(q_point == 0 && timestep_number > 2){
				for (const auto &face20 : cell->face_iterators())
				{
					if (face20->at_boundary() && (face20->boundary_id() == 11)){
						unsigned int cindex = cell_indices.at(at_cell);
						for(unsigned int i = 0; i < DIM; ++i){
							for(unsigned int j = 0; j < DIM; ++j){
								strain_dbs.at(cindex)[i*DIM + j] = symmetrize(old_solution_gradients[q_point])[i][j];
								stress_dbs.at(cindex)[i*DIM + j] = PK[i][j];
							}
						}

						FILE* out2;
						out2 = std::fopen(filename2.c_str(), "a");
						fprintf(out2, "%lg %lg %lg %lg %lg %lg %lg %lg %lg    %lg %lg %lg %lg %lg %lg %lg %lg %lg \n",
								strain_dbs.at(cindex)[0], strain_dbs.at(cindex)[1], strain_dbs.at(cindex)[2], 
								strain_dbs.at(cindex)[3], strain_dbs.at(cindex)[4], strain_dbs.at(cindex)[5], 
								strain_dbs.at(cindex)[6], strain_dbs.at(cindex)[7], strain_dbs.at(cindex)[8],
								stress_dbs.at(cindex)[0], stress_dbs.at(cindex)[1], stress_dbs.at(cindex)[2],
								stress_dbs.at(cindex)[3], stress_dbs.at(cindex)[4], stress_dbs.at(cindex)[5],
								stress_dbs.at(cindex)[6], stress_dbs.at(cindex)[7], stress_dbs.at(cindex)[8]);
						fclose(out2);
					}
				}
		}
}

        for (unsigned int n = 0; n < dofs_per_cell; ++n)
        {

          for (unsigned int m = 0; m < dofs_per_cell; ++m)
          {
            for (unsigned int i = 0; i < DIM; i++)
            {
              for (unsigned int j = 0; j < DIM; j++)
              {
                for (unsigned int k = 0; k < DIM; k++)
                {
                  for (unsigned int l = 0; l < DIM; l++)
                  {
                    cell_matrix(n, m) -= hessian[i][j][k][l] * scratch.fe_values[u].symmetric_gradient(n, q_point)[i][j] *
                                         scratch.fe_values[u].symmetric_gradient(m, q_point)[k][l] *
                                         scratch.fe_values.JxW(q_point);
                  }
                }
              }
            }

          } // end of m shape fun (cell matrix)

          for (unsigned int k = 0; k < DIM; k++)
          {
            for (unsigned int l = 0; l < DIM; l++)
            {
              R_local(n) -= PK[k][l] * scratch.fe_values[u].symmetric_gradient(n, q_point)[k][l] * scratch.fe_values.JxW(q_point);
            }
          }

        } // end of n shape fun

      }// end of q point iteration

// END QUADRATURE ITERATION FOR THE CELL: --------------------------------------------------------------------------------------------------


	      copy_data.local_dof_indices.resize(dofs_per_cell);

	      cell->get_dof_indices (copy_data.local_dof_indices);
	      copy_data.R_local = R_local;
	      copy_data.cell_matrix = cell_matrix;

  } //end of local assemble system


  void ElasticProblem::copy_local_to_global(const AssemblyCopyData &copy_data)
  {
        for (unsigned int i = 0; i < copy_data.local_dof_indices.size(); ++i)
        {
        	system_rhs[copy_data.local_dof_indices[i]] -= copy_data.R_local[i];

          for (unsigned int j = 0; j < copy_data.local_dof_indices.size(); ++j)
          {
            system_matrix.add(copy_data.local_dof_indices[i],
                              copy_data.local_dof_indices[j],
                              copy_data.cell_matrix[i][j]);
          }
        }
  } //end of copy local to global




void ElasticProblem::copy_local_to_global_NOTHING(const IntvarCopyData &copy_data){
  //do nothing
}


void ElasticProblem::calculate_scalar_value()
  {
    scalarvalue = 0.0;

    QGauss<1> quad_x(qx);
    QGauss<1> quad_y(qy);
    QGauss<1> quad_z(qz);
    QAnisotropic<DIM> quadrature_formula(quad_x, quad_y, quad_z);

    QGauss<DIM - 1> face_quadrature_formula(fe.degree + 1);

	  WorkStream::run(dof_handler.begin_active(),
			  dof_handler.end(),
			  *this,
			  &ElasticProblem::local_scalar_value,
			  &ElasticProblem::copy_local_to_global_NOTHING,    //passing a void function
			  AssemblyScratchData_adj(fe, quadrature_formula,face_quadrature_formula),
			  IntvarCopyData());
  }

  ElasticProblem::AssemblyScratchData_adj::
  AssemblyScratchData_adj (const FiniteElement<DIM> &fe, Quadrature<DIM> &quad, QGauss<DIM-1> &quad_face)
    :
    fe_values (fe, quad,
               update_gradients |
               update_JxW_values),

    fe_face_values (fe, quad_face,
                    update_values   | 
                    update_gradients |
                    update_normal_vectors | 
                    update_JxW_values)
  {}

  ElasticProblem::AssemblyScratchData_adj::
  AssemblyScratchData_adj (const AssemblyScratchData_adj &scratch_data)
    :
    fe_values (scratch_data.fe_values.get_fe(),
               scratch_data.fe_values.get_quadrature(),
               update_gradients |
               update_JxW_values),

    fe_face_values (scratch_data.fe_face_values.get_fe(),
                    scratch_data.fe_face_values.get_quadrature(),
                    update_values   | 
                    update_gradients |
                    update_normal_vectors | 
                    update_JxW_values)               
  {}



void ElasticProblem::local_scalar_value( 
  const typename DoFHandler<DIM>::active_cell_iterator &cell,
  AssemblyScratchData_adj                                  &scratch,
  IntvarCopyData                                     &copy_data)
  {
    solution_ut = solution_u; //////////////////// TODO: PROCEED WITH CAUTION. BEWARE OF THE VARIABLE HERE

    const unsigned int dofs_per_cell   = fe.dofs_per_cell;
    const unsigned int n_q_points      = scratch.fe_values.get_quadrature().size();
    const unsigned int n_face_q_points = scratch.fe_face_values.get_quadrature().size();

    const FEValuesExtractors::Vector u(0);
    std::vector<Tensor<2, DIM>> old_solution_gradients(n_q_points);
    std::vector<Tensor<2, DIM>> old_solution_gradients_face(n_face_q_points);      
    
    unsigned int at_cell2 = cell->active_cell_index();
    
      Tensor<2, DIM> dWdF_face;
      dWdF_face = 0.0; // need cell average for face values
      
      scratch.fe_values.reinit(cell);
      scratch.fe_values[u].get_function_gradients(solution_ut, old_solution_gradients); /// need to get the solution of the forward problem
      for (unsigned int q_point = 0; q_point < n_q_points; ++q_point)
      {
        Tensor<2, DIM> dWdF;

		// 1) DD option:
		if(ifDD == 1)
        sigma_convert_to_tensor(old_solution_gradients[q_point], dWdF);

		// 2) LE option:
		if(ifLE == 1)
        LE.get_sigma(old_solution_gradients[q_point], dWdF);

        for (unsigned int i = 0; i < DIM; i++)
        {
          for (unsigned int j = 0; j < DIM; j++)
          {
            dWdF_face[i][j] += dWdF[i][j] / n_q_points; // need cell average for face values
          }
        }
      } //end of q point iteration in the cell

        for (const auto &face1 : cell->face_iterators())
        {

          if (face1->at_boundary() && (face1->boundary_id() == 1))
          {
            scratch.fe_face_values.reinit(cell, face1);
            scratch.fe_face_values[u].get_function_gradients(solution_ut, old_solution_gradients_face);

            for (unsigned int q_point = 0; q_point < n_face_q_points; ++q_point)
            {
              Tensor<2,DIM> dWdF;
              // 1) Option 1: DD
			  if(ifDD == 1)
			  sigma_convert_to_tensor(old_solution_gradients[q_point], dWdF);
              // 2) Option 2: LE
			  if(ifLE == 1)
			  LE.get_sigma(old_solution_gradients_face[q_point],dWdF);
              
			  Tensor<1, DIM> normalvect;
              normalvect = scratch.fe_face_values.normal_vector(q_point);

              for (unsigned int i = 0; i < DIM; i++)
              {
                for (unsigned int j = 0; j < DIM; j++)
                {
// TODO 2: CELL, THEN FACE
                  scalarvalue += (i == 0) ? 2 * (dWdF_face[i][j] * normalvect[j] * scratch.fe_face_values.JxW(q_point)) : 0.0; //cell avf for face values
                  //scalarvalue += (i == 0) ? 2 * (dWdF[i][j] * normalvect[j] * fe_face_values.JxW(q_point)) : 0.0; //pure face values                  
                }
              }
            }
          }
        } //end of face iterator in the cell

  
  } //end of local assemble system to get scalar value


	void ElasticProblem::apply_boundaries_and_constraints() ////////////////// TODO DD /////////////////////////
	{
		constraints.condense(system_matrix);
		constraints.condense(system_rhs);

		std::map<types::global_dof_index, double> boundary_values;

// TODOOOOOOOOOOOOOOOOOOOOOOOOOOO DIFF BC!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!1
  		std::vector<bool> encastre = {true, true, true};
		ComponentMask encastre_mask(encastre);
		VectorTools::interpolate_boundary_values(dof_handler,
												 0,
												 ZeroFunction<DIM, double>(DIM),
												 boundary_values,
												 encastre_mask);
												 
 		VectorTools::interpolate_boundary_values(dof_handler,
												 1,
												 ZeroFunction<DIM, double>(DIM), //IncrementalBoundaryValuesDISP<DIM>(time, timestep_number, time_step),
												 boundary_values,
												 encastre_mask);   

		MatrixTools::apply_boundary_values(boundary_values,
										   system_matrix,
										   newton_update,
										   system_rhs);
	}

	void ElasticProblem::solve()
	{
		std::cout<< "norm "<<system_rhs.l1_norm()<<std::endl;
		if(timestep_number == 1){
			newton_update = 0.0;
		}

		else if(system_rhs.l1_norm() == 0){
			newton_update = 0.0;
		}

		else{
			if (dof_handler.n_dofs() < 10000)
			{
				std::cout << "Direct" << std::endl;
				std::cout<<"system rhs :"<<system_rhs<<std::endl;
				std::cout<<"newton update :"<<newton_update<<std::endl;
				SparseDirectUMFPACK A_direct;
				A_direct.initialize(system_matrix);
				A_direct.vmult(newton_update, system_rhs);
				std::cout<<"newton update 2: "<<std::endl<<newton_update<<std::endl;
			}
			else
			{
				SolverControl solver_control(dof_handler.n_dofs(), 1e-11);
				SolverCG<> solver(solver_control);

				PreconditionSSOR<SparseMatrix<double>> preconditioner;
				preconditioner.initialize(system_matrix, 1.2);

				solver.solve(system_matrix, newton_update, system_rhs, preconditioner);
			}

			constraints.distribute(newton_update);
			current_solution += newton_update;
			std::cout<<"current solution "<<std::endl<< current_solution<<std::endl;
		}
	}

	void ElasticProblem::propagate_u() 
	{
		solution_u = current_solution;
		setup_system(false);
	}

	double ElasticProblem::compute_residual() // if newton update is done
	{
		Vector<double> residual(dof_handler.n_dofs());
		residual = 0.0;

		Vector<double> eval_point(dof_handler.n_dofs());
		eval_point = current_solution;

		QGauss<1> quad_x(qx);
		QGauss<1> quad_y(qy);
		QGauss<1> quad_z(qz);
		QAnisotropic<DIM> quadrature_formula(quad_x, quad_y, quad_z);
		FEValues<DIM> fe_values(fe, quadrature_formula,
								update_values | update_gradients |
									update_quadrature_points | update_JxW_values);

		unsigned int n_q_points = quadrature_formula.size();
		const unsigned int dofs_per_cell = fe.dofs_per_cell;

		Vector<double> cell_residual(dofs_per_cell);

		std::vector<Tensor<2, DIM>> old_solution_gradients(n_q_points);


		std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

		const FEValuesExtractors::Vector u(0);

		typename DoFHandler<DIM>::active_cell_iterator cell = dof_handler.begin_active(),
													   endc = dof_handler.end();

		unsigned int at_cell = 0;
		for (; cell != endc; ++cell)
		{

			cell_residual = 0.0;
			at_cell = cell->active_cell_index();

			fe_values.reinit(cell);
			fe_values[u].get_function_gradients(eval_point, old_solution_gradients); //////////////////////////////////
			
			for (unsigned int q_point = 0; q_point < n_q_points; ++q_point)
			{

				Tensor<2, DIM> PKres;
				// 1) DD option:
				if(ifDD == 1)
				sigma_convert_to_tensor(old_solution_gradients[q_point], PKres);
				// 2) LE option:
				if(ifLE == 1)
				LE.get_sigma(old_solution_gradients[q_point], PKres);

				for (unsigned int n = 0; n < dofs_per_cell; ++n)
				{

					for (unsigned int k = 0; k < DIM; k++)
					{
						for (unsigned int l = 0; l < DIM; l++)
						{
							cell_residual(n) -= PKres[k][l] * fe_values[u].symmetric_gradient(n, q_point)[k][l] * fe_values.JxW(q_point);
						}
					}

				} // end of n shape fun

			} // end of q point iteration

			cell->get_dof_indices(local_dof_indices);

			for (unsigned int n = 0; n < dofs_per_cell; ++n)
				residual(local_dof_indices[n]) += cell_residual(n);

		} // end of cell iteration

		constraints.condense(residual);

		std::vector<bool> encastre = {true, true, true};
		ComponentMask encastre_mask(encastre);

		for (types::global_dof_index i :
			 DoFTools::extract_boundary_dofs(dof_handler, encastre_mask, {0, 1}))
			residual(i) = 0.0;
		return residual.l2_norm();
	}


  void ElasticProblem::assemble_system_residual()
  {

	residual = 0.0;

    QGauss<1> quad_x(qx);
    QGauss<1> quad_y(qy);
    QGauss<1> quad_z(qz);
    QAnisotropic<DIM> quadrature_formula(quad_x, quad_y, quad_z);

	  WorkStream::run(dof_handler.begin_active(),
			  dof_handler.end(),
			  *this,
			  &ElasticProblem::compute_residual_parallel,
			  &ElasticProblem::copy_local_to_global_res,
			  AssemblyScratchData(fe, quadrature_formula), //already defined above...
			  AssemblyCopyData_res());
  }// end of assemble system

  
 void ElasticProblem::compute_residual_parallel(
 	const typename DoFHandler<DIM>::active_cell_iterator &cell,
 	AssemblyScratchData                                  &scratch,
 	AssemblyCopyData_res                                 &copy_data)
   {

 	    const unsigned int dofs_per_cell   = fe.dofs_per_cell;
 	    const unsigned int n_q_points      = scratch.fe_values.get_quadrature().size();

 	    Vector<double> cell_residual(dofs_per_cell);
 	    cell_residual = 0.0;

 	    scratch.fe_values.reinit (cell);

 	    std::vector<Tensor<2,DIM>> old_solution_gradients(n_q_points);

 	    const FEValuesExtractors::Vector u(0);
 	    scratch.fe_values[u].get_function_gradients(current_solution,old_solution_gradients);

 	    unsigned int  at_cell = cell -> active_cell_index();


 	    for (unsigned int q_point=0; q_point<n_q_points; ++q_point)
      {
        Tensor<2, DIM> PKres;
		// 1) DD option:
		if(ifDD == 1)
        sigma_convert_to_tensor(old_solution_gradients[q_point], PKres);
		// 2) LE option:
		if(ifLE == 1)
        LE.get_sigma(old_solution_gradients[q_point], PKres);


        for (unsigned int n = 0; n < dofs_per_cell; ++n)
        {

          for (unsigned int k = 0; k < DIM; k++)
          {
            for (unsigned int l = 0; l < DIM; l++)
            {
              cell_residual(n) -= PKres[k][l] * scratch.fe_values[u].symmetric_gradient(n, q_point)[k][l] * scratch.fe_values.JxW(q_point);
            }
          }

        } // end of n shape fun
      }

  	   // END QUADRATURE ITERATION FOR THE CELL: --------------------------------------------------------------------------------------------------

 	    copy_data.local_dof_indices.resize(dofs_per_cell);

 	    cell->get_dof_indices (copy_data.local_dof_indices);
 	    copy_data.cell_residual = cell_residual;

   }// end of local assemble systems

  void ElasticProblem::copy_local_to_global_res(const AssemblyCopyData_res &copy_data)
  {
	    /* Takes the elemental contributions to the matrix, and adds them up in a thread-secture dealii way. */
	    for (unsigned int i=0; i<copy_data.local_dof_indices.size(); ++i){
	    	residual[copy_data.local_dof_indices[i]] += copy_data.cell_residual[i];
	    }
  }

  double ElasticProblem::get_residual_parallel()
  {
      constraints.condense(residual);

//TODO 1: DIFF BC!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!1
    std::vector<bool> encastre = {true, true, true};
    ComponentMask encastre_mask(encastre);

    for (types::global_dof_index i :
         DoFTools::extract_boundary_dofs(dof_handler, encastre_mask, {0, 1}))
      residual(i) = 0.0;

    return residual.l2_norm();
    residual = 0.0;

  }



	void ElasticProblem::output_forces()
	{

		std::string filename1(output_directory);
		filename1 += "/forces";
		filename1 += vary;

		struct stat st;
		if (stat(filename1.c_str(), &st) == -1)
			mkdir(filename1.c_str(), 0700);

		filename1 += "/forces";
		filename1 += vary;
		filename1 += "-";
		filename1 += std::to_string(timestep_number);
		filename1 += ".out";
		FILE *out1;
		out1 = std::fopen(filename1.c_str(), "w");
		fclose(out1);

				out1 = std::fopen(filename1.c_str(), "a");
				fprintf(out1, "%lg %lg \n",
						time, scalarvalue*0.5);
				fclose(out1);
	}


	void ElasticProblem::output_results(const unsigned int cycle) const
	{

		std::vector<std::string> solution_names;
		switch (DIM)
		{
		case 1:
			solution_names.push_back("displacement");
			break;
		case 2:
			solution_names.push_back("x1_displacement");
			solution_names.push_back("x2_displacement");
			break;
		case 3:
			solution_names.push_back("x1_displacement");
			solution_names.push_back("x2_displacement");
			solution_names.push_back("x3_displacement");
			break;
		default:
			Assert(false, ExcNotImplemented());
			break;
		}

		std::vector<std::string> solution_names_speck;
		switch (DIM)
		{
		case 1:
			solution_names_speck.push_back("speck");
			break;
		case 2:
			solution_names_speck.push_back("x1_speck");
			solution_names_speck.push_back("x2_speck");
			break;
		case 3:
			solution_names_speck.push_back("x1_speck");
			solution_names_speck.push_back("x2_speck");
			solution_names_speck.push_back("x3_speck");
			break;
		default:
			Assert(false, ExcNotImplemented());
			break;
		}


		std::vector<DataComponentInterpretation::DataComponentInterpretation>
			interpretation(DIM,
						   DataComponentInterpretation::component_is_part_of_vector);

		std::string filename0(output_directory);

		filename0 += "/lagrangian_solution";
		filename0 += vary;

		// see if the directory exists...
		struct stat st;
		if (stat(filename0.c_str(), &st) == -1)
			mkdir(filename0.c_str(), 0700);

		filename0 += "/lagrangian_solution_holedic";
		filename0 += vary;
		filename0 += "-";
		filename0 += std::to_string(cycle);
		filename0 += ".vtk";
		std::ofstream output_lagrangian_solution(filename0.c_str());

		DataOut<DIM> data_out_lagrangian;

		data_out_lagrangian.add_data_vector(dof_handler,
											solution_u,
											solution_names,
											interpretation);

		data_out_lagrangian.add_data_vector(dof_handler,
											solution_g,
											solution_names_speck,
											interpretation);

		data_out_lagrangian.build_patches();
		data_out_lagrangian.write_vtk(output_lagrangian_solution);
	}

	void ElasticProblem::initiate_guess() // TODOOOOOOOOOOOOOOOOOOOOOOOOOOO DIFF BC!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!1
	{
			std::vector<bool> side_x = {true, false, false};
			ComponentMask side_x_mask(side_x);
			DoFTools::extract_boundary_dofs(dof_handler,
											side_x_mask,
											selected_dofs_x,
											{1});

			for (unsigned int n = 0; n < dof_handler.n_dofs(); ++n)
			{
				if (selected_dofs_x[n]){
					current_solution[n] = (time - deltat) * velocity_qs;// * (support_points[n][1]/domain_dimensions[1]); //disp bc varying wrt y position 
				}
			}

			std::vector<bool> side_yz = {false, true, true};
			ComponentMask side_yz_mask(side_yz);
			DoFTools::extract_boundary_dofs(dof_handler,
											side_yz_mask,
											selected_dofs_yz,
											{1});

			for (unsigned int n = 0; n < dof_handler.n_dofs(); ++n)
			{
				if (selected_dofs_yz[n])
					current_solution[n] = 0.0;
			}   

	}

	void ElasticProblem::initiate_guess_DD(const bool counter) // TODO
	{
		if(counter==1){

			if(timestep_number==1){
				current_solution = 0.0;
			}

			else{
				std::vector<bool> side_x = {true, false, false};
				ComponentMask side_x_mask(side_x);
				IndexSet side_x_set = DoFTools::extract_dofs(dof_handler, side_x_mask);

				std::vector<bool> side_y = {false, true, false};
				ComponentMask side_y_mask(side_y);
				IndexSet side_y_set = DoFTools::extract_dofs(dof_handler, side_y_mask);

				std::vector<bool> side_z = {false, false, true};
				ComponentMask side_z_mask(side_z);
				IndexSet side_z_set = DoFTools::extract_dofs(dof_handler, side_z_mask);

				for (unsigned int n = 0; n < dof_handler.n_dofs(); ++n){
					if (side_x_set.is_element(n)){
						double xcoord = support_points_all_dofs[n][0];
						current_solution[n] = (time - time_step) * velocity_qs * xcoord/L0;	
					}
					if (side_y_set.is_element(n)){
						double ycoord = support_points_all_dofs[n][1];
						current_solution[n] = 0.001 * (time - time_step) * velocity_qs * (-ycoord+yheight/2.0)/yheight;
					}
					if (side_z_set.is_element(n)){
						double zcoord = support_points_all_dofs[n][2];
						current_solution[n] = 0.0001 * (time - time_step) * velocity_qs * (-zcoord+zheight/2.0)/zheight;
					}
				}

	//take care of boundary dofs from dirichlet boundary condition
				DoFTools::extract_boundary_dofs(dof_handler,
												side_x_mask,
												selected_dofs_x,
												{1});

				for (unsigned int n = 0; n < dof_handler.n_dofs(); ++n)
				{
					if (selected_dofs_x[n]){
						current_solution[n] = (time - time_step) * velocity_qs;
					}
				}

				std::vector<bool> side_yz = {false, true, true};
				ComponentMask side_yz_mask(side_yz);
				DoFTools::extract_boundary_dofs(dof_handler,
												side_yz_mask,
												selected_dofs_yz,
												{0,1});

				for (unsigned int n = 0; n < dof_handler.n_dofs(); ++n)
				{
					if (selected_dofs_yz[n])
						current_solution[n] = 0.0;
				} 
			}  
			std::cout<<"current solution "<<current_solution <<std::endl;
		}

		else{
			/* insert code*/
		}
	}


	void ElasticProblem::calculate_end_disp(Vector<double> &soln)
	{
			  std::vector<bool> side_x = {true, false, false};
			  ComponentMask side_x_mask(side_x);
			  DoFTools::extract_boundary_dofs (dof_handler,
					  side_x_mask,
					  selected_dofs_x,
					  {1});

		for (unsigned int n = 0; n < dof_handler.n_dofs(); ++n)
		{
			if (selected_dofs_x[n])
			{
				end_disp = soln[n]; // current_solution[n];
				break;
			}
		}
		std::cout << "end disp " << end_disp << " , prescribed disp " << deltat * (timestep_number - 1) * velocity_qs << std::endl;
	}

	void ElasticProblem::run_forward_problem() //////////////////////////////
	{

		if(HOLE==1){create_mesh_hole();}
		if(RECT==1){create_mesh();}
		setup_system(true); 
		output_results(0); 

		std::string filename4(output_directory);
	  	filename4 += "/forces";
	  	filename4 += vary;
	  	filename4 += ".out";	  
		FILE* out4;
	  	out4 = std::fopen(filename4.c_str(), "w");
	  	fclose(out4);

		if(get_data == 1){
			// std::string filename2(output_directory);
			filename2 = output_directory;
			filename2 += "/DBs_stress_strain_average";
			filename2 += vary;
			filename2 += ".out";
			FILE *out2;
			out2 = std::fopen(filename2.c_str(), "w");
			fclose(out2);
		}

		std::cout<<"in forward problem ===== " << vary<<std::endl;

		for (; time <= time_step * endsteptime; time += time_step, ++timestep_number)
		{

			std::cout << "time step " << timestep_number << " at t= " << time << "   " << "--------------------------------------------------------" << std::endl;

			double last_residual_norm = std::numeric_limits<double>::max();
			counter = 1;

			while ((last_residual_norm > 1.0e-7) && (counter < 10))
			{ 
			converge = 0;
				
				if (counter == 1)
				{
					initiate_guess();
				}
				assemble_system_rhs_parallel();
				apply_boundaries_and_constraints();
				solve();

				assemble_system_residual();
				last_residual_norm = get_residual_parallel();
				std::cout << " Iteration : " << counter << "  Residual : " << last_residual_norm << std::endl;

				setup_system(false);
				++counter;
			} 

			propagate_u();
			calculate_scalar_value(); //calculate force at the boundary
			converge = 1;
			output_forces();
			calculate_end_disp(current_solution);

			out4 = std::fopen(filename4.c_str(), "a");
			fprintf(out4, "%lg %lg \n",
					time, scalarvalue*0.5);
			fclose(out4);

			unsigned int step_out = 1;
			if (timestep_number % step_out == 0)
			{
				output_results(42069 + (timestep_number));

			}
		}
	}

void ElasticProblem::run_forward_problem_2() ////////////////////////////// TODO DD
	{

		if(HOLE==1){create_mesh_hole();}
		if(RECT==1){create_mesh();}
		setup_system(true); 
		output_results(0); 

		// std::string filename4(output_directory);
	  	// filename4 += "/forces";
	  	// filename4 += vary;
	  	// filename4 += ".out";	  
		// FILE* out4;
	  	// out4 = std::fopen(filename4.c_str(), "w");
	  	// fclose(out4);

		if(get_data == 1){
			// std::string filename2(output_directory);
			filename2 = output_directory;
			filename2 += "/DBs_stress_strain_average";
			filename2 += vary;
			filename2 += ".out";
			FILE *out2;
			out2 = std::fopen(filename2.c_str(), "w");
			fclose(out2);
		}

		std::cout<<"in forward problem ===== " << vary<<std::endl;

		for (; time <= time_step * endsteptime; time += time_step, ++timestep_number)
		{

			std::cout << "time step " << timestep_number << " at t= " << time << "   " << "--------------------------------------------------------" << std::endl;

			double last_residual_norm = std::numeric_limits<double>::max();
			counter = 1;

			while ((last_residual_norm > 1.0e-7) && (counter < 10))
			{ 
			converge = 0;
				
				if (counter == 1)
				{
					//initiate_guess();
					if(ifDD==1)
					initiate_guess_DD(true); //TODO: INITIAL GUESS OF THE STRAIN
					
					if(ifLE==1)
					current_solution = 0.0;
				}
				// else{
				// 	if(ifDD==1)
				// 	initiate_guess_DD(false);
				// }
				assemble_system_rhs();
				//assemble_system_rhs_parallel();
				apply_boundaries_and_constraints();
				solve();
				std::cout<<"solved"<<std::endl;
				
				if(timestep_number == 1){
					last_residual_norm = 0.0;
				}

				if(timestep_number > 1){
					last_residual_norm = compute_residual();
					//assemble_system_residual();
					//last_residual_norm = get_residual_parallel();
					std::cout << " Iteration : " << counter << "  Residual : " << last_residual_norm << std::endl;
				}

				setup_system(false);
				++counter;
			} 

			propagate_u();
			//calculate_scalar_value(); //calculate force at the boundary
			converge = 1;
			//output_forces();
			//calculate_end_disp(current_solution);

			// out4 = std::fopen(filename4.c_str(), "a");
			// fprintf(out4, "%lg %lg \n",
			// 		time, scalarvalue*0.5);
			// fclose(out4);

			unsigned int step_out = 1;
			if (timestep_number % step_out == 0)
			{
				output_results(42069 + (timestep_number));

			}
		}
	}

void ElasticProblem::read_input_file_DIC_exp_speckle()  //TODO 30 CHANGE SYNTHETIC DATA
  { 
	std::string filename2;
	unsigned int maxread;

	//TODO 50 : DIFFERENT PROBLEM HOLE
	if (HOLE == 1)
	{
		maxread = 3500; //(grid_dimensions[0] + 1)*(grid_dimensions[1]+1); //3500;
		speckle.resize(maxread);
		filename2 = "specklereal_speckle10_rect_expanded"; //"specklereal_speckle8"
		filename2 += ".out";
	}

	if (RECT == 1)
	{
		maxread = (grid_dimensions[0] + 1)*(grid_dimensions[1]+1); //3500;
		speckle.resize(maxread);
		filename2 = "specklereal_speckle10_rect_expanded"; //"specklereal_speckle8_rect"; 
		filename2 += ".out";
	}

	  FILE *fid;
      int endOfFileFlag;
      char nextLine[MAXLINE];

      int valuesWritten;
      bool fileReadErrorFlag = false;

      fid = std::fopen(filename2.c_str(), "r");
      if (fid == NULL)
      {
        std::cout << "Unable to open file DIC \"" << filename2 << "\"" << std::endl;
        fileReadErrorFlag = true;
      }

      else
      {
         for(unsigned int index = 0; index < maxread; ++index){ //index is looping through all number of cells
          getNextDataLine(fid, nextLine, MAXLINE, &endOfFileFlag);
          if (endOfFileFlag == 0){
          valuesWritten = sscanf(nextLine, "%lg ", 
          &speckle[index]); //////////////
          }
        } 

      }

      if (fileReadErrorFlag)
      {
        std::cout << "Error reading input file DIC, Exiting.\n"
                  << std::endl;
        exit(1);
      }
    }  


	void ElasticProblem::read_input_file(char *filename)
	{
		FILE *fid;
		int endOfFileFlag;
		char nextLine[MAXLINE];

		int valuesWritten;
		bool fileReadErrorFlag = false;

		grid_dimensions.resize(DIM);
		domain_dimensions.resize(DIM);

		fid = std::fopen(filename, "r");
		if (fid == NULL)
		{
			std::cout << "Unable to open file \"" << filename << "\"" << std::endl;
			fileReadErrorFlag = true;
		}
		else
		{

			// Read in the output name
			char directory_name[MAXLINE];
			getNextDataLine(fid, nextLine, MAXLINE, &endOfFileFlag);
			valuesWritten = sscanf(nextLine, "%s", directory_name);
			if (valuesWritten != 1)
			{
				fileReadErrorFlag = true;
				goto fileClose;
			}

			sprintf(output_directory, "output/");
			strcat(output_directory, directory_name);

			// Read in the grid dimensions
			getNextDataLine(fid, nextLine, MAXLINE, &endOfFileFlag);
			valuesWritten = sscanf(nextLine, "%u %u %u", &grid_dimensions[0], &grid_dimensions[1], &grid_dimensions[2]); //////////////
			// if (valuesWritten != 3)																						 //////////////////////////////////////////////////////////////////////////////////////////////
			// {
			// 	fileReadErrorFlag = true;
			// 	goto fileClose;
			// }

			// Read in the domain dimensions
			getNextDataLine(fid, nextLine, MAXLINE, &endOfFileFlag);
			valuesWritten = sscanf(nextLine, "%lg %lg %lg", &domain_dimensions[0], &domain_dimensions[1], &domain_dimensions[2]); ////////
			// if (valuesWritten != 3)																								  ///////////////////////////////////////////////////////////////////////////////////////////////
			// {
			// 	fileReadErrorFlag = true;
			// 	goto fileClose;
			// }

			// read in the number of guass points in the x and y direction
			getNextDataLine(fid, nextLine, MAXLINE, &endOfFileFlag);
			valuesWritten = sscanf(nextLine, "%u  %u %u", &qx, &qy, &qz);
			// if (valuesWritten != 3)
			// {
			// 	fileReadErrorFlag = true;
			// 	goto fileClose;
			// }

		fileClose:
		{
			fclose(fid);
		}
		}

		if (fileReadErrorFlag)
		{
			// default parameter values
			std::cout << "Error reading input file, Exiting.\n"
					  << std::endl;
			exit(1);
		}
		else
			std::cout << "Input file successfully read" << std::endl;

		// make the output directory
		struct stat st;
		if (stat("./output", &st) == -1)
			mkdir("./output", 0700);

		if (stat(output_directory, &st) == -1)
			mkdir(output_directory, 0700);
	}


	void ElasticProblem::getNextDataLine(FILE *const filePtr, char *nextLinePtr,
										 int const maxSize, int *const endOfFileFlag)
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

}

#endif // LINEAR_ELASTIC_CC_
