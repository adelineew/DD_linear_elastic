/*
 *
 *
 *  Created on: October 3, 2022
 *      Author: Adeline wihardja
 */

#ifndef LINEAR_ELASTIC_H_
#define LINEAR_ELASTIC_H_

#include <deal.II/base/conditional_ostream.h>

#include <deal.II/distributed/grid_refinement.h>

#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function.h>
#include <deal.II/base/tensor.h>

//#include <deal.II/lac/constraint_matrix.h>

#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/precondition.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_refinement.h>
//#include <deal.II/grid/tria_boundary.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/grid_in.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/dofs/dof_renumbering.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_system.h>

  #include <deal.II/grid/tria.h>
  #include <deal.II/grid/grid_generator.h>

#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/error_estimator.h>

#include <deal.II/fe/mapping_q_eulerian.h>

#include <deal.II/lac/sparse_direct.h>
#include <deal.II/lac/solver_minres.h>
#include <deal.II/lac/lapack_full_matrix.h>
#include <deal.II/lac/arpack_solver.h>



#include <deal.II/lac/trilinos_sparse_matrix.h>

#include <deal.II/lac/petsc_sparse_matrix.h>
//#include <deal.II/lac/petsc_parallel_vector.h>
#include <deal.II/lac/slepc_solver.h>


#include <deal.II/lac/trilinos_sparse_matrix.h>

#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>

#include "Constituitive.h"
#include "ConstituitiveDD.h"


#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function.h>
#include <deal.II/base/tensor.h>

#include <deal.II/lac/vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/affine_constraints.h>

#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_refinement.h>
#include <deal.II/grid/manifold_lib.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/lac/affine_constraints.h>

//////////////////////////////////////////////
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function.h>
#include <deal.II/base/utilities.h>

#include <deal.II/lac/vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/affine_constraints.h>

#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_refinement.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/fe_q.h>

#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/error_estimator.h>

#include <deal.II/lac/identity_matrix.h>
#include <deal.II/base/work_stream.h>
#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Eigenvalues>
#include <nanoflann.hpp>

////////////////////////

#define MAXLINE 1024
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
  /****************************************************************
                       Class Declarations
  ****************************************************************/


  /****  ElasticProblem  *****
   * This is the primary class used, with all the dealii stuff
   */
  class ElasticProblem
  {
  public:
    ElasticProblem();
    ~ElasticProblem();

    void create_mesh();
	void create_mesh_hole();
    void   setup_system(const bool initial_step);

    void run_forward_problem();
	void run_force();
    void output_results(const unsigned int cycle) const;
    void read_input_file(char* filename);
	void run_DD();
	void testrun_DD();
	void run_forward_problem_2();


    unsigned int get_n_dofs(){return dof_handler.n_dofs();};
    unsigned int get_number_active_cells(){return triangulation.n_active_cells();};

  private:

    void setup_system_constraints();
    void apply_boundaries_and_constraints();

    void assemble_system_rhs();

// paralelization for assemble system rhs://///////////////////////////////////////
void assemble_system_rhs_parallel();

    struct AssemblyScratchData
    {
      AssemblyScratchData (const FiniteElement<DIM> &fe, Quadrature<DIM> &quad);
      AssemblyScratchData (const AssemblyScratchData &scratch_data);

      FEValues<DIM>     fe_values;
    };
	struct AssemblyCopyData
	    {
	      dealii::Vector<double>                       R_local;
		  dealii::FullMatrix<double>                   cell_matrix;
	      std::vector<types::global_dof_index> local_dof_indices;
	    };
    void local_assemble_system_rhs(
    		const typename DoFHandler<DIM>::active_cell_iterator &cell,
			AssemblyScratchData                                  &scratch,
			AssemblyCopyData                                     &copy_data);
	void copy_local_to_global(const AssemblyCopyData &copy_data);

// paralelization for adjoint://////////////////////////////////////////////
void calculate_scalar_value();

    struct AssemblyScratchData_adj
    {
      AssemblyScratchData_adj (const FiniteElement<DIM> &fe, Quadrature<DIM> &quad, QGauss<DIM - 1> 	&quad_face);
      AssemblyScratchData_adj (const AssemblyScratchData_adj &scratch_data);

      FEValues<DIM>     fe_values;
	  FEFaceValues<DIM> fe_face_values;
    };
    struct IntvarCopyData
    {};
	void copy_local_to_global_NOTHING(const IntvarCopyData &copy_data);
    void local_scalar_value(
    		const typename DoFHandler<DIM>::active_cell_iterator &cell,
			AssemblyScratchData_adj                                  &scratch,
			IntvarCopyData                                     		&copy_data);




// paralelization for residual:
double get_residual_parallel();


	struct AssemblyCopyData_res
	{
		dealii::Vector<double> cell_residual;
		std::vector<types::global_dof_index> local_dof_indices;
	};
	void assemble_system_residual();
	void compute_residual_parallel( //local
		 	const typename DoFHandler<DIM>::active_cell_iterator &cell,
		 	AssemblyScratchData                                  &scratch,
		 	AssemblyCopyData_res                                 &copy_data);

	void copy_local_to_global_res(const AssemblyCopyData_res &copy_data);


    void getNextDataLine( FILE* const filePtr, char* nextLinePtr,
        int const maxSize, int* const endOfFileFlag);
    void   solve();
    double compute_residual();
	void propagate_u();
	void grid_1();
	void initiate_guess();
	void initiate_guess_DD(const bool counter);	
	void calculate_end_disp( dealii::Vector<double> &soln );

	void output_forces();
	void read_input_file_DIC_exp_speckle();
	

	//DD SHIT:::
	void sigma_convert_to_tensor(const Tensor<2,DIM> &tensorin, Tensor<2, DIM> &tensorout);
	void tangent_convert_to_tensor(const Tensor<2,DIM> &tensorin, Tensor<4, DIM> &tensorout);
	void sens_convert_to_tensor(const MatrixXd &tensorin, std::vector<Tensor<4, DIM>> &tensorout);


    Triangulation<3>   triangulation;


    DoFHandler<DIM>      dof_handler;

    FESystem<DIM>        fe;

    AffineConstraints<double>  constraints;

    SparsityPattern      sparsity_pattern;
    SparseMatrix<double> system_matrix;

    dealii::Vector<double>       system_rhs;

    char output_directory[MAXLINE];

    std::vector<unsigned int>  grid_dimensions;
    std::vector<double> domain_dimensions;

    LinearElastic LE;
	DDElasticMaterial *DD1 = nullptr;
	std::vector<double> props; 

    SparseDirectUMFPACK  A_direct;

    //
    dealii::Vector<double> current_solution;
    dealii::Vector<double> newton_update;
    unsigned int counter; int f_increment;

    dealii::Vector<double> solution_u, solution_g, solution_ut;
	double total_force_exp, total_force_exp2;
	std::vector< Point< DIM > > support_points, support_points_all_dofs, support_points_y_dofs,  support_points_z_dofs;
	std::vector<Point<DIM>> pixel_coord, pixel_coord2;
	std::vector<std::vector<double>> specklepixels_all;
	std::vector<double> speckle;
	std::vector<int> dofindex;
	double scalarvalue = 0.0;
	dealii::Vector<double> residual;

    double       time_step;
    double       time;
    unsigned int timestep_number;

    unsigned int corner_dof = 0;

    unsigned int qx = 2;
    unsigned int qy = 2;
    unsigned int qz = 2;


	std::vector<bool> selected_dofs_x, selected_dofs_yz;

    double mu= 4.0*1000000.0; //4000000.0; //10000000.0; //49375.2;//400000;//49375.2;
    double nu= 0.3; //0.3;

	double mu2= 20.0*1000000.0;
	double nu2 = 0.43;
	double lowlim = 0.006;
	double uplim = 0.006+0.006;

    double rho= 1000.0;  //
	double bulk = 2*mu*(1+nu)/(3*(1-2*nu));

	double end_disp= 0.0;
	double L0 = 0.036;
	double strain_rate = 0.001; //0.001; //todo
	double deltat= time_step;
	double velocity_qs = strain_rate*L0;
	unsigned int endsteptime = 500;
	bool FORCEBC = 1;
	bool DISPBC = 1;
	bool HOLE = 0;
	bool RECT = 1;	

	double zheight, yheight;	
	
	std::string filename2;
	std::vector<unsigned int> cell_indices;
	std::vector<Tensor<1, DIM*DIM>> strain_dbs, stress_dbs;
	bool get_data = 0;
	bool ifLE = 0;
	bool ifDD = 1;

    std::string vary = "test1" ; //"DB_dumbDD_rectdomain_4ME_0.3nu_speckle10e_steptime0.5_end1000_refine3x";

	bool converge;


  };

  //////////////////// 2) displacement boundary condition /////////////////////////////////////////////////////////////////////////////////////
  	template <int dim>
  	class IncrementalBoundaryValuesDISP : public Function<dim>
  	{
  	public:
  		IncrementalBoundaryValuesDISP(const double present_time,
  				const double present_timestep,
  				const double timestep);

  		virtual void vector_value(const dealii::Point<dim> &p,
  				dealii::Vector<double> &  values) const override;

  	private:
  		const double velocity; //INITIAL VALUE
  		const double present_time;
  		const double present_timestep;
  		const double timestep;
  	};


  	template <int dim>
  	IncrementalBoundaryValuesDISP<dim>::IncrementalBoundaryValuesDISP(
  			const double present_time,
  			const double present_timestep,
  			const double timestep)
  	: Function<dim>(dim)
  	  , velocity(0.04)
  	  , present_time(present_time)
  	  , present_timestep(present_timestep)
  	  , timestep(timestep)
  	  {}

  	template <int dim>
  	void
  	IncrementalBoundaryValuesDISP<dim>::vector_value(const dealii::Point<dim> &p,
  			dealii::Vector<double> &values) const
  	{
  		AssertDimension(values.size(), dim);

  		if(present_timestep == 1){
  			  values = 0.0;
  			  values(0) = 0.0; //present_time*velocity;
  		  }

  		else{
  			  values = 0.0;
  			  values(0) = (present_time-timestep)*0.036*0.001;
  		  }
  	} //end of vector value for the BC function

}

#endif /* LINEAR_ELASTIC_H_ */
