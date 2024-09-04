#include <fstream>

#include "../include/LinearElasticity_implicit.h"


using namespace dealii;
int main (int argc, char** argv)
{

  lce_elastic::ElasticProblem ep;

  char fileName[MAXLINE];
  std::cout << "Please enter an input file: " << std::endl;
  std::cin >> fileName;
  ep.read_input_file(fileName);

  //READ FORCE FILE//////////////////////
  // a) symmetric hole
  //ep.read_input_file_force_exp("forcesHOLErectdomain_4ME_0.3nu_100x40_specklereal_speckle10_steptime0.5_end1000_refine3x.out");
  // b) assymetric hole 1
  //  ep.read_input_file_force_exp("forcesasym1HOLErectdomain_4ME_0.3nu_100x40_specklereal_speckle10_steptime0.5_end1000_refine3x.out");
  // c) assymetric hole 2
  //  ep.read_input_file_force_exp("forcesasym2HOLErectdomain_4ME_0.3nu_100x40_speckle10e_steptime0.5_end1000_refine4x.out");
 // ep.read_input_file_force_exp("forcesBI_asym2_HOLErectdomain_4ME_0.3nu_20ME_0.43nu_speckle10e_steptime0.5_end1000_refine4x.out");
////////////////////////////////////
  //ep.run_forward_problem();

  ep.run_forward_problem_2();
  //std::cout<<"run dd"<<" ";
  //ep.testrun_DD();
  //std::cout << "Done with DD" << std::endl;
  
  return(0);
}

