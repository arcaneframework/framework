/*
 * main.c
 *
 *  Created on: Nov 25, 2020
 *      Author: gratienj
 */
#include <mpi.h>
//extern "C" {
#include <alien/c/alienc.h>
//}
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(int argc, char** argv)
{
  typedef long int uid_type ;
  int i,k, r;
  int system_id, param_system_id, solver_id, error;
  int global_nrows, local_nrows, nb_ghosts ;
  int local_nnz ;
  int* row_offset, *ghost_owners;
  uid_type *row_uids, *col_uids, *ghost_uids ;
  double diag_values, offdiag_values, gnorm2, gerr2 ;
  double *matrix_values, *rhs_values, *solution_values, *ref_solution_values ;
  int nprocs, my_rank, domain_offset ;
  MPI_Comm comm;
  struct ALIEN_Solver_Status solver_status ;

  MPI_Init(&argc,&argv) ;
  comm = MPI_COMM_WORLD ;
  MPI_Comm_size(comm,&nprocs) ;
  MPI_Comm_rank(comm,&my_rank) ;

  printf("NPROCS %d RANK %d \n",nprocs,my_rank);


  /*
   * INITIALIZE ALIEN
   */
  ALIEN_init(argc,argv) ;


  /*
   * CREATE LINEAR SYSTEM A*X=B : MATRIX A, VECTOR SOLUTION X, VECTOR RHS B
   */
  system_id = ALIEN_create_linear_system(comm) ;

  printf("SYSTEM ID %d: \n",system_id);
  /*
   * DEFINE MATRIX PROFILE
   */
  global_nrows = 32 ;
  local_nrows = global_nrows / nprocs ;
  r = global_nrows % nprocs ;
  if(my_rank < r)
  {
    ++local_nrows ;
    domain_offset = my_rank*local_nrows ;
  }
  else
  {
    domain_offset = my_rank * local_nrows + r ;
  }

  row_uids = (uid_type*) malloc(local_nrows*sizeof(uid_type)) ;
  row_offset = (int*) malloc((local_nrows+1)*sizeof(int)) ;
  row_offset[0] = 0 ;
  for(i=0;i<local_nrows;++i)
  {
    row_uids[i] = domain_offset+i ;
    int row_size = 3 ;
    if((domain_offset+i == 0 ) || (domain_offset+i == global_nrows -1)) row_size = 2 ;
    row_offset[i+1] = row_offset[i] + row_size ;
  }

  ghost_uids = (uid_type*) malloc(2*sizeof(uid_type)) ;
  ghost_owners = (int*) malloc(2*sizeof(int)) ;
  nb_ghosts = 0 ;
  if(my_rank>0)
  {
    ghost_uids[nb_ghosts] = domain_offset-1 ;
    ghost_owners[nb_ghosts] = my_rank-1 ;
    ++nb_ghosts ;
  }
  if(my_rank<nprocs-1)
  {
    ghost_uids[nb_ghosts] = domain_offset+local_nrows ;
    ghost_owners[nb_ghosts] = my_rank+1 ;
    ++nb_ghosts ;
  }


  local_nnz = row_offset[local_nrows] ;
  col_uids = (uid_type*) malloc(local_nnz*sizeof(uid_type)) ;
  matrix_values = (double*) malloc(local_nnz*sizeof(double)) ;

  rhs_values          = (double*) malloc(local_nrows*sizeof(double)) ;
  solution_values     = (double*) malloc(local_nrows*sizeof(double)) ;
  ref_solution_values = (double*) malloc(local_nrows*sizeof(double)) ;

  diag_values = 10. ;
  offdiag_values  = -1. ;
  k = 0 ;
  for(i=0;i<local_nrows;++i)
  {
    col_uids[k] = domain_offset+i ;
    matrix_values[k] = diag_values ;
    double rhs_value = diag_values*(domain_offset+i) ;
    ++k ;
    if(domain_offset+i != 0 )
    {
      col_uids[k] = domain_offset+i-1 ;
      matrix_values[k] = offdiag_values ;
      rhs_value += offdiag_values*(domain_offset+i-1) ;
      ++k ;
    }
    if(domain_offset+i != global_nrows -1)
    {
      col_uids[k] = domain_offset+i+1 ;
      matrix_values[k] = offdiag_values ;
      rhs_value += offdiag_values*(domain_offset+i+1) ;
      ++k ;
    }
    rhs_values[i] = rhs_value ;
    ref_solution_values[i] = domain_offset+i ;
  }

  printf("INIT SYSTEM ID %d: gsize=%d lsize=%d\n",system_id,global_nrows,local_nrows);
  error = ALIEN_init_linear_system(system_id,global_nrows,local_nrows,row_uids,nb_ghosts,ghost_uids,ghost_owners) ;

  printf("DEFINE MATRIX PROFILE ID %d: gsize=%d lsize=%d\n",system_id,global_nrows,local_nrows);
  error += ALIEN_define_matrix_profile(system_id,local_nrows,row_uids,row_offset,col_uids) ;

  /*
   * SET MATRIX VALUES
   */

  printf("SET MATRIX VALUES %d: gsize=%d lsize=%d\n",system_id,global_nrows,local_nrows);
  error += ALIEN_set_matrix_values(system_id,local_nrows,row_uids,row_offset,col_uids,matrix_values) ;

  /*
   * SET RHS VALUES
   */

  printf("SET RHS VALUES %d: gsize=%d lsize=%d\n",system_id,global_nrows,local_nrows);
  error += ALIEN_set_rhs_values(system_id,local_nrows,row_uids,rhs_values) ;

  if(error!=0)
    printf("ERRORS while setting the Linear System") ;

  /*
   * CREATE PARAMETER SYSTEM
   */
  printf("CREATE PARAMETER SYSTEM \n") ;
  param_system_id = ALIEN_create_parameter_system() ;

  ALIEN_set_parameter_string_value(param_system_id,"solver-package","petsc") ;
  ALIEN_set_parameter_double_value(param_system_id,"tol",1.0e-10) ;
  ALIEN_set_parameter_integer_value(param_system_id,"max-iter",1000) ;
  ALIEN_set_parameter_string_value(param_system_id,"petsc-solver","bicgs") ;
  ALIEN_set_parameter_string_value(param_system_id,"petsc-precond" ,"bjacobi") ;

  /*
   * CREATE SOLVER
   */
  printf("CREATE SOLVER \n") ;
  solver_id = ALIEN_create_solver(comm,NULL) ;

  printf("INIT SOLVER \n") ;
  ALIEN_init_solver_with_parameters(solver_id,param_system_id) ;

  /*
   * LINEAR SYSTEM RESOLUTION
   */

  printf(" SOLVE \n") ;
  error += ALIEN_solve(solver_id,system_id) ;

  /*
   * GET SOLUTION VALUES
   */
  error += ALIEN_get_solution_values(system_id,local_nrows,row_uids,solution_values) ;

  if(error!=0)
    printf("ERRORS while solving the Linear System") ;

  ALIEN_get_solver_status(solver_id,&solver_status) ;
  if(solver_status.code == 0)
  {
    /*
     * COMPUTE ERROR TO REF SOLUTION
     */
    double norm2 = 0. ;
    double err2 = 0. ;
    for(i = 0;i<local_nrows;++i)
    {
      norm2 += rhs_values[i]*rhs_values[i] ;
      double err = solution_values[i] - ref_solution_values[i] ;
      err2 += err*err ;
    }

    MPI_Allreduce(&norm2,&gnorm2,1,MPI_DOUBLE,MPI_SUM,comm) ;
    MPI_Allreduce(&err2,&gerr2,1,MPI_DOUBLE,MPI_SUM,comm) ;

    printf("REL ERROR2 : %f\n",gerr2/gnorm2) ;
  }

  /*
   * DESTROY SOLVER AND LINEAR SYSTEM
   */
  printf("DESTROY Linear Solver\n") ;
  ALIEN_destroy_solver(solver_id) ;

  printf("DESTROY Parameter System\n") ;
  ALIEN_destroy_parameter_system(param_system_id) ;

  printf("DESTROY Linear System\n") ;
  ALIEN_destroy_linear_system(system_id) ;

  printf("FREE ARRAYS\n") ;
  free(row_uids) ;
  free(col_uids) ;
  free(row_offset) ;
  free(ghost_uids) ;
  free(ghost_owners) ;
  free(matrix_values) ;
  free(rhs_values) ;
  free(solution_values) ;
  free(ref_solution_values) ;

  /*
   * FINALYZE ALIEN
   */
  printf("FINALIZE ALIEN\n") ;
  ALIEN_finalize() ;


  MPI_Finalize() ;

  return 0 ;
}
