/*
 * alienc.h
 *
 *  Created on: Nov 25, 2020
 *      Author: gratienj
 */

#ifndef MODULES_INTERFACE_C_SRC_ALIEN_F_ALIENC_H_
#define MODULES_INTERFACE_C_SRC_ALIEN_F_ALIENC_H_

#include <mpi.h>

#include <alien/c/alienc.h>
#include <alien/fortran/alienf.h>
#define NULL 0


void F2C(alien_init_f)()
{
  ALIEN_init(0,NULL) ;
}

void F2C(alien_finalize_f)()
{
  ALIEN_finalize() ;
}

int F2C(alien_create_linear_system_f)(MPI_Fint* f_comm)
{
  MPI_Comm comm = MPI_Comm_f2c(*f_comm) ;
  return ALIEN_create_linear_system(comm) ;
}

void F2C(alien_destroy_linear_system_f)(int* system_id)
{
  ALIEN_destroy_linear_system(*system_id) ;
}

void F2C(alien_init_linear_system_f)(int* system_id,
                                     int* global_nrows,
                                     int* local_nrows,
                                     uid_type* row_uids,
                                     int* nb_ghosts,
                                     uid_type* ghost_uids,
                                     int* ghost_owners)
{
  ALIEN_init_linear_system(*system_id,
                           *global_nrows,
                           *local_nrows,
                           row_uids,
                           *nb_ghosts,
                           ghost_uids,
                           ghost_owners) ;
}

void F2C(alien_define_matrix_profile_f)(int* system_id,
                                     int* local_nrows,
                                     uid_type* row_uids,
                                     int* row_offset,
                                     uid_type* col_uids)
{
  ALIEN_define_matrix_profile(*system_id,
                              *local_nrows,
                              row_uids,
                              row_offset,
                              col_uids) ;
}

void F2C(alien_set_matrix_values_f)(int* system_id,
                                 int* local_nrows,
                                 uid_type* row_uids,
                                 int* row_offset,
                                 uid_type* col_uids,
                                 double* values)
{
  ALIEN_set_matrix_values(*system_id,
                          *local_nrows,
                          row_uids,
                          row_offset,
                          col_uids,
                          values) ;
}

void F2C(alien_set_rhs_values_f)(int* system_id,
                              int* local_nrows,
                              uid_type* row_uids,
                              double const* values)
{
  ALIEN_set_rhs_values(*system_id,
                       *local_nrows,
                       row_uids,
                       values) ;
}

void F2C(alien_get_solution_values_f)(int* system_id,
                                   int* local_nrows,
                                   uid_type* row_uids,
                                   double* values)
{
  ALIEN_get_solution_values(*system_id,
                            *local_nrows,
                            row_uids,
                            values) ;
}

int F2C(alien_create_solver_f)(MPI_Fint* f_comm)
{
  MPI_Comm comm = MPI_Comm_f2c(*f_comm) ;
  return ALIEN_create_solver(comm,NULL) ;
}

void F2C(alien_init_solver_f)(int* solver_id,
                              const char* config_file,
                              int* length)
{
  char* string = (char*) config_file ;
  string[*length] = 0 ;
  ALIEN_init_solver_with_configfile(*solver_id,config_file) ;
}

void F2C(alien_destroy_solver_f)(int* solver_id)
{
  ALIEN_destroy_solver(*solver_id) ;
}

void F2C(alien_solve_f)(int* solver_id, int* system_id)
{
  ALIEN_solve(*solver_id,*system_id) ;
}

void F2C(alien_get_solver_status_f)(int* solver_id,int* code, int* num_iterations, double* residual)
{
  struct ALIEN_Solver_Status status ;
  ALIEN_get_solver_status(*solver_id,&status) ;
  *code = status.code ;
  *num_iterations = status.num_iterations ;
  *residual = status.residual ;
}

#endif /* MODULES_INTERFACE_C_SRC_ALIEN_C_ALIENC_H_ */
