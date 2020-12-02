/*
 * alienc.h
 *
 *  Created on: Nov 25, 2020
 *      Author: gratienj
 */
#ifndef ALIENF_H_
#define ALIENF_H_

#define F2C_
#if defined(_F2C) | defined(F2C_)
#define F2C(function) function##_
#else
#define F2C(function) function
#endif

//typedef long int uid_type ;

void F2C(alien_init_f)() ;

void F2C(alien_finalize_f)() ;

int F2C(alien_create_linear_system_f)(MPI_Fint* comm) ;

void F2C(alien_destroy_linear_system_f)(int* system_id) ;

void F2C(alien_init_linear_system_f)(int* system_id,
                                     int* global_nrows,
                                     int* local_nrows,
                                     uid_type* row_uids,
                                     int* nb_ghosts,
                                     uid_type* ghost_uids,
                                     int* ghost_owners) ;

void F2C(alien_define_matrix_profile_f)(int* system_id,
                                        int* local_nrows,
                                        uid_type* row_uids,
                                        int* row_offset,
                                        uid_type* col_uids) ;

void F2C(alien_set_matrix_values_f)(int* system_id,
                                 int* local_nrows,
                                 uid_type* row_uids,
                                 int* row_offset,
                                 uid_type* col_uids,
                                 double* values) ;

void F2C(alien_set_rhs_values_f)(int* system_id,
                              int* local_nrows,
                              uid_type* row_uids,
                              double const* values) ;

void F2C(alien_get_solution_values_f)(int* system_id,
                                   int* local_nrows,
                                   uid_type* row_uids,
                                   double* values) ;

int F2C(alien_create_solver_f)(MPI_Fint* comm) ;

void F2C(alien_init_solver_f)(int* solver_id, const char* config_file, int* lenghth) ;

void F2C(alien_destroy_solver_f)(int* solver_id) ;

void F2C(alien_solve_f)(int* solver_id, int* system_id) ;

void F2C(alien_get_solver_status_f)(int* solver_id,int* code, int* num_iterations, double* residual) ;

#endif /* MODULES_INTERFACE_C_SRC_ALIEN_C_ALIENC_H_ */
