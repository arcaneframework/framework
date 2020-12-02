/*
 * alienc.h
 *
 *  Created on: Nov 25, 2020
 *      Author: gratienj
 */

#ifndef MODULES_INTERFACE_C_SRC_ALIEN_C_ALIENC_H_
#define MODULES_INTERFACE_C_SRC_ALIEN_C_ALIENC_H_



typedef long int uid_type ;

int ALIEN_init(int argc, char** argv) ;

int ALIEN_finalize() ;

int ALIEN_create_linear_system(MPI_Comm comm) ;
int ALIEN_destroy_linear_system(int system_id) ;

int ALIEN_init_linear_system(int system_id,
                             int global_nrows,
                             int local_nrows,
                             uid_type* row_uids,
                             int nb_ghosts,
                             uid_type* ghost_uids,
                             int* ghost_owners) ;

int ALIEN_define_matrix_profile(int system_id,
                                int local_nrows,
                                uid_type* row_uids,
                                int* row_offset,
                                uid_type* col_uids) ;

int ALIEN_set_matrix_values(int system_id,
                            int local_nrows,
                            uid_type* row_uids,
                            int* row_offset,
                            uid_type* col_uids,
                            double* values) ;

int ALIEN_set_rhs_values(int system_id,
                         int local_nrows,
                         uid_type* row_uids,
                         double const* values) ;

int ALIEN_get_solution_values(int system_id,
                              int local_nrows,
                              uid_type* row_uids,
                              double* values) ;

int ALIEN_create_solver(MPI_Comm comm, const char* config_file) ;

int ALIEN_init_solver(int solver_id,int argc, char** argv) ;

int ALIEN_init_solver_with_configfile(int solver_id,const char* config_file) ;

int ALIEN_destroy_solver(int solver_id) ;

int ALIEN_solve(int solver_id, int system_id) ;


struct ALIEN_Solver_Status
{
  int    code ;
  double residual ;
  int    num_iterations ;
};

int ALIEN_get_solver_status(int solver_id, struct ALIEN_Solver_Status* status) ;

#endif /* MODULES_INTERFACE_C_SRC_ALIEN_C_ALIENC_H_ */
