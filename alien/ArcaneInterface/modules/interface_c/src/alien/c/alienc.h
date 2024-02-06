// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2024 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* alienc                                         (C) 2000-2024              */
/*                                                                           */
/* Interface C for alien                                                     */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
#ifndef MODULES_INTERFACE_C_SRC_ALIEN_C_ALIENC_H_
#define MODULES_INTERFACE_C_SRC_ALIEN_C_ALIENC_H_

#include <alien/AlienInterfaceCExport.h>

typedef long int uid_type ;

ALIEN_INTERFACE_C_EXPORT int ALIEN_init(int argc, char** argv) ;

ALIEN_INTERFACE_C_EXPORT int ALIEN_finalize() ;

ALIEN_INTERFACE_C_EXPORT int ALIEN_create_linear_system(MPI_Comm comm) ;
ALIEN_INTERFACE_C_EXPORT int ALIEN_destroy_linear_system(int system_id) ;

ALIEN_INTERFACE_C_EXPORT int ALIEN_init_linear_system(int system_id,
                             int global_nrows,
                             int local_nrows,
                             uid_type* row_uids,
                             int nb_ghosts,
                             uid_type* ghost_uids,
                             int* ghost_owners) ;

ALIEN_INTERFACE_C_EXPORT int ALIEN_define_matrix_profile(int system_id,
                                int local_nrows,
                                uid_type* row_uids,
                                int* row_offset,
                                uid_type* col_uids) ;

ALIEN_INTERFACE_C_EXPORT int ALIEN_set_matrix_values(int system_id,
                            int local_nrows,
                            uid_type* row_uids,
                            int* row_offset,
                            uid_type* col_uids,
                            double* values) ;

ALIEN_INTERFACE_C_EXPORT int ALIEN_set_rhs_values(int system_id,
                         int local_nrows,
                         uid_type* row_uids,
                         double const* values) ;

ALIEN_INTERFACE_C_EXPORT int ALIEN_get_solution_values(int system_id,
                              int local_nrows,
                              uid_type* row_uids,
                              double* values) ;

ALIEN_INTERFACE_C_EXPORT int ALIEN_create_solver(MPI_Comm comm, const char* config_file) ;

ALIEN_INTERFACE_C_EXPORT int ALIEN_init_solver(int solver_id,int argc, char** argv) ;

ALIEN_INTERFACE_C_EXPORT int ALIEN_init_solver_with_configfile(int solver_id,const char* config_file) ;

ALIEN_INTERFACE_C_EXPORT int ALIEN_destroy_solver(int solver_id) ;

ALIEN_INTERFACE_C_EXPORT int ALIEN_solve(int solver_id, int system_id) ;


struct ALIEN_Solver_Status
{
  int    code ;
  double residual ;
  int    num_iterations ;
};

ALIEN_INTERFACE_C_EXPORT int ALIEN_get_solver_status(int solver_id, struct ALIEN_Solver_Status* status) ;

#endif /* MODULES_INTERFACE_C_SRC_ALIEN_C_ALIENC_H_ */
