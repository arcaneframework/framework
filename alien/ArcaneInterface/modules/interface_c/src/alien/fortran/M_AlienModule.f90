module M_AlienModule
    implicit none
    include 'mpif.h'

    public
    logical :: AlienIsInitialized = .false.

    contains

    subroutine ALIEN_Init()
      call alien_init_f()
      AlienIsInitialized = .true.
    end subroutine ALIEN_Init

    subroutine ALIEN_Finalize()
      call alien_finalize_f()
    end subroutine ALIEN_Finalize

    integer function ALIEN_CreateLinearSystem(comm)
      integer :: alien_create_linear_system_f
      integer, intent(in) :: comm
      ALIEN_CreateLinearSystem = alien_create_linear_system_f(comm)
    end function

    subroutine ALIEN_DestroyLinearSystem(system_id)
        integer, intent(in) :: system_id
        call alien_destroy_linear_system_f(system_id)
    end subroutine ALIEN_DestroyLinearSystem

    subroutine ALIEN_InitLinearSystem(system_id,    &
                                        global_nrows, &
                                        local_nrows,  &
                                        row_uids,     &
                                        nb_ghosts,    &
                                        ghost_uids,   &
                                        ghost_owners)
         integer, intent(in) :: system_id
         integer, intent(in) :: global_nrows
         integer, intent(in) :: local_nrows
         integer(8), dimension(:), intent(in) :: row_uids
         integer, intent(in) :: nb_ghosts
         integer(8), dimension(:), intent(in) :: ghost_uids
         integer, dimension(:), intent(in) :: ghost_owners
         print*,'InitLinearSystem',system_id
         call alien_init_linear_system_f(system_id,    &
                                       global_nrows, &
                                       local_nrows,  &
                                       row_uids,     &
                                       nb_ghosts,    &
                                       ghost_uids,   &
                                       ghost_owners)
    end subroutine

    subroutine ALIEN_DefineMatrixProfile(system_id,   &
                                         local_nrows, &
                                         row_uids,    &
                                         row_offset,  &
                                         col_uids)
        integer, intent(in) :: system_id
        integer, intent(in) :: local_nrows
        integer(8), dimension(:), intent(in) :: row_uids
        integer, dimension(:), intent(in) :: row_offset
        integer(8), dimension(:), intent(in) :: col_uids

        call alien_define_matrix_profile_f(system_id,   &
                                         local_nrows, &
                                         row_uids,    &
                                         row_offset,  &
                                         col_uids)
    end subroutine

    subroutine ALIEN_SetMatrixValues(system_id,   &
                                         local_nrows, &
                                         row_uids,    &
                                         row_offset,  &
                                         col_uids,    &
                                         values)
        integer, intent(in) :: system_id
        integer, intent(in) :: local_nrows
        integer(8), dimension(:), intent(in) :: row_uids
        integer, dimension(:), intent(in) :: row_offset
        integer(8), dimension(:), intent(in) :: col_uids
        double precision, dimension(:), intent(in) :: values

        call alien_set_matrix_values_f(system_id,   &
                                     local_nrows, &
                                     row_uids,    &
                                     row_offset,  &
                                     col_uids,    &
                                     values)
    end subroutine

    subroutine ALIEN_SetRHSValues(system_id,   &
                                 local_nrows, &
                                 row_uids,    &
                                 values)
        integer, intent(in) :: system_id
        integer, intent(in) :: local_nrows
        integer(8), dimension(:), intent(in) :: row_uids
        double precision, dimension(:), intent(in) :: values

        call alien_set_rhs_values_f(system_id,   &
                                  local_nrows, &
                                  row_uids,    &
                                  values)
    end subroutine


    subroutine ALIEN_GetSolutionValues(system_id,   &
                                       local_nrows, &
                                       row_uids,    &
                                       values)
        integer, intent(in) :: system_id
        integer, intent(in) :: local_nrows
        integer(8), dimension(:), intent(in) :: row_uids
        double precision, dimension(:), intent(out) :: values

        call alien_get_solution_values_f(system_id,   &
                                       local_nrows, &
                                       row_uids,    &
                                       values)
    end subroutine

    integer function ALIEN_CreateSolver(comm)
        integer, intent(in) :: comm
        integer :: alien_create_solver_f
        ALIEN_CreateSolver = alien_create_solver_f(MPI_COMM_WORLD)
    end function

    subroutine ALIEN_InitSolver(solver_id,config_file)
        integer, intent(in) :: solver_id
        character(len=80), intent(in) :: config_file
        call alien_init_solver_f(solver_id,trim(config_file),len(trim(config_file)))
    end subroutine

    subroutine ALIEN_DestroySolver(solver_id)
        integer, intent(in) :: solver_id

        call alien_destroy_solver_f(solver_id)
    end subroutine


    subroutine ALIEN_Solve(solver_id,system_id)
        integer, intent(in) :: solver_id
        integer, intent(in) :: system_id

        call alien_solve_f(solver_id,system_id)
    end subroutine

    subroutine ALIEN_GetSolverStatus(solver_id,      &
                                     code,           &
                                     num_iterations, &
                                     residual)

        integer, intent(in) :: solver_id
        integer, intent(out) :: code
        integer, intent(out) :: num_iterations
        double precision, intent(out) :: residual

        call alien_get_solver_status_f(solver_id,code,num_iterations,residual)
    end subroutine


end module M_AlienModule
