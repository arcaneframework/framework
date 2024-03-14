subroutine test
    use M_AlienModule
    implicit none

  integer :: i,k, r
  integer :: system_id, solver_id
  integer :: global_nrows, local_nrows, nb_ghosts
  integer :: row_size, local_nnz
  integer, allocatable, dimension(:) :: row_offset, ghost_owners
  integer(8), allocatable, dimension(:) :: row_uids, col_uids, ghost_uids
  double precision :: diag_values, offdiag_values, rhs_value, norm2, err, err2, gnorm2, gerr2
  double precision, allocatable, dimension(:) :: matrix_values, rhs_values, solution_values, ref_solution_values
  integer :: nprocs, my_rank, domain_offset
  integer :: comm, ierr
  integer :: code, num_iterations
  real(8) ::residual
  character(len=80) :: config_file
  integer :: alloc_stat

  !
  ! MPI INITIALIZATION
  !
  call MPI_Init(ierr)
  comm = MPI_COMM_WORLD
  call MPI_Comm_size(comm,nprocs,ierr)
  call MPI_Comm_rank(comm,my_rank,ierr)

  !
  ! ALIEN INITIALIZATION
  !
  print*, "NPROCS RANK",nprocs,my_rank
  call ALIEN_init()


  !
  ! CREATE LINEAR SYSTEM A*X=B : MATRIX A, VECTOR SOLUTION X, VECTOR RHS B
  !
  system_id = ALIEN_CreateLinearSystem(comm)
  print*,'SYSTEM ID',system_id

  !
  ! DEFINE MATRIX PROFILE
  !
  global_nrows = 32
  local_nrows = global_nrows / nprocs
  r = modulo(global_nrows ,nprocs)
  if(my_rank < r) then
    local_nrows = local_nrows + 1
    domain_offset = my_rank*local_nrows
  else
    domain_offset = my_rank * local_nrows + r
  endif

  allocate(row_uids(local_nrows),stat=alloc_stat)
  allocate(row_offset(local_nrows+1),stat=alloc_stat)
  row_offset(1) = 0
  do i=1,local_nrows
    row_uids(i) = domain_offset + i - 1
    row_size = 3
    if ( (domain_offset + i -1 .eq. 0 ) .or. (domain_offset + i -1 .eq. global_nrows -1)) then
      row_size = 2
    endif
    row_offset(i+1) = row_offset(i) + row_size
  enddo

  allocate(ghost_uids(2),stat=alloc_stat)
  allocate(ghost_owners(2),stat=alloc_stat)
  nb_ghosts = 0
  if (my_rank .gt. 0) then
    nb_ghosts = nb_ghosts + 1;
    ghost_uids(nb_ghosts) = domain_offset-1
    ghost_owners(nb_ghosts) = my_rank-1
  endif
  if(my_rank .lt. nprocs-1) then
    nb_ghosts = nb_ghosts+1;
    ghost_uids(nb_ghosts) = domain_offset+local_nrows
    ghost_owners(nb_ghosts) = my_rank+1
  endif


  local_nnz = row_offset(local_nrows+1)
  allocate(col_uids(local_nnz),stat=alloc_stat)
  allocate(matrix_values(local_nnz),stat=alloc_stat)

  allocate(rhs_values(local_nrows),stat=alloc_stat)
  allocate(solution_values(local_nrows),stat=alloc_stat)
  allocate(ref_solution_values(local_nrows),stat=alloc_stat)

  diag_values = 10.
  offdiag_values  = -1.
  k = 1
  do i=1,local_nrows
    col_uids(k) = domain_offset + i - 1
    matrix_values(k) = diag_values
    rhs_value = diag_values*(domain_offset + i - 1)
    k = k + 1
    if (domain_offset+i-1 .ne. 0) then

      col_uids(k) = domain_offset + i - 2
      matrix_values(k) = offdiag_values
      rhs_value = rhs_value + offdiag_values*(domain_offset + i - 2)
      k = k + 1
    endif
    if (domain_offset+i-1 .ne. global_nrows -1) then
      col_uids(k) = domain_offset + i
      matrix_values(k) = offdiag_values

      rhs_value = rhs_value + offdiag_values*(domain_offset + i)
      k = k + 1
    endif
    rhs_values(i) = rhs_value
    ref_solution_values(i) = domain_offset + i - 1
  enddo

  call ALIEN_InitLinearSystem(system_id,global_nrows,local_nrows,row_uids,nb_ghosts,ghost_uids,ghost_owners)

  call ALIEN_DefineMatrixProfile(system_id,local_nrows,row_uids,row_offset,col_uids)

  call ALIEN_SetMatrixValues(system_id,local_nrows,row_uids,row_offset,col_uids,matrix_values)

  call ALIEN_SetRhsValues(system_id,local_nrows,row_uids,rhs_values)


  !
  ! CREATE LINEAR SOLVER 
  !
  solver_id = ALIEN_CreateSolver(comm)

  !
  ! LINEAR SOLVER SET UP WITH CONFIG FILE 
  !
  config_file = "solver.json"
  call ALIEN_InitSolver(solver_id,config_file)

  !
  ! LINEAR SYSTEM RESOLUTION
  !
  call ALIEN_Solve(solver_id,system_id)

  call ALIEN_GetSolutionValues(system_id,local_nrows,row_uids,solution_values)

  call ALIEN_GetSolverStatus(solver_id,code,num_iterations,residual)
  if(code .eq. 0) then
    norm2 = 0.
    do i = 1,local_nrows

      norm2 = norm2 + rhs_values(i)*rhs_values(i)
      err = solution_values(i) - ref_solution_values(i)
      err2 = err2 + err*err
    enddo
    call MPI_ALLREDUCE(norm2,gnorm2,1,MPI_DOUBLE,MPI_SUM,comm,ierr) ;
    call MPI_ALLREDUCE(err2,gerr2,1,MPI_DOUBLE,MPI_SUM,comm,ierr) ;
    print*,"REL ERROR2 : ",gerr2/gnorm2 ;
  endif

  !
  ! LINEAR SOLVER AND SYSTEM DESTRUCTION
  !
  call ALIEN_DestroySolver(solver_id)
  call ALIEN_DestroyLinearSystem(system_id)

  ! 
  ! ALIEN FINALIZE
  !
  call ALIEN_finalize()


end subroutine test
