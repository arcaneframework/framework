# Tutorial C API {#aliendoc_tutorials_tutorialc}

[TOC]

## Intro

We consider the Laplacian problem on a 2D square mesh of size \f$N_X \times N_Y\f$. Unknowns are related to the mesh nodes
indexed by \f$(i,j)\f$. We use a 5-Points stencil to discretize the problem.

First of all %Alien provides tools to initialize the library, the MPI parallel environment to parametrize IO.
The following code illustrates how to initialize :

- the ParallelMng object

- the TraceMng object


```c
int nprocs, my_rank, domain_offset ;
MPI_Comm comm;

/*
 * MPI Initialization
 */
MPI_Init(&argc,&argv) ;
comm = MPI_COMM_WORLD ;
MPI_Comm_size(comm,&nprocs) ;
MPI_Comm_rank(comm,&my_rank) ;
printf("NPROCS %d RANK %d \n",nprocs,my_rank);

/*
 * INITIALIZE ALIEN
 */
ALIEN_init(argc,argv) ;
```



## Linear System Set Up

The Matrix concept represents a set of \f$N_i\f$ linear equations (rows) \f$(y_i)\f$ of \f$N_j\f$ unknowns \f$(x_j)\f$ (columns).
This represents a linear application \f$S_X \mapsto S_Y\f$ with \f$x \in S_X\f$, \f$y \in S_Y\f$ and \f$x \mapsto y=A*x\f$.
Usually the dimension of \f$S_X\f$ and \f$S_Y\f$ are equal, \f$N_i=N_j\f$. In that case the matrix is square.

A linear system is a set of a Matrix A, two vectors B the right hand side and X the solution of the system A*X=B

```c
int i,k, r;
int system_id, solver_id, error;
int global_nrows, local_nrows, nb_ghosts ;
int row_size, local_nnz ;
int* row_offset, *ghost_owners;
uid_type *row_uids, *col_uids, *ghost_uids ;
double diag_values, offdiag_values, rhs_value, norm2, err, err2, gnorm2, gerr2 ;
double *matrix_values, *rhs_values, *solution_values, *ref_solution_values ;
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
  row_size = 3 ;
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
  rhs_value = diag_values*(domain_offset+i) ;
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
```


## Linear Systems resolution

A linear system is reprensented by a matrix \f$A\f$, and two vectors \f$B\f$ and \f$X\f$ where \f$B\f$ is the system right hand side and \f$X\f$ the solution.

Solving the linear system consists in finding the solution X such that \f$A*X=B\f$ applying a linear solver algorithm.


```c
/*
 * CREATE SOLVER
 */
printf("CREATE SOLVER \n") ;
solver_id = ALIEN_create_solver(comm,"./solver_config.xml") ;

printf("INIT SOLVER \n") ;
ALIEN_init_solver(solver_id,argc,argv) ;
  
    /*
 * LINEAR SYSTEM RESOLUTION
 */

printf(" SOLVE \n") ;
error += ALIEN_solve(solver_id,system_id) ;

/*
 * GET SOLUTION VALUES
 */
error += ALIEN_get_solution_values(system_id,local_nrows,row_uids,solution_values) ;

ALIEN_get_solver_status(solver_id,&solver_status) ;
if(solver_status.code == 0)
{
  /*
   * COMPUTE ERROR TO REF SOLUTION
   */
  norm2 = 0. ;
  for(i = 0;i<local_nrows;++i)
  {
    norm2 += rhs_values[i]*rhs_values[i] ;
    err = solution_values[i] - ref_solution_values[i] ;
    err2 += err*err ;
  }

  MPI_Allreduce(&norm2,&gnorm2,1,MPI_DOUBLE,MPI_SUM,comm) ;
  MPI_Allreduce(&err2,&gerr2,1,MPI_DOUBLE,MPI_SUM,comm) ;

  printf("REL ERROR2 : %f\n",gerr2/gnorm2) ;
}
```


## Destroy Linear System Objects

You have to destroy all created objects

```c
  /*
   * DESTROY SOLVER AND LINEAR SYSTEM
   */
  ALIEN_destroy_solver(solver_id) ;
  ALIEN_destroy_linear_system(system_id) ;

  free(row_uids) ;
  free(col_uids) ;
  free(row_offset) ;
  free(matrix_values) ;
  free(rhs_values) ;
  free(solution_values) ;
  free(ref_solution_values) ;
```

## Finalyze ALIEN


```c
/*
 * FINALYZE ALIEN
 */
ALIEN_finalize() ;
```


____

<div class="section_buttons">
<span class="back_section_button">
\ref aliendoc_tutorials_tutorialcpp
</span>
<span class="next_section_button">
\ref aliendoc_tutorials_tutorialf
</span>
</div>
