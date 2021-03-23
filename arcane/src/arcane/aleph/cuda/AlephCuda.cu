/*---------------------------------------------------------------------------*/
/* AlephCuda.cc                                                     (C) 2011 */
/*                                                                           */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

// Visiblement CUDA (au moins jusqu'a la version 6.5) ne supporte
// pas 'nullptr' meme avec l'option '-std=c++11' active

#define nullptr 0

#include "arcane/aleph/AlephArcane.h"
#include "AlephCuda.h"

// Thread block size (THREAD_BLOCK_SIZE² = 768 at max)
#warning HARD-CODED THREAD_BLOCK_SIZE
#define THREAD_BLOCK_SIZE 32
#define SSE2_ALIGNEMENT 32
#define ELEM_SIZE 8


dim3 CNCdimGrid_vec;
dim3 CNCdimBlock_vec;


/****************************************************************************
 * cnc_cuda_set_dim_vec
 ****************************************************************************/
void cnc_cuda_set_dim_vec ( unsigned int dim_grid_x, unsigned int dim_grid_y,
                            unsigned int dim_block_x, unsigned int dim_block_y ) {
//  printf("\n[cnc_cuda_set_dim_vec] grid:%dx%d block:%dx%d", dim_grid_x, dim_grid_y, dim_block_x, dim_block_y);
  CNCdimGrid_vec.x  = dim_grid_x ;
  CNCdimGrid_vec.y  = dim_grid_y ;
  CNCdimBlock_vec.x = dim_block_x ;
  CNCdimBlock_vec.y = dim_block_y ;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_BEGIN_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/******************************************************************************
 * Aleph::Cuda
 *****************************************************************************/
Cuda::Cuda(): gpu_r(NULL),
              gpu_d(NULL),
              gpu_h(NULL),
              gpu_Ad(NULL),
              gpu_diag_inv(NULL),
              gpu_b(NULL),
              gpu_x(NULL),
              gpu_temp(NULL),
              gpu_temp0(NULL),
              gpu_temp1(NULL),
              gpu_res0(NULL){
/*  printf("\t[Aleph::Cuda::Cuda] NEW manager\n");
  printf("\t[Aleph::Cuda::Cuda] RESETing\n");
  cudaDeviceReset();
  printf("\t[Aleph::Cuda::Cuda] SYNCHRONIZING\n");
  cudaDeviceSynchronize();
  printf("\t[Aleph::Cuda::Cuda] Initializing CUBLAS\n");
  cublas_get_error(cublasInit());*/
}

  
/******************************************************************************
 * Aleph::~Cuda
 *****************************************************************************/
Cuda::~Cuda(){
  printf("\t[Aleph::Cuda::~Cuda] DELETE'ing manager\n");
  cublasFree(gpu_r);
  cublasFree(gpu_d);
  cublasFree(gpu_h);
  cublasFree(gpu_Ad);
  cublasFree(gpu_diag_inv);
  cublasFree(gpu_x);
  cublasFree(gpu_b);
  cublasFree(gpu_temp);
  cublasFree(gpu_temp0);
  cublasFree(gpu_temp1);
  cublasFree(gpu_res0);
  cublas_get_error(cublasShutdown()) ;
}

  
/****************************************************************************
 * compute_thread_index
 ****************************************************************************/
__device__ unsigned int compute_thread_index () {
  return ( blockIdx.x*THREAD_BLOCK_SIZE*THREAD_BLOCK_SIZE+
           blockIdx.y*THREAD_BLOCK_SIZE*THREAD_BLOCK_SIZE*gridDim.x+
           threadIdx.x+threadIdx.y*THREAD_BLOCK_SIZE) ;
}


/****************************************************************************
 * CNCVecVecMultKernel
 ****************************************************************************/
__global__ void CNCVecVecMultKernel(unsigned int size, double * x, double * y, double * r ) {
  // Thread index
  const unsigned int index = compute_thread_index () ;
  if ( index < size )
    r[index] = x[index]*y[index] ;
}


/****************************************************************************
 * CNCMat1x1VecMultKernel
 ****************************************************************************/
__global__ void CNCMat1x1VecMultKernel ( double * matrix, unsigned int size_matrix,
                                         uint2 * rowptr, unsigned int size_rowptr,
                                         unsigned int * colind, unsigned int size_colind,
                                         double * x, double * b, unsigned int size_vec ) {
  // Thread index
  const unsigned int index = compute_thread_index () ;
  if (index<size_vec){
    uint2 rowptr_bounds = rowptr[index] ;
    double res = 0.0f ;
    // for each block of the block_row, mult
    for ( unsigned int i=rowptr_bounds.x; i<rowptr_bounds.y; i++ ) { 
      res += matrix[i]*x[colind[i]] ;
    }
    b[index] = res ;
  }
}

/****************************************************************************
 * mat1x1vecmult
 ****************************************************************************/
void cnc_cuda_mat1x1vecmult(double * matrix, unsigned int size_matrix,
                            uint2 * rowptr, unsigned int size_rowptr,
                            unsigned int * colind, unsigned int size_colind,
                            double * x, double * b, unsigned int size_vec ) {
  // Launch the device computation
//  printf("\n[cnc_cuda_mat1x1vecmult]");
  CNCMat1x1VecMultKernel<<<CNCdimGrid_vec, CNCdimBlock_vec>>>(matrix, size_matrix, rowptr, size_rowptr,
                                                              colind, size_colind, x, b, size_vec);
}


/****************************************************************************
 * cnc_cuda_vecvecmult
 ****************************************************************************/
void cnc_cuda_vecvecmult( unsigned int size, double * x, double * y, double * r ) {
  // Launch the device computation
//   printf("\n[cnc_cuda_vecvecmult]");
  CNCVecVecMultKernel<<<CNCdimGrid_vec, CNCdimBlock_vec>>>(size, x, y, r);
}

  
/****************************************************************************
 * Cuda::solve_cg
 ****************************************************************************/
void Cuda::convert_matrix(const CNC_Matrix& rhs, CNC_MatrixCRS<double>& A, bool separate_diag ) {
  A.separate_diag = separate_diag ;
  A.symmetric_storage = rhs.has_symmetric_storage() ;
  A.N = rhs.n() ;
  A.rowptr.allocate(rhs.m() + 1) ;
  unsigned int nnz = rhs.nnz() ;
  if(separate_diag) 
    nnz -= rhs.diag_size() ;
  A.colind.allocate(nnz) ;
  A.a.allocate(nnz, SSE2_ALIGNEMENT) ;
  A.diag.allocate(rhs.diag_size(), SSE2_ALIGNEMENT) ;
  unsigned int cur = 0 ;
  for(int i=0; i<rhs.m(); i++) {
    A.rowptr[i] = cur ;
    const CNCSparseRowColumn & R = rhs.row(i) ;
    for(int jj=0; jj<R.nb_coeffs(); jj++) {
      if(!separate_diag || (R.coeff(jj).index != i )) {
        A.a[cur] = R.coeff(jj).a ;
        A.colind[cur] = R.coeff(jj).index ;
        cur++ ;
      }
    }
  }
  A.rowptr[rhs.m()] = nnz ;
  for(int i=0; i<rhs.diag_size(); i++) {
    A.diag[i] = rhs.diag(i) ;
  }
}

  
/****************************************************************************
 *
 ****************************************************************************/
void Cuda::cnc_cuda_set_dim_vec_from_n(long N){
//    printf("\n[Cuda::cnc_cuda_set_dim_vec_from_n] N: %ld", N);
  cnc_cuda_set_dim_vec ( (unsigned int)(sqrt((double)N)/THREAD_BLOCK_SIZE+1),
                         (unsigned int)(sqrt((double)N)/THREAD_BLOCK_SIZE+1),
                         THREAD_BLOCK_SIZE, THREAD_BLOCK_SIZE ) ;
    
//    printf("\n[cnc_cuda_set_dim_vec_from_n] GPU Vectors ALLOCATION") ;
  cublasAlloc(N+16, ELEM_SIZE, &gpu_r);
  cublasAlloc(N+16, ELEM_SIZE, &gpu_d);
  cublasAlloc(N+16, ELEM_SIZE, &gpu_h);
  cublasAlloc(N+16, ELEM_SIZE, &gpu_Ad);
  cublasAlloc(N+16, ELEM_SIZE, &gpu_diag_inv);
  cublasAlloc(N+16, ELEM_SIZE, &gpu_x);
  cublasAlloc(N+16, ELEM_SIZE, &gpu_b);
  cublasAlloc(N+16, ELEM_SIZE, &gpu_temp);
  cublasAlloc(N+16, ELEM_SIZE, &gpu_temp0);
  cublasAlloc(N+16, ELEM_SIZE, &gpu_temp1);
  cublasAlloc(N+16, ELEM_SIZE, &gpu_res0);
}

  
/****************************************************************************
 * Internal implementation of the Conjugate Gradient Solver on the GPU
 ****************************************************************************/
bool Cuda::solve(CNC_MatrixCRS<double> &A,
                 const CNC_Vector<double> &b,
                 CNC_Vector<double> &x,
                 const unsigned int nb_iter_max,
                 const double epsilon,
                 Integer& nb_iteration,
                 Real* residual_norm) {
  
  register const long N = x.size() ;
  
  // vars to be defined specifically for each storage format (CRS/BCRS...)
  const int size_matrix = A.a.size() ;
  const double * diag_matrix = A.diag.data() ;
  
//    printf("\n[solve_cg_internal] N: %ld", N);
//    printf("\n[solve_cg_internal] size matrix: %d", size_matrix);

  // matrix allocation and upload
  //printf("\n[solve_cg_internal] GPU Matrix allocation") ;
  A.gpu_upload() ;

  // building the Jacobi preconditionner
//    printf("\n[solve_cg_internal] Building the Jacobi preconditionner") ;
  CNC_Vector<double> cpu_diag_inv(N+16);
  for(long i=0; i<N; i++) {
    cpu_diag_inv[i] = (double)(((i >= N) || (diag_matrix[i] == 0.0)) ? 1.0 : 1.0 / diag_matrix[i]);
  }

//    printf("\n[solve_cg_internal] Setting vectors") ;
  cublasSetVector(N, ELEM_SIZE, cpu_diag_inv.data(), 1, gpu_diag_inv, 1) ;
  cublasSetVector(N, ELEM_SIZE, x.data(), 1, gpu_x, 1);
  cublasSetVector(N, ELEM_SIZE, b.data(), 1, gpu_b, 1);
//    printf("\n[solve_cg_internal] gpu x=");
//    gpu_vector_dump(gpu_x,N);
  //   printf("\n[solve_cg_internal] gpu b=");
//    gpu_vector_dump(gpu_b,N);
  //printf("\n[solve_cg_internal] SetComparison") ;
  //compare_vec(x.data(),false,(double*)gpu_x,true,N,0.0001);
  nb_iteration=0;
  double alpha, beta;
	
  // r=A*x
//    printf("\n[solve_cg_internal] r=A*x") ;
  A.gpu_mult(gpu_x, gpu_r, N);
    
//    printf("\n[solve_cg_internal] gpu_r=") ;
//    gpu_vector_dump(gpu_r,N);
    
  //r=b-A*x
//    printf("\n[solve_cg_internal] r=b-A*x") ;
  cublasDaxpy(N,-1.0f,(double*)gpu_b,1,(double*)gpu_r,1);
  cublasDscal(N,-1.0f,(double*)gpu_r,1);

//d=M-1*r
//    printf("\n[solve_cg_internal] d=M-1*r") ;
  cnc_cuda_vecvecmult(N,(double*)gpu_diag_inv,(double*)gpu_r,(double*)gpu_d);

//cur_err=rT*d
  double cur_err=cublasDdot(N,(double*)gpu_r,1,(double*)gpu_d,1);
//    printf("\n[solve_cg_internal] cur_err=%f", cur_err) ;

//err=cur_err
  double err=(double)(cur_err*epsilon*epsilon);
  while((cur_err>err)&&(nb_iteration<nb_iter_max)){
//      if(!(nb_iteration & 31u)) printf("\n\t%d : %f -- %f", nb_iteration, cur_err, err );
    // Ad = A*d
    A.gpu_mult(gpu_d, gpu_Ad, N );
    // alpha = cur_err / (dT*Ad)
    alpha = cur_err / cublasDdot(N, (double*)gpu_d, 1, (double*)gpu_Ad, 1 );
    // x = x + alpha * d
    cublasDaxpy(N, alpha, (double*)gpu_d, 1, (double*)gpu_x, 1 );
    // r = r - alpha * Ad
    cublasDaxpy(N, -alpha, (double*)gpu_Ad, 1, (double*)gpu_r, 1 );
    // h = M-1r
    cnc_cuda_vecvecmult(N, (double*)gpu_diag_inv, (double*)gpu_r, (double*)gpu_h );
    double old_err = cur_err;
    // cur_err = rT * h
    cur_err = cublasDdot(N, (double*)gpu_r, 1, (double*)gpu_h, 1 );
    beta = cur_err / old_err;
    // d = h + beta * d
    cublasDscal(N, beta, (double*)gpu_d, 1 );
    cublasDaxpy(N, 1.0f, (double*)gpu_h, 1, (double*)gpu_d, 1 );
    ++nb_iteration;
  }
    
  // Get back results
//    nb_iteration=its;
//    residual_norm[0]=cur_err;
  residual_norm[0]=math::sqrt(cublasDdot(N,(double*)gpu_r,1,(double*)gpu_r,1)/
                              cublasDdot(N,(double*)gpu_b,1,(double*)gpu_b,1));
    
//    printf("\n------------------------------------------------------------" );
/*    if(nb_iteration==nb_iter_max ) {
      printf("\nMaximum #itr reached: SOLVER DID NOT CONVERGE !!!" ); 
      printf("\n------------------------------------------------------------" );
      }
      printf("\nCG Used %d iterations", nb_iteration );
*/
    
//    long long int flop = 8*N+size_matrix*2+nb_iteration*(size_matrix*2+11*N+(long long int)(2.*N) );
//    printf("\nGPU whole CG GFlops: %lg", 1e-9 * flop);
  //   printf("\n############################################################\n\n" );


  cublasGetVector(N, ELEM_SIZE, (double*)gpu_x, 1, x.data(), 1);

  return (nb_iteration<nb_iter_max);
}


/******************************************************************************
 * AlephCudaManager::cublas_get_error
 *****************************************************************************/
void Cuda::cublas_get_error(cublasStatus_t st){
//  printf("\t[Aleph::Cuda::cublas_get_error]\n");
  if (st==CUBLAS_STATUS_SUCCESS) return;
  switch (st) {
  case CUBLAS_STATUS_NOT_INITIALIZED:	printf ("CUBLAS_STATUS_NOT_INITIALIZED\n");  break ;
  case CUBLAS_STATUS_ALLOC_FAILED:	   printf ("CUBLAS_STATUS_ALLOC_FAILED\n");	   break ;
  case CUBLAS_STATUS_INVALID_VALUE:	   printf ("CUBLAS_STATUS_INVALID_VALUE\n");	   break ;
  case CUBLAS_STATUS_MAPPING_ERROR:	   printf ("CUBLAS_STATUS_MAPPING_ERROR\n");	   break ;
  case CUBLAS_STATUS_EXECUTION_FAILED: printf ("CUBLAS_STATUS_EXECUTION_FAILED\n"); break ;
  case CUBLAS_STATUS_INTERNAL_ERROR:	printf ("CUBLAS_STATUS_INTERNAL_ERROR\n");   break ;
  default: printf ("unkown error message\n"); break ;
  }
  throw FatalErrorException("Cuda Cublas error!");
}


/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_END_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
