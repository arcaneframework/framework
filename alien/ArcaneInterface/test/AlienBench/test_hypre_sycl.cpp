
#define MPICH_SKIP_MPICXX 1
#include <mpi.h>


#include <boost/lexical_cast.hpp>
#include <boost/program_options/options_description.hpp>
#include <boost/program_options/parsers.hpp>
#include <boost/program_options/cmdline.hpp>
#include <boost/program_options/variables_map.hpp>

#include <_hypre_utilities.h>
#include <HYPRE_utilities.h>
#include <HYPRE.h>
#include <HYPRE_parcsr_mv.h>

#include <HYPRE_IJ_mv.h>
#include <HYPRE_parcsr_ls.h>
#include <HYPRE_parcsr_mv.h>

#include <vector>
#include <iostream>
#include <cstdio>


#include <cuda.h>
#include <cuda_runtime_api.h>

int hypre_sycl(int num_rows,
               int** num_cols_d, int** rows_d, int** cols_d, double** data_d, double** x_data_d,
               int* num_cols_h, int*rows_h, int* cols_h, double* data_h, double* x_data_h) ;

int hypre_sycl_buffer(int num_rows,
               int** num_cols_d, int** rows_d, int** cols_d, double** data_d,
               int* num_cols_h, int*rows_h, int* cols_h, double* data_h) ;

int main(int argc, char *argv[])
{

  // clang-format off
  using namespace boost::program_options ;
  options_description desc;
  desc.add_options()
      ("help",                                           "produce help")
      ("use-mem-device",  value<int>()->default_value(1),"use mem device")
      ("use-cuda",        value<int>()->default_value(0),"use cuda")
      ("use-sycl-ptr",    value<int>()->default_value(0),"use sycl ptr")
      ("use-sycl-buffer", value<int>()->default_value(0),"use sycl buf") ;
  // clang-format on

  variables_map vm;
  store(parse_command_line(argc, argv, desc), vm);
  notify(vm);

  if (vm.count("help")) {
    std::cout << desc << "\n";
    return 1;
  }

  // clang-format off
  bool use_mem_device  = vm["use-mem-device"].as<int>() == 1 ;
  bool use_cuda        = vm["use-cuda"].as<int>() == 1  ;
  bool use_sycl_ptr    = vm["use-sycl-ptr"].as<int>() == 1  ;
  bool use_sycl_buffer = vm["use-sycl-buffer"].as<int>() == 1  ;
  // clang-format on

  //cudaSetDevice(device_id); /* GPU binding */
  std::cout<<"MPI INIT "<<std::endl ;
  MPI_Init(&argc, &argv);
  std::cout<<"MPI INIT OK"<<std::endl ;

  if (!HYPRE_Initialized()){
    HYPRE_Initialize();
  }

  if(use_mem_device)
  {
     /* AMG in GPU memory (default) */
     HYPRE_SetMemoryLocation(HYPRE_MEMORY_DEVICE);
     /* setup AMG on GPUs */
     HYPRE_SetExecutionPolicy(HYPRE_EXEC_DEVICE);
  }
  else
  {
      HYPRE_SetMemoryLocation(HYPRE_MEMORY_HOST);
      HYPRE_SetExecutionPolicy(HYPRE_EXEC_HOST);
  }

   /* setup IJ matrix A */
   int first_row = 0 ;
   int last_row = 10 ;
   int first_col = 0 ;
   int last_col = 10 ;
   int num_rows = 10 ;
   int nnz      = 10 ;
   std::vector<int> num_cols(num_rows) ;
   std::vector<int> rows(num_rows) ;
   for(int i=0;i<num_rows;++i)
   {
     num_cols[i] = 1 ;
     rows[i] = i ;
   }
   std::vector<int> cols(nnz) ;
   std::vector<double> data(nnz) ;
   std::vector<double> x_values(num_rows) ;
   for(int i=0;i<nnz;++i)
   {
     cols[i] = i ;
     data[i] = 1.+i ;
   }

   for(int i=0;i<nnz;++i)
   {
       x_values[i] = 1. ;
   }

   HYPRE_IJMatrix ij_A;
   HYPRE_ParCSRMatrix  parcsr_A;
   HYPRE_IJMatrixCreate(MPI_COMM_WORLD, first_row, last_row, first_col, last_col, &ij_A);
   HYPRE_IJMatrixSetObjectType(ij_A, HYPRE_PARCSR);
   HYPRE_IJMatrixInitialize(ij_A);

   HYPRE_IJVector ij_X;
   HYPRE_ParVector  par_X;
   HYPRE_IJVectorCreate(MPI_COMM_WORLD, first_row, last_row, &ij_X);
   HYPRE_IJVectorSetObjectType(ij_X, HYPRE_PARCSR);
   HYPRE_IJVectorInitialize(ij_X);


   if(use_mem_device)
   {

     int* num_cols_d = nullptr;
     int* rows_d     = nullptr;
     int* cols_d     = nullptr;
     double* data_d  = nullptr;
     double* x_values_d  = nullptr;

     if(use_cuda)
     {
       std::cout<<"TEST WITH CUDA"<<std::endl;
       cudaMalloc((void**)&data_d, num_rows*sizeof(double));
       cudaMalloc((void**)&x_values_d, num_rows*sizeof(double));
       cudaMalloc((void**)&num_cols_d, num_rows*sizeof(int));
       cudaMalloc((void**)&rows_d, num_rows*sizeof(int));
       cudaMalloc((void**)&cols_d, num_rows*sizeof(int));

       cudaMemcpy(data_d, data.data(), num_rows*sizeof(double), cudaMemcpyHostToDevice);
       cudaMemcpy(x_values_d, x_values.data(), num_rows*sizeof(double), cudaMemcpyHostToDevice);
       cudaMemcpy(num_cols_d, num_cols.data(), num_rows*sizeof(int), cudaMemcpyHostToDevice);
       cudaMemcpy(rows_d, rows.data(), num_rows*sizeof(int), cudaMemcpyHostToDevice);
       cudaMemcpy(cols_d, cols.data(), num_rows*sizeof(int), cudaMemcpyHostToDevice);
      
       std::cout<<"HYPRE SET VALUES"<<std::endl;
       int ierr = HYPRE_IJMatrixSetValues(ij_A, num_rows, num_cols_d, rows_d, cols_d, data_d);
       std::cout<<"HYPRE SET VALUES ERR="<<ierr<<std::endl;

       HYPRE_IJVectorSetValues(ij_X, num_rows, rows_d, x_values_d);
     }
     
     if(use_sycl_ptr)
     {
       std::cout<<"TEST WITH SYCL PTR"<<std::endl;
       hypre_sycl(num_rows,
                  &num_cols_d,&rows_d,&cols_d,&data_d,&x_values_d,
                  num_cols.data(), rows.data(), cols.data(), data.data(), x_values.data()) ;
        
        std::cout<<"HYPRE SET VALUES"<<std::endl;
        int ierr = HYPRE_IJMatrixSetValues(ij_A, num_rows, num_cols_d, rows_d, cols_d, data_d);
        std::cout<<"HYPRE SET VALUES ERR="<<ierr<<std::endl;

        HYPRE_IJVectorSetValues(ij_X, num_rows, rows_d, x_values_d);
      }

     if(use_sycl_buffer)
     {
        std::cout<<"TEST WITH SYCL BUFFER"<<std::endl;
        hypre_sycl_buffer(num_rows,
                  &num_cols_d,&rows_d,&cols_d,&data_d,
                  num_cols.data(), rows.data(), cols.data(), data.data()) ;

        std::cout<<"HYPRE SET VALUES"<<std::endl;
        int ierr = HYPRE_IJMatrixSetValues(ij_A, num_rows, num_cols_d, rows_d, cols_d, data_d);
        std::cout<<"HYPRE SET VALUES ERR="<<ierr<<std::endl;
      }
   }
   else
   {
     
      std::cout<<"TEST ON HOST"<<std::endl;
      //HYPRE_IJMatrixSetRowSizes(m_internal, lineSizes.unguardedBasePointer());
      std::cout<<"HYPRE SET VALUES"<<std::endl;
      int ierr = HYPRE_IJMatrixSetValues(ij_A, num_rows, num_cols.data(), rows.data(), cols.data(), data.data());
      std::cout<<"HYPRE SET VALUES ERR="<<ierr<<std::endl;

      HYPRE_IJVectorSetValues(ij_X, num_rows, rows.data(), x_values.data());
   }
   HYPRE_IJMatrixAssemble(ij_A);
   HYPRE_IJVectorAssemble(ij_X);

   HYPRE_IJMatrixGetObject(ij_A, (void **) &parcsr_A);
   HYPRE_IJVectorGetObject(ij_X, (void **) &par_X);

   double dot_prod = 0;
   HYPRE_ParVectorInnerProd(par_X, par_X, &dot_prod);
   std::cout<<"DOT_PROD(X) : "<<dot_prod<<std::endl ;
   HYPRE_Finalize(); /* must be the last HYPRE function call */

   MPI_Finalize();
}

