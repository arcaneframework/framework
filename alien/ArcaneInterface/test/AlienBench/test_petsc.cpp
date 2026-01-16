
#define MPICH_SKIP_MPICXX 1
#include <mpi.h>


#include <boost/lexical_cast.hpp>
#include <boost/program_options/options_description.hpp>
#include <boost/program_options/parsers.hpp>
#include <boost/program_options/cmdline.hpp>
#include <boost/program_options/variables_map.hpp>


#include <vector>
#include <iostream>
#include <cstdio>


#include <petscmat.h>

#include <cuda.h>
#include <cuda_runtime_api.h>


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
  MPI_Comm m_mpi_main_communicator MPI_COMM_NULL;
  MPI_Comm_dup(MPI_COMM_WORLD,&m_mpi_main_communicator);
  //m_main_communicator = MP::Communicator(m_mpi_main_communicator);
  int rank, size;
  MPI_Comm_rank(m_mpi_main_communicator,&rank);
  MPI_Comm_size(m_mpi_main_communicator,&size);
  std::cout<<"MpiParallelSuperMng::build RANK SIZE"<<rank<<" "<<size<<std::endl ;

  MPI_Comm mpi_machine_communicator = MPI_COMM_NULL;
  MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &mpi_machine_communicator);

  PetscInitialize(&argc, &argv, NULL, "PETSc Initialisation");

  // Reduce memory due to log for graphical viewer
  PetscLogActions(PETSC_FALSE);
  PetscLogObjects(PETSC_FALSE);


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
      
     }
   }
   else
   {
     
      std::cout<<"TEST ON HOST"<<std::endl;
      //HYPRE_IJMatrixSetRowSizes(m_internal, lineSizes.unguardedBasePointer());
      std::cout<<"HYPRE SET VALUES"<<std::endl;
   }

   MPI_Finalize();
}

