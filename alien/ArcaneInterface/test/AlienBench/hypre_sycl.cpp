
#define MPICH_SKIP_MPICXX 1
#include "mpi.h"

#include <vector>
#include <cstdio>

#ifdef ALIEN_USE_SYCL
#ifdef USE_SYCL2020
#include <sycl/sycl.hpp>
#else
#include <CL/sycl.hpp>
#endif
#endif

int hypre_sycl(int num_rows,
               int** num_cols_d, int** rows_d, int** cols_d, double** data_d, double** x_data_d,
               int* num_cols_h, int*rows_h, int* cols_h, double* data_h, double* x_data_h)
{
#ifdef ALIEN_USE_SYCL
   sycl::queue Q(sycl::gpu_selector{});

   *data_d     = malloc_device<double>(num_rows, Q);
   *x_data_d   = malloc_device<double>(num_rows, Q);
   *num_cols_d = malloc_device<int>(num_rows, Q);
   *rows_d     = malloc_device<int>(num_rows, Q);
   *cols_d     = malloc_device<int>(num_rows, Q);

   // in a handler
   Q.submit([&](sycl::handler& cgh){
     // untyped API
     cgh.memcpy(*data_d, data_h, num_rows*sizeof(double));
     //cgh.copy(data_d, data_h, num_rows);
   });

   Q.submit([&](sycl::handler& cgh){
     // untyped API
     cgh.memcpy(*x_data_d, x_data_h, num_rows*sizeof(double));
     //cgh.copy(data_d, data_h, num_rows);
   });

   Q.submit([&](sycl::handler& cgh){
     cgh.memcpy(*num_cols_d, num_cols_h, num_rows*sizeof(int));
   });
   Q.submit([&](sycl::handler& cgh){
     cgh.memcpy(*rows_d, rows_h, num_rows*sizeof(int));
   });
   Q.submit([&](sycl::handler& cgh){
     cgh.memcpy(*cols_d, cols_h, num_rows*sizeof(int));
     // or typed API
     //cgh.copy(x_d, x_h.data(), N);
   });
   Q.wait() ;
#endif
   return 0 ;
}

int hypre_sycl_buffer(int num_rows,
               int** num_cols_d, int** rows_d, int** cols_d, double** data_d,
               int* num_cols_h, int*rows_h, int* cols_h, double* data_h)
{
#ifdef ALIEN_USE_SYCL
   std::cout<<"PTR IN "<<*data_d<<std::endl ;

   sycl::queue Q(sycl::gpu_selector{});

   *data_d     = malloc_device<double>(num_rows, Q);
   *num_cols_d = malloc_device<int>(num_rows, Q);
   *rows_d     = malloc_device<int>(num_rows, Q);
   *cols_d     = malloc_device<int>(num_rows, Q);
     {
        sycl::buffer<double, 1> data_b(data_h,sycl::range<1>(num_rows)) ;
        sycl::buffer<int, 1> num_cols_b(num_cols_h,sycl::range<1>(num_rows)) ;
        sycl::buffer<int, 1> rows_b(rows_h,sycl::range<1>(num_rows)) ;
        sycl::buffer<int, 1> cols_b(cols_h,sycl::range<1>(num_rows)) ;
        {
          Q.submit([&](sycl::handler& cgh){

                      //auto data_acc = sycl::accessor{data_b, cgh} ;
                      auto data_acc = data_b.get_access<sycl::access::mode::read>(cgh);
                      cgh.copy(data_acc,*data_d) ;
                      //*data_d = data_acc.get_pointer().get() ;
                      //std::cout<<"PTR DATA"<<*data_acc.get_pointer().get()<<std::endl ;
                   }) ;
          Q.submit([&](sycl::handler& cgh){

                      auto acc = rows_b.get_access<sycl::access::mode::read>(cgh);
                      cgh.copy(acc,*num_cols_d) ;
                   }) ;
          Q.submit([&](sycl::handler& cgh){

                      auto acc = cols_b.get_access<sycl::access::mode::read>(cgh);
                      cgh.copy(acc,*cols_d) ;
                   }) ;
          Q.submit([&](sycl::handler& cgh){

                      auto acc = num_cols_b.get_access<sycl::access::mode::read>(cgh);
                      cgh.copy(acc,*num_cols_d) ;
                   }) ;
          Q.wait() ;
        }
        std::cout<<"PTR OUT "<<*data_d<<std::endl ;
     }
#endif
     return 0 ;
}

