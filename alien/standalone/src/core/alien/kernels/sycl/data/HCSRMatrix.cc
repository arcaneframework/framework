// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2024 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------


#include <alien/kernels/sycl/SYCLPrecomp.h>


#include <alien/kernels/sycl/data/HCSRMatrix.h>
#include <alien/kernels/sycl/data/HCSRMatrixInternal.h>

namespace Alien {

template <typename ValueT>
void HCSRMatrix<ValueT>::
initDevicePointers(int** ncols, int** rows, int** cols, ValueT** values) const
{
   auto& hypre_profile = m_internal->getHypreProfile(m_local_offset) ;
   auto env = SYCLEnv::instance() ;
   auto max_num_treads = env->maxNumThreads() ;
   auto nnz = m_profile->getNnz() ;
   auto& queue = env->internal()->queue() ;
   auto values_ptr = malloc_device<ValueT>(nnz, queue);
   auto ncols_ptr  = malloc_device<IndexType>(m_local_size, queue);
   auto rows_ptr   = malloc_device<IndexType>(m_local_size, queue);
   auto cols_ptr   = malloc_device<IndexType>(nnz, queue);
  
   queue.submit( [&](sycl::handler& cgh)
                {
                  auto access_x = m_internal->m_values.template get_access<sycl::access::mode::read>(cgh);
                  auto access_cols = m_internal->m_cols.template get_access<sycl::access::mode::read>(cgh);
                  auto y_length = nnz ;
                  cgh.parallel_for<class init_ptr>(sycl::range<1>{max_num_treads}, [=] (sycl::item<1> itemId)
                                                    {
                                                        auto id = itemId.get_id(0);
                                                        for (auto i = id; i < y_length; i += itemId.get_range()[0])
                                                        {
                                                          values_ptr[i] = access_x[i];
                                                          cols_ptr[i]   = access_cols[i];
                                                        }
                                                    });
                });
  

   queue.submit( [&](sycl::handler& cgh)
                {
                  auto access_ncols = hypre_profile.m_ncols.template get_access<sycl::access::mode::read>(cgh);
                  auto access_rows  = hypre_profile.m_rows.template get_access<sycl::access::mode::read>(cgh);
                  auto y_length = m_local_size ;
                  cgh.parallel_for<class init_ptr2>(sycl::range<1>{max_num_treads}, [=] (sycl::item<1> itemId)
                                                    {
                                                        auto id = itemId.get_id(0);
                                                        for (auto i = id; i < y_length; i += itemId.get_range()[0])
                                                        {
                                                          ncols_ptr[i] = access_ncols[i];
                                                          rows_ptr[i]  = access_rows[i];
                                                        }
                                                    });
                }) ;
   queue.wait() ;
  
   *values = values_ptr ;
   *cols   = cols_ptr ;
   *ncols  = ncols_ptr ;
   *rows   = rows_ptr ;
}

template <typename ValueT>
void HCSRMatrix<ValueT>::freeDevicePointers(int* ncols, int* rows, int* cols, ValueT* values) const
{  
  auto env = SYCLEnv::instance() ;
  auto& queue = env->internal()->queue() ;
  sycl::free(values,queue) ;
  sycl::free(ncols,queue) ;
  sycl::free(rows,queue) ;
  sycl::free(cols,queue) ;
}

template class HCSRMatrix<Real>;

} // namespace Alien

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
