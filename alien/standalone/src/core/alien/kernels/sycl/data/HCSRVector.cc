// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2024 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------


#include <alien/kernels/sycl/SYCLPrecomp.h>


#include <alien/kernels/sycl/data/HCSRVectorInternal.h>

namespace Alien {

template <typename ValueT>
typename HCSRVector<ValueT>::ValueType const*
HCSRVector<ValueT>::dataPtr() const
{
  if(m_internal.get()==nullptr)
    return nullptr ;

  auto env = SYCLEnv::instance() ;
  auto& queue = env->internal()->queue() ;
  auto max_num_treads = env->maxNumThreads() ;

  auto values = malloc_device<ValueT>(m_local_size, queue);
  
  queue.submit( [&](sycl::handler& cgh)
                {
                  auto access_x = m_internal->m_values.template get_access<sycl::access::mode::read>(cgh);
                  std::size_t y_length = m_local_size ;
                  cgh.parallel_for<class init_vector_ptr>(sycl::range<1>{max_num_treads}, [=] (sycl::item<1> itemId)
                                                    {
                                                        auto id = itemId.get_id(0);
                                                        for (auto i = id; i < y_length; i += itemId.get_range()[0])
                                                          values[i] = access_x[i];
                                                    });
                });
  queue.wait() ;
  return values ;
}
template <typename ValueT>
void HCSRVector<ValueT>::initDevicePointers(int** rows, ValueType** values) const
{
  if(m_internal.get()==nullptr)
    return ;

  auto env = SYCLEnv::instance() ;
  auto& queue = env->internal()->queue() ;
  auto max_num_treads = env->maxNumThreads() ;

  auto values_ptr = malloc_device<ValueT>(m_local_size, queue);
  auto rows_ptr   = malloc_device<IndexType>(m_local_size, queue);
 
  queue.submit( [&](sycl::handler& cgh)
                {
                  auto access_x = m_internal->m_values.template get_access<sycl::access::mode::read>(cgh);
                  std::size_t y_length = m_local_size ;
                  cgh.parallel_for<class init_hcsrvector_ptr>(sycl::range<1>{max_num_treads}, [=] (sycl::item<1> itemId)
                                                    {
                                                        auto id = itemId.get_id(0);
                                                        for (auto i = id; i < y_length; i += itemId.get_range()[0])
                                                        {
                                                          values_ptr[i] = access_x[i];
                                                          rows_ptr[i] = i ;
                                                        }
                                                    });
                });
  queue.wait() ;
  *values = values_ptr;
  *rows   = rows_ptr;
}

template <typename ValueT>
void HCSRVector<ValueT>::freeDevicePointers(int* rows, ValueType* values) const
{
  auto env = SYCLEnv::instance() ;
  auto& queue = env->internal()->queue() ;
  sycl::free(values,queue) ;
  sycl::free(rows,queue) ;
}


template class HCSRVector<Real>;

} // namespace Alien

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
