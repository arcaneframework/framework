// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------

#include <cassert>

#include "SYCLVector.h"
#include "SYCLVectorInternal.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

using namespace Arccore;

namespace Alien
{

  //! Constructeur sans association un MultiImpl
  template <typename ValueT>
  SYCLVector<ValueT>::SYCLVector()
  : IVectorImpl(nullptr, AlgebraTraits<BackEnd::tag::sycl>::name())
  {}

  //! Constructeur avec association ? un MultiImpl
  template <typename ValueT>
  SYCLVector<ValueT>::SYCLVector(const MultiVectorImpl* multi_impl)
  : IVectorImpl(multi_impl, AlgebraTraits<BackEnd::tag::sycl>::name())
  {}


  template <typename ValueT>
  SYCLVector<ValueT>::~SYCLVector()
  {}

  template <typename ValueT>
  void SYCLVector<ValueT>::allocate()
  {
    //delete m_internal;
    auto block_size = blockSize() ;
    m_h_values.resize(m_local_size*block_size);
    m_internal.reset(new VectorInternal(m_h_values.data(), m_local_size*block_size));
  }

  template <typename ValueT>
  void SYCLVector<ValueT>::resize(Integer alloc_size) const
  {
    //delete m_internal;
    m_h_values.resize(alloc_size);
    m_internal.reset(new VectorInternal(m_h_values.data(), alloc_size));
  }

  template <typename ValueT>
  void SYCLVector<ValueT>::clear()
  {
    //delete m_internal;
    //m_internal = nullptr;
    m_internal.reset() ;
    std::vector<ValueType>().swap(m_h_values);
  }

  template <typename ValueT>
  void SYCLVector<ValueT>::setValuesFromHost()
  {
    //delete m_internal;
    m_internal.reset(new VectorInternal(m_h_values.data(), m_local_size*blockSize()));
  }

  template <typename ValueT>
  void SYCLVector<ValueT>::setValues(std::size_t size, ValueType const* ptr)
  {
    //delete m_internal;
    m_h_values.resize(m_local_size*blockSize());
    std::copy(ptr, ptr + size, m_h_values.begin());
    m_internal.reset(new VectorInternal(m_h_values.data(), m_local_size*blockSize()));
  }

  template <typename ValueT>
  void SYCLVector<ValueT>::setValuesFromDevice(std::size_t size, ValueType const* ptr)
  {
    assert(m_internal.get());
    m_internal->setValuesFromDevice(size,ptr);
  }

  template <typename ValueT>
  void SYCLVector<ValueT>::setValuesFromHost(std::size_t size, ValueType const* ptr)
  {
    assert(m_internal.get());
    m_internal->setValuesFromHost(size,ptr);
  }


  template <typename ValueT>
  void SYCLVector<ValueT>::copyValuesTo(std::size_t size, ValueType* ptr) const
  {
    if (m_internal.get())
      m_internal->copyValuesToHost(size, ptr);
  }

  template <typename ValueT>
  void
  SYCLVector<ValueT>::copyValuesToDevice(std::size_t size, ValueT* ptr) const
  {
    if (m_internal.get())
      m_internal->copyValuesToDevice(size, ptr);
  }



  template <typename ValueT>
  void SYCLVector<ValueT>::initDevicePointers(int** rows, ValueType** values) const
  {
    if(m_internal.get()==nullptr)
      return ;

    auto env = SYCLEnv::instance() ;
    auto& queue = env->internal()->queue() ;
    auto max_num_treads = env->maxNumThreads() ;

    auto alloc_size = m_local_size*blockSize() ;
    auto values_ptr = malloc_device<ValueT>(alloc_size, queue);
    auto rows_ptr   = malloc_device<IndexType>(alloc_size, queue);

    queue.submit( [&](sycl::handler& cgh)
                  {
                    auto access_x = m_internal->m_values.template get_access<sycl::access::mode::read>(cgh);
                    std::size_t y_length = alloc_size ;
                    cgh.parallel_for<class init_vector_ptr>(sycl::range<1>{max_num_treads}, [=] (sycl::item<1> itemId)
                                                      {
                                                          auto id = itemId.get_id(0);
                                                          for (auto i = id; i < y_length; i += itemId.get_range()[0])
                                                          {
                                                            values_ptr[i] = access_x[i];
                                                            rows_ptr[i] = IndexType(i) ;
                                                          }
                                                      });
                  });
    queue.wait() ;
    *values = values_ptr;
    *rows   = rows_ptr;
  }

  template <typename ValueT>
  void SYCLVector<ValueT>::freeDevicePointers(int* rows, ValueType* values)
  {
    auto env = SYCLEnv::instance() ;
    auto& queue = env->internal()->queue() ;
    sycl::free(values,queue) ;
    sycl::free(rows,queue) ;
  }

  template <typename ValueT>
  void SYCLVector<ValueT>::freeDevicePointers(ValueType* values)
  {
    auto env = SYCLEnv::instance() ;
    auto& queue = env->internal()->queue() ;
    sycl::free(values,queue) ;
  }

  template <typename ValueT>
  void SYCLVector<ValueT>::allocateDevicePointers(std::size_t alloc_size,
                                                  int** rows,
                                                  ValueType** values)
  {

    auto env = SYCLEnv::instance() ;
    auto& queue = env->internal()->queue() ;
    auto max_num_treads = env->maxNumThreads() ;

    auto values_ptr = malloc_device<ValueT>(alloc_size, queue);
    auto rows_ptr   = malloc_device<IndexType>(alloc_size, queue);
    queue.submit( [&](sycl::handler& cgh)
                  {
                    std::size_t y_length = alloc_size ;
                    cgh.parallel_for<class init2_vector_ptr>(sycl::range<1>{max_num_treads}, [=] (sycl::item<1> itemId)
                                                      {
                                                          auto id = itemId.get_id(0);
                                                          for (auto i = id; i < y_length; i += itemId.get_range()[0])
                                                          {
                                                            values_ptr[i] = 0.;
                                                            rows_ptr[i] = IndexType(i) ;
                                                          }
                                                      });
                  });

    queue.wait() ;
    *values = values_ptr;
    *rows   = rows_ptr;
  }


  template <typename ValueT>
  void SYCLVector<ValueT>::allocateDevicePointers(std::size_t alloc_size,
                                                  ValueType** values)
  {

    auto env = SYCLEnv::instance() ;
    auto& queue = env->internal()->queue() ;
    auto max_num_treads = env->maxNumThreads() ;

    auto values_ptr = malloc_device<ValueT>(alloc_size, queue);
    queue.submit( [&](sycl::handler& cgh)
                  {
                    std::size_t y_length = alloc_size ;
                    cgh.parallel_for<class init2_vector_ptr>(sycl::range<1>{max_num_treads}, [=] (sycl::item<1> itemId)
                                                      {
                                                          auto id = itemId.get_id(0);
                                                          for (auto i = id; i < y_length; i += itemId.get_range()[0])
                                                          {
                                                            values_ptr[i] = 0.;
                                                          }
                                                      });
                  });

    queue.wait() ;
    *values = values_ptr;
  }


  template <typename ValueT>
  void SYCLVector<ValueT>::initDevicePointers(std::size_t alloc_size,
                                              ValueType const* host_values,
                                              int** rows,
                                              ValueType** values)
  {

    auto env = SYCLEnv::instance() ;
    auto& queue = env->internal()->queue() ;
    auto max_num_treads = env->maxNumThreads() ;

    auto values_ptr = malloc_device<ValueT>(alloc_size, queue);
    auto rows_ptr   = malloc_device<IndexType>(alloc_size, queue);
    sycl::buffer<ValueT, 1> values_buf(host_values, sycl::range<1>(alloc_size)) ;
    queue.submit( [&](sycl::handler& cgh)
                  {
                    auto access_x = values_buf.template get_access<sycl::access::mode::read>(cgh);
                    std::size_t y_length = alloc_size ;
                    cgh.parallel_for<class init3_vector_ptr>(sycl::range<1>{max_num_treads}, [=] (sycl::item<1> itemId)
                                                      {
                                                          auto id = itemId.get_id(0);
                                                          for (auto i = id; i < y_length; i += itemId.get_range()[0])
                                                          {
                                                            values_ptr[i] = access_x[i];
                                                            rows_ptr[i] = IndexType(i) ;
                                                          }
                                                      });
                  });

    queue.wait() ;
    *values = values_ptr;
    *rows   = rows_ptr;
  }


  template <typename ValueT>
  void SYCLVector<ValueT>::copyDeviceToHost(std::size_t alloc_size,
                                            ValueType const* device_values,
                                            ValueType* host_values)
  {

    auto env = SYCLEnv::instance() ;
    auto& queue = env->internal()->queue() ;
    auto max_num_treads = env->maxNumThreads() ;

    sycl::buffer<ValueT, 1> values_buf(host_values, sycl::range<1>(alloc_size)) ;
    queue.submit( [&](sycl::handler& cgh)
                  {
                    auto access_x = sycl::accessor { values_buf, cgh, sycl::write_only, sycl::property::no_init{}};
                    //auto access_x = values_buf.template get_access<sycl::access::mode::write>(cgh);
                    std::size_t y_length = alloc_size ;
                    cgh.parallel_for<class copy_vector_ptr>(sycl::range<1>{max_num_treads}, [=] (sycl::item<1> itemId)
                                                      {
                                                          auto id = itemId.get_id(0);
                                                          for (auto i = id; i < y_length; i += itemId.get_range()[0])
                                                          {
                                                              access_x[i] = device_values[i];
                                                          }
                                                      });
                  });

    queue.wait() ;
  }

  template <typename ValueT>
  void SYCLVector<ValueT>::pointWiseMult(SYCLVector const& y, SYCLVector& z) const
  {
    auto block_size = blockSize() ;
    auto block_size_y = y.blockSize() ;
    auto block_size_z = z.blockSize() ;
    assert(block_size_y==block_size_z) ;
    if(block_size==1)
      m_internal->pointWiseMult(y.m_internal->m_values,
                                z.m_internal->m_values) ;
    else
    {
      assert(block_size==block_size_z*block_size_z) ;
      m_internal->blockMult(m_local_size,
                            block_size_z,
                            y.m_internal->m_values,
                            z.m_internal->m_values) ;
    }
  }
  /*---------------------------------------------------------------------------*/

  template class ALIEN_EXPORT SYCLVector<Real>;

  /*---------------------------------------------------------------------------*/

} // namespace Alien

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
