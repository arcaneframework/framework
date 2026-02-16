// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------

#pragma once

#include <alien/kernels/sycl/SYCLPrecomp.h>

#ifdef USE_SYCL2020
#include <sycl/sycl.hpp>
#else
#include <CL/sycl.hpp>
#endif

#include "SYCLEnv.h"
#include "SYCLEnvInternal.h"
/*---------------------------------------------------------------------------*/

namespace Alien::SYCLInternal
{

/*---------------------------------------------------------------------------*/

#ifndef USE_SYCL2020
  using namespace cl ;
#endif

template <typename ValueT = Real>
class VectorInternal
{
 public:
  // clang-format off
  typedef ValueT                           ValueType;
  typedef VectorInternal<ValueType>        ThisType;
  typedef sycl::buffer<ValueType, 1>       ValueBufferType;
  typedef std::unique_ptr<ValueBufferType> ValueBufferPtrType;
  // clang-format on

 public:
  VectorInternal(ValueType const* ptr, std::size_t size)
  : m_values(ptr, sycl::range<1>(size))
  {
    m_values.set_final_data(nullptr);
  }

  virtual ~VectorInternal() {}

  ValueBufferType& values()
  {
    return m_values;
  }

  ValueBufferType& values() const
  {
    return m_values;
  }

  ValueBufferType& ghostValues(Integer ghost_size) const
  {
    if (m_ghost_values.get() == nullptr || ghost_size > m_ghost_size) {
      m_ghost_size = ghost_size;
      m_ghost_values.reset(new ValueBufferType(m_ghost_size));
    }
    return *m_ghost_values;
  }

  void copyValuesToHost(std::size_t size, ValueT* ptr)
  {
    auto h_values = m_values.get_host_access();
    for (std::size_t i = 0; i < size; ++i)
      ptr[i] = h_values[i];
  }

  void copyValuesToDevice(std::size_t size, ValueT* ptr) const
  {
    auto env = SYCLEnv::instance() ;
    auto& queue = env->internal()->queue() ;
    auto max_num_treads = env->maxNumThreads() ;

    queue.submit( [&](sycl::handler& cgh)
                  {
                    auto access_x = m_values.template get_access<sycl::access::mode::read>(cgh);
                    std::size_t y_length = size ;
                    cgh.parallel_for<class init_vector_ptr>(sycl::range<1>{max_num_treads}, [=] (sycl::item<1> itemId)
                                                      {
                                                          auto id = itemId.get_id(0);
                                                          for (auto i = id; i < y_length; i += itemId.get_range()[0])
                                                            ptr[i] = access_x[i];
                                                      });
                  });
    queue.wait() ;
  }

  void setValuesFromDevice(std::size_t size, ValueT const* ptr) const
  {
    auto env = SYCLEnv::instance() ;
    auto& queue = env->internal()->queue() ;
    auto max_num_treads = env->maxNumThreads() ;

    queue.submit( [&](sycl::handler& cgh)
                  {
                    auto access_x = m_values.template get_access<sycl::access::mode::discard_write>(cgh);
                    std::size_t y_length = size ;
                    cgh.parallel_for<class init_vector_ptr>(sycl::range<1>{max_num_treads}, [=] (sycl::item<1> itemId)
                                                      {
                                                          auto id = itemId.get_id(0);
                                                          for (auto i = id; i < y_length; i += itemId.get_range()[0])
                                                            access_x[i] = ptr[i];
                                                      });
                  });
    queue.wait() ;
    /*
    {
      sycl::host_accessor<ValueT, 1, sycl::access::mode::read> vec_acc(m_values);
      for(int irow=0;irow<size;++irow)
      {
         std::cout<<"VEC["<<irow<<"]"<<vec_acc[irow]<<std::endl;
      }
    }*/
  }

  void setValuesFromHost(std::size_t size, ValueT const* ptr) const
  {
    auto env = SYCLEnv::instance() ;
    auto& queue = env->internal()->queue() ;
    auto max_num_treads = env->maxNumThreads() ;

    auto rhs = ValueBufferType(ptr,sycl::range<1>(size)) ;

    queue.submit( [&](sycl::handler& cgh)
                  {
                    auto access_x = m_values.template get_access<sycl::access::mode::discard_write>(cgh);
                    auto access_rhs = rhs.template get_access<sycl::access::mode::read>(cgh);
                    std::size_t y_length = size ;
                    cgh.parallel_for<class init_vector_ptr>(sycl::range<1>{max_num_treads}, [=] (sycl::item<1> itemId)
                                                      {
                                                          auto id = itemId.get_id(0);
                                                          for (auto i = id; i < y_length; i += itemId.get_range()[0])
                                                            access_x[i] = access_rhs[i];
                                                      });
                  });
    queue.wait() ;
    /*
    {
      sycl::host_accessor<ValueT, 1, sycl::access::mode::read> vec_acc(m_values);
      for(int irow=0;irow<size;++irow)
      {
         std::cout<<"VEC["<<irow<<"]"<<vec_acc[irow]<<std::endl;
      }
    }*/
  }

  void copy(ValueBufferType& src)
  {
    auto env = SYCLEnv::instance() ;
    env->internal()->queue().submit([&](sycl::handler& cgh)
                                     {
                                       auto access_x = m_values.template get_access<sycl::access::mode::read_write>(cgh);
                                       auto access_src = src.template get_access<sycl::access::mode::read>(cgh);
                                       cgh.copy(access_src,access_x) ;
                                     }) ;
  }

  void pointWiseMult(ValueBufferType& y, ValueBufferType& z)
  {
    auto env = SYCLEnv::instance() ;
    auto& queue = env->internal()->queue() ;
    auto max_num_treads = env->maxNumThreads() ;

    queue.submit( [&](sycl::handler& cgh)
                  {
                    auto access_x = m_values.template get_access<sycl::access::mode::read>(cgh);
                    auto access_y = y.template get_access<sycl::access::mode::read>(cgh);
                    auto access_z = z.template get_access<sycl::access::mode::discard_write>(cgh);
                    std::size_t x_length = m_values.size() ;
                    cgh.parallel_for<class init_vector_ptr>(sycl::range<1>{max_num_treads}, [=] (sycl::item<1> itemId)
                                                      {
                                                          auto id = itemId.get_id(0);
                                                          for (auto i = id; i < x_length; i += itemId.get_range()[0])
                                                            access_z[i] = access_x[i]*access_y[i];
                                                      });
                  });
  }

  void blockMult(std::size_t nrows,
                 int block_size,
                 ValueBufferType& y,
                 ValueBufferType& z)
  {
    auto env = SYCLEnv::instance() ;
    auto& queue = env->internal()->queue() ;
    auto max_num_treads = env->maxNumThreads() ;
    int N = block_size ;
    int NxN = N*N ;
    assert(m_values.size()>=nrows*NxN) ;
    assert(y.size()>=nrows*N) ;
    assert(z.size()>=nrows*N) ;
    queue.submit( [&](sycl::handler& cgh)
                  {
                    auto access_x = m_values.template get_access<sycl::access::mode::read>(cgh);
                    auto access_y = y.template get_access<sycl::access::mode::read>(cgh);
                    auto access_z = z.template get_access<sycl::access::mode::discard_write>(cgh);
                    cgh.parallel_for<class vector_block_mult>(
                        sycl::range<1>{max_num_treads},
                        [=] (sycl::item<1> itemId)
                        {
                            auto id = itemId.get_id(0);
                            for (auto irow = id; irow < nrows; irow += itemId.get_range()[0])
                            {
                                for(int ieq=0;ieq<N;++ieq)
                                {
                                  ValueType value = 0. ;
                                  for(int j=0;j<N;++j)
                                  {
                                     value += access_x[irow*NxN+ieq*N+j]*access_y[irow*N+j] ;
                                  }
                                  access_z[irow*N+ieq] = value ;
                                }
                            }
                        });
                  });
    /*
    {
      sycl::host_accessor<ValueT, 1, sycl::access::mode::read> diag_acc(m_values);
      sycl::host_accessor<ValueT, 1, sycl::access::mode::read> z_acc(z);

      for (std::size_t irow = 0; irow < nrows; ++irow)
      {
        std::cout<<"INV DIAG["<<irow<<"]:\n";
        for(int i=0;i<N;++i)
        {
          for(int j=0;j<N;++j)
            std::cout<<diag_acc[irow*NxN+i*N+j]<<" ";
          std::cout<<std::endl;
        }
        std::cout<<"Y["<<irow<<"]=\n";
        for(int i=0;i<N;++i)
          std::cout<<z_acc[irow*N+i]<<std::endl;
      }
    }*/
  }


  //VectorInternal<ValueT>* clone() const { return new VectorInternal<ValueT>(*this); }

  // clang-format off
  mutable ValueBufferType    m_values;

  mutable Integer            m_ghost_size = 0 ;
  mutable ValueBufferPtrType m_ghost_values;
  // clang-format on
};

/*---------------------------------------------------------------------------*/

} // namespace Alien::SYCLInternal

/*---------------------------------------------------------------------------*/
