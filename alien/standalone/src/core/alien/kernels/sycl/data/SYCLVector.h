// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------

#pragma once

#include <vector>
#include <alien/core/impl/IVectorImpl.h>
#include <alien/data/ISpace.h>
#include <alien/kernels/sycl/SYCLBackEnd.h>
#include <alien/kernels/sycl/SYCLPrecomp.h>
#include <iostream>

/*---------------------------------------------------------------------------*/

namespace Alien
{

namespace SYCLInternal
{
  template <typename ValueT>
  class VectorInternal;
}

template <typename ValueT>
class ALIEN_EXPORT SYCLVector : public IVectorImpl
{
 public:
  typedef ValueT  ValueType;
  typedef Integer IndexType;

  typedef SYCLInternal::VectorInternal<ValueType> VectorInternal;

  //! Constructeur sans association un MultiImpl
  SYCLVector() ;

  //! Constructeur avec association ? un MultiImpl
  SYCLVector(const MultiVectorImpl* multi_impl) ;

  //virtual ~SYCLVector();

  VectorInternal* internal()
  {
    return m_internal.get();
  }

  VectorInternal const* internal() const
  {
    return m_internal.get();
  }

  Integer getAllocSize() const
  {
    return m_local_size;
  }

  void allocate();

  void resize(Integer alloc_size) const;

  void clear();

  void init(const VectorDistribution& dist, const bool need_allocate)
  {
    //alien_debug([&] { cout() << "Initializing SYCLVector " << this; });
    if (this->m_multi_impl) {
      m_local_size = this->scalarizedLocalSize();
    }
    else {
      // Not associated vector
      m_own_distribution = dist;
      m_local_size = m_own_distribution.localSize();
    }
    if (need_allocate) {
      allocate();
    }
    //alien_debug([&] { cout() << "After Initializing SYCLVector " << m_local_size<<" "<<m_h_values.size(); });
    //Universe().traceMng()->flush() ;
  }

  const VectorDistribution& distribution() const
  {
    if (this->m_multi_impl)
      return IVectorImpl::distribution();
    else
      return m_own_distribution;
  }

  Arccore::Integer scalarizedLocalSize() const
  {
    if (this->m_multi_impl)
      return IVectorImpl::scalarizedLocalSize();
    else
      return m_own_distribution.localSize();
  }

  Arccore::Integer scalarizedGlobalSize() const
  {
    if (this->m_multi_impl)
      return IVectorImpl::scalarizedGlobalSize();
    else
      return m_own_distribution.globalSize();
  }

  Arccore::Integer scalarizedOffset() const
  {
    if (this->m_multi_impl)
      return IVectorImpl::scalarizedOffset();
    else
      return m_own_distribution.offset();
  }

  ValueType* getDataPtr() { return m_h_values.data(); }
  ValueType* data() { return m_h_values.data(); }

  ValueType const* getDataPtr() const { return m_h_values.data(); }
  ValueType const* data() const { return m_h_values.data(); }
  ValueType const* getAddressData() const { return m_h_values.data(); }

  void initDevicePointers(int** rows, ValueType** values) const ;

  static void allocateDevicePointers(std::size_t local_size,
                                     int** rows,
                                     ValueType** values);
  static void freeDevicePointers(int* rows, ValueType* values);

  static void initDevicePointers(std::size_t local_size,
                                 ValueType const* host_values,
                                 int** rows,
                                 ValueType** values) ;

  static void copyDeviceToHost(std::size_t local_size,
                               ValueType const* device_values,
                               ValueType* host_values) ;

  template <typename LambdaT>
  void apply(LambdaT const& lambda)
  {
    for (std::size_t i = 0; i < m_local_size; ++i) {
      m_h_values[i] = lambda(i);
    }
    setValuesFromHost();
  }

  void setValues(std::size_t size, ValueType const* ptr);

  void setValuesFromHost();

  void copyValuesTo(std::size_t size, ValueType* ptr) const;

  void copyValuesToDevice(std::size_t size, ValueType* ptr) const;

  void copyValuesToDevice(ValueType* ptr) const
  {
    copyValuesToDevice(m_local_size,ptr) ;
  }

  // FIXME: not implemented !
  template <typename E>
  SYCLVector& operator=(E const& expr);

 private:
  // clang-format off
  //mutable VectorInternal*        m_internal = nullptr;
  mutable std::unique_ptr<VectorInternal> m_internal ;
  mutable std::vector<ValueType>          m_h_values ;
  std::size_t                             m_local_size = 0;
  VectorDistribution                      m_own_distribution ;
  // clang-format on
};

//extern template class SYCLVector<Real>;
} // namespace Alien
