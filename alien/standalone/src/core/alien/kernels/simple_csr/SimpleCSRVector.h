/*
 * Copyright 2020 IFPEN-CEA
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <alien/core/block/VBlockOffsets.h>
#include <alien/core/impl/IVectorImpl.h>
#include <alien/data/ISpace.h>
#include <alien/kernels/simple_csr/SimpleCSRBackEnd.h>
#include <alien/kernels/simple_csr/SimpleCSRPrecomp.h>
#include <iostream>

/*---------------------------------------------------------------------------*/

namespace Alien
{

/*---------------------------------------------------------------------------*/

template <typename ValueT>
class SimpleCSRVector : public IVectorImpl
{
 public:
  typedef ValueT ValueType;

  //! Constructeur sans association ? un MultiImpl
  SimpleCSRVector()
  : IVectorImpl(nullptr, AlgebraTraits<BackEnd::tag::simplecsr>::name())
  , m_local_size(0)
  , m_vblock(nullptr)
  {}

  //! Constructeur avec association ? un MultiImpl
  SimpleCSRVector(const MultiVectorImpl* multi_impl)
  : IVectorImpl(multi_impl, AlgebraTraits<BackEnd::tag::simplecsr>::name())
  , m_local_size(0)
  , m_vblock(nullptr)
  {}

  void allocate()
  {
    m_values.resize(m_local_size);
    if (this->vblock())
      m_vblock = new VBlockImpl(*this->vblock(), this->distribution());
  }

  void resize(Integer alloc_size) const
  {
    if (alloc_size > m_local_size)
      m_values.resize(alloc_size);
    if (this->vblock()) {
      delete m_vblock;
      m_vblock = new VBlockImpl(*this->vblock(), this->distribution());
    }
  }

  Integer getAllocSize() const { return m_values.size(); }

  void clear()
  {
    m_values.dispose();
    delete m_vblock;
    m_vblock = nullptr;
  }

  // values on local part
  Arccore::ArrayView<ValueType> values() { return m_values.subView(0, m_local_size); }
  Arccore::ConstArrayView<ValueType> values() const
  {
    return m_values.subConstView(0, m_local_size);
  }

  // Algebra adds ghost values
  ArrayView<ValueType> fullValues() { return m_values; }
  ConstArrayView<ValueType> fullValues() const { return m_values; }

  void setArrayValues(UniqueArray<ValueType>&& rhs) { m_values.copy(rhs); }

  UniqueArray<ValueType> const& getArrayValues() const { return m_values; }

  ValueType* getDataPtr() { return m_values.data(); }
  ValueType* data() { return m_values.data(); }

  ValueType const* getDataPtr() const { return m_values.data(); }
  ValueType const* data() const { return m_values.data(); }
  ValueType const* getAddressData() const { return m_values.data(); }

  ValueType& operator[](Integer index) { return m_values[index]; }

  ValueType const& operator[](Integer index) const { return m_values[index]; }

  // FIXME: not implemented !
  template <typename E>
  SimpleCSRVector& operator=(E const& expr);

  void init(const VectorDistribution& dist,
            const bool need_allocate) override
  {
    alien_debug([&] { cout() << "Initializing SimpleCSRVector " << this; });
    if (this->m_multi_impl) {
      if (this->vblock()) {
        delete m_vblock;
        m_vblock = new VBlockImpl(*this->vblock(), this->distribution());
      }
      m_local_size = this->scalarizedLocalSize();
    }
    else {
      // Not associated vector
      m_own_distribution = dist;
      m_local_size = m_own_distribution.localSize();
    }
    if (need_allocate) {
      m_values.resize(m_local_size);
      m_values.fill(ValueT());
    }
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

  const VBlockImpl& vblockImpl() const { return *m_vblock; }

  //@{ @name Interface ? soi-m?me
  void update(const SimpleCSRVector<ValueT>& v)
  {
    ALIEN_ASSERT((this == &v), ("Unexpected error"));
  }
  //@}

  SimpleCSRVector<ValueT>* cloneTo(const MultiVectorImpl* impl) const
  {
    SimpleCSRVector<ValueT>* vector = new SimpleCSRVector<ValueT>(impl);
    vector->init(this->distribution(), true);
    ConstArrayView<ValueType> thisValues = this->fullValues();
    vector->resize(thisValues.size());
    for (Integer i = 0; i < thisValues.size(); ++i)
      (*vector)[i] = m_values[i];
    return vector;
  }

 private:
  mutable UniqueArray<ValueT> m_values;
  Integer m_local_size = 0;
  mutable VBlockImpl* m_vblock = nullptr;
  VectorDistribution m_own_distribution;
};

/*---------------------------------------------------------------------------*/

} // namespace Alien

/*---------------------------------------------------------------------------*/
