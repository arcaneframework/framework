// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------

#pragma once

#include <alien/core/block/VBlockOffsets.h>
#include <alien/core/impl/IVectorImpl.h>
#include <alien/core/impl/MultiVectorImpl.h>
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

  Integer blockSize() const
  {
    if (block())
    {
       return block()->size();
    }
    else if (vblock()) {
      return -1 ;
    }
    else {
      return 1 ;
    }
  }

  void setBlockSize(Integer block_size)
  {
    if(this->m_multi_impl)
      const_cast<MultiVectorImpl*>(this->m_multi_impl)->setBlockInfos(block_size) ;
    else
      m_own_block_size = block_size ;
  }

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

  void clear() override
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
      m_values.resize(m_local_size*m_own_block_size);
      m_values.fill(ValueT());
    }
  }

  void init(const VectorDistribution& dist,
            Integer block_size,
            const bool need_allocate)
  {
    alien_debug([&] { cout() << "Initializing SimpleCSRVector " << this; });
    setBlockSize(block_size) ;
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
      m_values.resize(m_local_size*m_own_block_size);
      m_values.fill(ValueT());
    }
  }

  const VectorDistribution& distribution() const override
  {
    if (this->m_multi_impl)
      return IVectorImpl::distribution();
    else
      return m_own_distribution;
  }

  Arccore::Integer scalarizedLocalSize() const override
  {
    if (this->m_multi_impl)
      return IVectorImpl::scalarizedLocalSize();
    else
      return m_own_distribution.localSize()*m_own_block_size;
  }

  Arccore::Integer scalarizedGlobalSize() const override
  {
    if (this->m_multi_impl)
      return IVectorImpl::scalarizedGlobalSize();
    else
      return m_own_distribution.globalSize()*m_own_block_size;
  }

  Arccore::Integer scalarizedOffset() const override
  {
    if (this->m_multi_impl)
      return IVectorImpl::scalarizedOffset();
    else
      return m_own_distribution.offset()*m_own_block_size;
  }

  const VBlockImpl& vblockImpl() const { return *m_vblock; }

  //@{ @name Interface à soi-même
  void update([[maybe_unused]] const SimpleCSRVector<ValueT>& v)
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
  Integer m_own_block_size = 1 ;
  mutable VBlockImpl* m_vblock = nullptr;
  VectorDistribution m_own_distribution;
};

/*---------------------------------------------------------------------------*/

} // namespace Alien

/*---------------------------------------------------------------------------*/
