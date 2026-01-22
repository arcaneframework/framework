// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------

#pragma once

#include <alien/kernels/trilinos/TrilinosPrecomp.h>
#include <alien/core/impl/IVectorImpl.h>
#include <alien/data/ISpace.h>
#include <alien/distribution/VectorDistribution.h>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

BEGIN_TRILINOSINTERNAL_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
template <typename ValueT, typename TagT> class VectorInternal;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

END_TRILINOSINTERNAL_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Alien {

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
template <typename ValueT, typename TagT> class TrilinosVector : public IVectorImpl
{
 public:
  typedef TrilinosInternal::VectorInternal<ValueT, TagT> VectorInternal;

 public:
  TrilinosVector(const MultiVectorImpl* multi_impl);

  virtual ~TrilinosVector();

 public:
  void init(const VectorDistribution& dist, const bool need_allocate);
  void allocate();

  void free() {}
  void clear() {}

 public:
  void setValues(const int nrows, ValueT const* values);

  void getValues(const int nrows, ValueT* values) const;

  ValueT* getDataPtr() { return nullptr; }

  ValueT const* getDataPtr() const { return nullptr; }

  ValueT norm1() const;

  ValueT norm2() const;

  ValueT dot(TrilinosVector const& y) const;

 public:
  VectorInternal* internal() { return m_internal.get(); }

  const VectorInternal* internal() const { return m_internal.get(); }

  void dump() const;
  void dump(std::string const& filename) const;

 private:
  bool assemble();

 private:
  std::unique_ptr<VectorInternal> m_internal;
  Integer m_local_offset = 0;
  Integer m_local_size   = 0;
  Integer m_global_size  = 0;
  Integer m_block_size   = 1 ;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Alien

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

