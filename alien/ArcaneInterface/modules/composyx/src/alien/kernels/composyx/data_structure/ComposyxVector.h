// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2024 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------

#pragma once

#include <alien/kernels/composyx/ComposyxPrecomp.h>
#include <alien/core/impl/IVectorImpl.h>
#include <alien/data/ISpace.h>
#include <alien/distribution/VectorDistribution.h>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

BEGIN_COMPOSYXINTERNAL_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
template <typename ValueT> class VectorInternal;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

END_COMPOSYXINTERNAL_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Alien {

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
template <typename ValueT> class ComposyxVector : public IVectorImpl
{
 public:
  typedef ComposyxInternal::VectorInternal<ValueT> VectorInternal;

 public:
  typedef SimpleCSRVector<ValueT> CSRVectorType;

  ComposyxVector(const MultiVectorImpl* multi_impl);

  virtual ~ComposyxVector();

 public:
  void init(const VectorDistribution& dist, const bool need_allocate);
  void allocate();

  void free() {}
  void clear() {}

 public:
  bool compute(IMessagePassingMng* parallel_mng, const CSRVectorType& B) ;

  void getValues(const int nrows, ValueT* values) const;

 public:
  VectorInternal* internal() { return m_internal.get(); }

  const VectorInternal* internal() const { return m_internal.get(); }

 private:
  std::unique_ptr<VectorInternal> m_internal;
  int m_local_offset;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
