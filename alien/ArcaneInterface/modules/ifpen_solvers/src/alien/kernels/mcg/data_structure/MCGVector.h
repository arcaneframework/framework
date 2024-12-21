// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2024 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0


#ifndef ALIEN_MCGIMPL_MCGVECTOR_H
#define ALIEN_MCGIMPL_MCGVECTOR_H

#include <alien/utils/Precomp.h>
#include <alien/core/impl/IVectorImpl.h>
#include <alien/data/ISpace.h>

#include "alien/kernels/mcg/MCGPrecomp.h"

BEGIN_MCGINTERNAL_NAMESPACE

class VectorInternal;

END_MCGINTERNAL_NAMESPACE

namespace Alien {

/*---------------------------------------------------------------------------*/

class MCGVector : public IVectorImpl
{
 public:
  typedef MCGInternal::VectorInternal VectorInternal;

 public:
  MCGVector(const MultiVectorImpl* multi_impl);

  virtual ~MCGVector();

 public:
  void init(const VectorDistribution& dist, const bool need_allocate);
  void allocate();

  void free() {}
  void clear() {}

 public:
  void setValues(double const* values);
  void getValues(double* values) const;

 public:
  VectorInternal* internal() { return m_internal; }
  const VectorInternal* internal() const { return m_internal; }

  void update(const MCGVector& v);

 private:
  VectorInternal* m_internal = nullptr;
};

}

#endif /* ALIEN_MCGIMPL_MCGVECTOR_H */
