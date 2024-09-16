﻿// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2024 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
#pragma once

#include <alien/AlienExternalPackagesPrecomp.h>
#include <alien/core/impl/IVectorImpl.h>
#include <alien/distribution/VectorDistribution.h>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class SimpleCSR_to_Hypre_VectorConverter;
class HCSR_to_Hypre_VectorConverter;
class SYCL_to_Hypre_VectorConverter;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Alien {

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class Block;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Internal {

  /*---------------------------------------------------------------------------*/
  /*---------------------------------------------------------------------------*/

  class VectorInternal;

  /*---------------------------------------------------------------------------*/
  /*---------------------------------------------------------------------------*/

} // namespace Internal

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class ALIEN_EXTERNAL_PACKAGES_EXPORT HypreVector : public IVectorImpl
{
 public:
  friend class ::SimpleCSR_to_Hypre_VectorConverter;
  friend class ::HCSR_to_Hypre_VectorConverter;
  friend class ::SYCL_to_Hypre_VectorConverter;

  typedef Internal::VectorInternal VectorInternal;

  typedef Integer IndexType ;
  typedef double  ValueType ;

 public:
  HypreVector(const MultiVectorImpl* multi_impl);

  virtual ~HypreVector();

 public:
  void init(const VectorDistribution& dist, const bool need_allocate);
  void allocate();

  void free() {}
  void clear() {}

 private:
  bool setValues(const int nrow, const double* values);

  bool setValues(const int nrow, const int* rows, const double* values);

 public:
  bool getValues(const int nrow, const int* rows, double* values) const;

  bool getValues(const int nrow, double* values) const;

 public:
  // Méthodes restreintes à usage interne de l'implémentation HYPRE
  VectorInternal* internal() { return m_internal; }
  const VectorInternal* internal() const { return m_internal; }

  // These functions should be removed when the relevant Converters will be implemented
  void update(const HypreVector& v);

 private:
  bool assemble();

 private:
  VectorInternal* m_internal;
  Arccore::Integer m_block_size;
  Arccore::Integer m_offset;
  Arccore::UniqueArray<Arccore::Integer> m_rows;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Alien

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
