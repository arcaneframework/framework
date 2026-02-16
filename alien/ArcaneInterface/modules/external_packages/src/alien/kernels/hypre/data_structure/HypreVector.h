// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
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

  BackEnd::Memory::eType getMemoryType() const ;

  BackEnd::Exec::eSpaceType getExecSpace() const ;

 public:
  void init(const VectorDistribution& dist, const bool need_allocate);
  void init(const VectorDistribution& dist, Integer block_size, const bool need_allocate);
  void init(std::tuple<const VectorDistribution*,Integer> const& resource, const bool need_allocate)
  {
    init(*std::get<0>(resource),std::get<1>(resource),need_allocate) ;
  }
  void allocate(const VectorDistribution& dist);

  void free() {}
  void clear() {}

 private:
  bool setValues(const int nrow, const double* values);

  bool setValues(const int nrow, const int* rows, const double* values);

 public:
  bool getValues(const int nrow, const int* rows, double* values) const;

  bool getValues(const int nrow, double* values) const;

  bool copyValuesToDevice(std::size_t nrow,
                          IndexType* rows_d,
                          ValueType* values_d) const ;

  bool copyValuesToHost(std::size_t nrow,
                        IndexType* rows_d,
                        ValueType* values_d) const ;

  bool setValuesToZeros() ;


  void allocateDevicePointers(std::size_t local_size, ValueType** values) const;

  void freeDevicePointers(ValueType* values) const;

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
