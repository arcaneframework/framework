// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2024 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
#include <iostream>

#include <alien/core/backend/IVectorConverter.h>
#include <alien/core/backend/VectorConverterRegisterer.h>

#include <alien/kernels/petsc/data_structure/PETScVector.h>

#include <alien/kernels/petsc/PETScBackEnd.h>
#include <alien/kernels/simple_csr/SimpleCSRVector.h>
#include <alien/kernels/simple_csr/SimpleCSRBackEnd.h>

using namespace Alien;

/*---------------------------------------------------------------------------*/

class PETSc_to_SimpleCSR_VectorConverter : public IVectorConverter
{
 public:
  PETSc_to_SimpleCSR_VectorConverter();
  virtual ~PETSc_to_SimpleCSR_VectorConverter() {}
 public:
  Alien::BackEndId sourceBackend() const
  {
    return AlgebraTraits<BackEnd::tag::petsc>::name();
  }
  Alien::BackEndId targetBackend() const
  {
    return AlgebraTraits<BackEnd::tag::simplecsr>::name();
  }
  void convert(const IVectorImpl* sourceImpl, IVectorImpl* targetImpl) const;
};

/*---------------------------------------------------------------------------*/

PETSc_to_SimpleCSR_VectorConverter::PETSc_to_SimpleCSR_VectorConverter()
{
  ;
}

/*---------------------------------------------------------------------------*/

void
PETSc_to_SimpleCSR_VectorConverter::convert(
    const IVectorImpl* sourceImpl, IVectorImpl* targetImpl) const
{
  const PETScVector& v = cast<PETScVector>(sourceImpl, sourceBackend());
  SimpleCSRVector<double>& v2 =
      cast<SimpleCSRVector<double>>(targetImpl, targetBackend());

  alien_debug([&] {
    cout() << "Converting PETScVector: " << &v << " to SimpleCSRVector " << &v2;
  });

  Arccore::ArrayView<Arccore::Real> values = v2.values();
  v.getValues(values.size(), values.unguardedBasePointer());
}

/*---------------------------------------------------------------------------*/

REGISTER_VECTOR_CONVERTER(PETSc_to_SimpleCSR_VectorConverter);
