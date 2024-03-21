// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2024 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------

#include <alien/core/backend/IVectorConverter.h>
#include <alien/core/backend/VectorConverterRegisterer.h>

#include <iostream>
#include <alien/kernels/petsc/data_structure/PETScVector.h>

#include <alien/kernels/petsc/PETScBackEnd.h>
#include <alien/kernels/simple_csr/SimpleCSRVector.h>
#include <alien/kernels/simple_csr/SimpleCSRBackEnd.h>
using namespace Alien;

/*---------------------------------------------------------------------------*/

class SimpleCSR_to_PETSc_VectorConverter : public IVectorConverter
{
 public:
  SimpleCSR_to_PETSc_VectorConverter();
  virtual ~SimpleCSR_to_PETSc_VectorConverter() {}
 public:
  Alien::BackEndId sourceBackend() const
  {
    return AlgebraTraits<BackEnd::tag::simplecsr>::name();
  }
  Alien::BackEndId targetBackend() const
  {
    return AlgebraTraits<BackEnd::tag::petsc>::name();
  }
  void convert(const IVectorImpl* sourceImpl, IVectorImpl* targetImpl) const;
};

/*---------------------------------------------------------------------------*/

SimpleCSR_to_PETSc_VectorConverter::SimpleCSR_to_PETSc_VectorConverter()
{
  ;
}

/*---------------------------------------------------------------------------*/

void
SimpleCSR_to_PETSc_VectorConverter::convert(
    const IVectorImpl* sourceImpl, IVectorImpl* targetImpl) const
{
  const SimpleCSRVector<double>& v =
      cast<SimpleCSRVector<double>>(sourceImpl, sourceBackend());
  PETScVector& v2 = cast<PETScVector>(targetImpl, targetBackend());

  alien_debug([&] {
    cout() << "Converting SimpleCSRVector: " << &v << " to PETScVector " << &v2;
  });

  Arccore::ConstArrayView<Arccore::Real> values = v.values();
  if (sourceImpl->block())
  {
      const Arccore::Integer block_size = sourceImpl->block()->size();
      if (not v2.setBlockValues(block_size,values))
        throw Arccore::FatalErrorException(A_FUNCINFO, "Error while setting values");
  }
  else
  {
    if (not v2.setValues(values))
      throw Arccore::FatalErrorException(A_FUNCINFO, "Error while setting values");
  }
}

/*---------------------------------------------------------------------------*/

REGISTER_VECTOR_CONVERTER(SimpleCSR_to_PETSc_VectorConverter);
