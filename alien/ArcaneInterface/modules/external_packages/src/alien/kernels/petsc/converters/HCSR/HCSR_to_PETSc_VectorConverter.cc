// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------

#include <iostream>
#include <vector>

#include <alien/core/backend/IVectorConverter.h>
#include <alien/core/backend/VectorConverterRegisterer.h>
#include <alien/kernels/petsc/data_structure/PETScVector.h>

#include <alien/kernels/petsc/PETScBackEnd.h>
#include <alien/kernels/petsc/data_structure/PETScVector.h>

#include <alien/kernels/sycl/SYCLBackEnd.h>

#include <alien/kernels/sycl/data/HCSRVector.h>

//#include "alien/kernels/sycl/data/HCSRVectorInternal.h"

using namespace Alien;

/*---------------------------------------------------------------------------*/

class HCSR_to_PETSc_VectorConverter : public IVectorConverter
{
 public:
  HCSR_to_PETSc_VectorConverter();
  virtual ~HCSR_to_PETSc_VectorConverter() {}

 public:
  Alien::BackEndId sourceBackend() const
  {
    return AlgebraTraits<BackEnd::tag::hcsr>::name();
  }

  Alien::BackEndId targetBackend() const
  {
    return AlgebraTraits<BackEnd::tag::petsc>::name();
  }

  void convert(const IVectorImpl* sourceImpl, IVectorImpl* targetImpl) const;
};

/*---------------------------------------------------------------------------*/

HCSR_to_PETSc_VectorConverter::HCSR_to_PETSc_VectorConverter()
{
  ;
}

/*---------------------------------------------------------------------------*/
void
HCSR_to_PETSc_VectorConverter::convert(
    const IVectorImpl* sourceImpl, IVectorImpl* targetImpl) const
{
  const HCSRVector<double>& v =
  cast<HCSRVector<double>>(sourceImpl, sourceBackend());
  auto& v2 = cast<PETScVector>(targetImpl, targetBackend());

  alien_debug([&] {
    cout() << "Converting HCSRVector: " << &v << " to PETScVector " << &v2;
  });

  Alien::PETScVector::ValueType* values_d = v2.getDataPtr()  ;
  v.copyValuesTo(values_d) ;
  if(not v2.restoreDataPtr(values_d))
  {
    throw Arccore::FatalErrorException(A_FUNCINFO, "Error while Converting HCSRVector to PETScVector");
  }
}
/*---------------------------------------------------------------------------*/

REGISTER_VECTOR_CONVERTER(HCSR_to_PETSc_VectorConverter);
