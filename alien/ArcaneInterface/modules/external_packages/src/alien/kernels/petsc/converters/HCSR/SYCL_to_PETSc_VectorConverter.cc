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

#include <alien/kernels/sycl/data/SYCLVector.h>


using namespace Alien;

/*---------------------------------------------------------------------------*/
class SYCL_to_PETSc_VectorConverter : public IVectorConverter
{
 public:
  SYCL_to_PETSc_VectorConverter();
  virtual ~SYCL_to_PETSc_VectorConverter() {}

 public:
  Alien::BackEndId sourceBackend() const
  {
    return AlgebraTraits<BackEnd::tag::sycl>::name();
  }

  Alien::BackEndId targetBackend() const
  {
    return AlgebraTraits<BackEnd::tag::petsc>::name();
  }

  void convert(const IVectorImpl* sourceImpl, IVectorImpl* targetImpl) const;
};

/*---------------------------------------------------------------------------*/

SYCL_to_PETSc_VectorConverter::SYCL_to_PETSc_VectorConverter()
{
  ;
}

/*---------------------------------------------------------------------------*/
void
SYCL_to_PETSc_VectorConverter::convert(
    const IVectorImpl* sourceImpl, IVectorImpl* targetImpl) const
{
  const SYCLVector<double>& v =
  cast<SYCLVector<double>>(sourceImpl, sourceBackend());
  auto& v2 = cast<PETScVector>(targetImpl, targetBackend());

  alien_debug([&] {
    cout() << "Converting SYCLVector: " << &v << " to PETScVector " << &v2;
  });

  Alien::PETScVector::ValueType* values_d = v2.getDataPtr()  ;
  v.copyValuesToDevice(values_d) ;
  if(not v2.restoreDataPtr(values_d))
  {
    throw Arccore::FatalErrorException(A_FUNCINFO, "Error while Converting HCSRVector to PETScVector");
  }
}
/*---------------------------------------------------------------------------*/

REGISTER_VECTOR_CONVERTER(SYCL_to_PETSc_VectorConverter);
