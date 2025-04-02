// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------

#include <iostream>
#include <vector>

#include <alien/core/backend/IVectorConverter.h>
#include <alien/core/backend/VectorConverterRegisterer.h>
#include <alien/kernels/hypre/data_structure/HypreVector.h>

#include <alien/kernels/hypre/HypreBackEnd.h>
#include <alien/kernels/hypre/data_structure/HypreVector.h>

#include <alien/kernels/sycl/SYCLBackEnd.h>

#include <alien/kernels/sycl/data/HCSRVector.h>

//#include "alien/kernels/sycl/data/HCSRVectorInternal.h"

using namespace Alien;

/*---------------------------------------------------------------------------*/

class HCSR_to_Hypre_VectorConverter : public IVectorConverter
{
 public:
  HCSR_to_Hypre_VectorConverter();
  virtual ~HCSR_to_Hypre_VectorConverter() {}

 public:
  Alien::BackEndId sourceBackend() const
  {
    return AlgebraTraits<BackEnd::tag::hcsr>::name();
  }

  Alien::BackEndId targetBackend() const
  {
    return AlgebraTraits<BackEnd::tag::hypre>::name();
  }

  void convert(const IVectorImpl* sourceImpl, IVectorImpl* targetImpl) const;
};

/*---------------------------------------------------------------------------*/

HCSR_to_Hypre_VectorConverter::HCSR_to_Hypre_VectorConverter()
{
  ;
}

/*---------------------------------------------------------------------------*/
void
HCSR_to_Hypre_VectorConverter::convert(
    const IVectorImpl* sourceImpl, IVectorImpl* targetImpl) const
{
  const HCSRVector<double>& v =
  cast<HCSRVector<double>>(sourceImpl, sourceBackend());
  auto& v2 = cast<HypreVector>(targetImpl, targetBackend());

  alien_debug([&] {
    cout() << "Converting HCSRVector: " << &v << " to HypreVector " << &v2;
  });

  Alien::HypreVector::IndexType* rows_d = nullptr;
  Alien::HypreVector::ValueType* values_d = nullptr ;
  v.initDevicePointers(&rows_d, &values_d) ;
  v2.setValues(v.getAllocSize(), rows_d, values_d);
  v2.assemble() ;
  v.freeDevicePointers(rows_d, values_d) ;
}
/*---------------------------------------------------------------------------*/

REGISTER_VECTOR_CONVERTER(HCSR_to_Hypre_VectorConverter);
