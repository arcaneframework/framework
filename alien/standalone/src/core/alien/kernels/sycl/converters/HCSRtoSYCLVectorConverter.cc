// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2024 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------

#include <iostream>
#include <vector>

#include <alien/core/backend/IVectorConverter.h>
#include <alien/core/backend/VectorConverterRegisterer.h>

#include <alien/kernels/sycl/SYCLBackEnd.h>

#include <alien/kernels/sycl/data/SYCLVector.h>
#include <alien/kernels/sycl/data/HCSRVector.h>

#include "alien/kernels/sycl/data/SYCLVectorInternal.h"
#include "alien/kernels/sycl/data/HCSRVectorInternal.h"

using namespace Alien;

/*---------------------------------------------------------------------------*/

class HCSRtoSYCLVectorConverter : public IVectorConverter
{
 public:
  HCSRtoSYCLVectorConverter();
  virtual ~HCSRtoSYCLVectorConverter() {}

 public:
  Alien::BackEndId sourceBackend() const
  {
    return AlgebraTraits<BackEnd::tag::hcsr>::name();
  }
  Alien::BackEndId targetBackend() const
  {
    return AlgebraTraits<BackEnd::tag::sycl>::name();
  }
  void convert(const IVectorImpl* sourceImpl, IVectorImpl* targetImpl) const;
};

/*---------------------------------------------------------------------------*/

HCSRtoSYCLVectorConverter::HCSRtoSYCLVectorConverter()
{
  ;
}

/*---------------------------------------------------------------------------*/

void HCSRtoSYCLVectorConverter::convert(
const IVectorImpl* sourceImpl, IVectorImpl* targetImpl) const
{
  const HCSRVector<double>& v =
  cast<HCSRVector<double>>(sourceImpl, sourceBackend());
  SYCLVector<double>& v2 =
  cast<SYCLVector<double>>(targetImpl, targetBackend());

  alien_debug(
  [&] { cout() << "Converting HCSRVector: " << &v << " to SYCLVector " << &v2; });

  v2.internal()->copy(v.internal()->values());
}

/*---------------------------------------------------------------------------*/

REGISTER_VECTOR_CONVERTER(HCSRtoSYCLVectorConverter);
