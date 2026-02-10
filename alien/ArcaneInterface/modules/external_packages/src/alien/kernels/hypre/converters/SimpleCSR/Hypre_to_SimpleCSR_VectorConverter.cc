// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
#include <alien/core/backend/IVectorConverter.h>
#include <alien/core/backend/VectorConverterRegisterer.h>

#include <iostream>
#include <alien/kernels/hypre/data_structure/HypreVector.h>

#include <alien/kernels/hypre/HypreBackEnd.h>
#include <alien/kernels/simple_csr/CSRStructInfo.h>
#include <alien/kernels/simple_csr/SimpleCSRVector.h>
#include <alien/kernels/simple_csr/SimpleCSRBackEnd.h>

#include <alien/kernels/sycl/SYCLBackEnd.h>
#include <alien/kernels/sycl/data/SYCLVector.h>

#include "Hypre_to_SimpleCSR_VectorConverter.h"

using namespace Alien;
using namespace Alien::SimpleCSRInternal;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Hypre_to_SimpleCSR_VectorConverter::Hypre_to_SimpleCSR_VectorConverter()
{
  ;
}

/*---------------------------------------------------------------------------*/

void
Hypre_to_SimpleCSR_VectorConverter::convert(
    const IVectorImpl* sourceImpl, IVectorImpl* targetImpl) const
{
  const auto& source = cast<HypreVector>(sourceImpl, sourceBackend());
  SimpleCSRVector<Arccore::Real>& target =
      cast<SimpleCSRVector<Arccore::Real>>(targetImpl, targetBackend());

  convert(source,target);
}

void
Hypre_to_SimpleCSR_VectorConverter::convert(const HypreVector& source,
                                            SimpleCSRVector<Arccore::Real>& target) const
{
  alien_debug([&] {
    cout() << "Converting HypreVector: " << &source << " to SimpleCSRVector " << &target;
  });

  if(source.getMemoryType()==Alien::BackEnd::Memory::Device)
  {
    Arccore::ArrayView<Arccore::Real> values = target.values();
    source.copyValuesToHost(values.size(), nullptr, values.data());
  }
  else
  {
    Arccore::ArrayView<Arccore::Real> values = target.values();
    source.getValues(values.size(), values.data());
  }

}
/*---------------------------------------------------------------------------*/

REGISTER_VECTOR_CONVERTER(Hypre_to_SimpleCSR_VectorConverter);
