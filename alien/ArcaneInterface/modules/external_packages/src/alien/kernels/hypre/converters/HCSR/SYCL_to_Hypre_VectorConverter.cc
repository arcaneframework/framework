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

#include <alien/handlers/accelerator/HCSRViewT.h>

#include <alien/kernels/hypre/data_structure/HypreVector.h>

#include <alien/kernels/hypre/HypreBackEnd.h>
#include <alien/kernels/hypre/data_structure/HypreVector.h>

#include "SYCL_to_Hypre_VectorConverter.h"

#include <alien/kernels/sycl/SYCLBackEnd.h>

#include <alien/kernels/sycl/data/SYCLVector.h>

#include "SYCL_to_Hypre_VectorConverter.h"

using namespace Alien;
/*---------------------------------------------------------------------------*/

SYCL_to_Hypre_VectorConverter::SYCL_to_Hypre_VectorConverter()
{
  ;
}

/*---------------------------------------------------------------------------*/
void
SYCL_to_Hypre_VectorConverter::convert(
    const IVectorImpl* sourceImpl, IVectorImpl* targetImpl) const
{
  const SYCLVector<double>& source =
  cast<SYCLVector<double>>(sourceImpl, sourceBackend());
  auto& target = cast<HypreVector>(targetImpl, targetBackend());

  alien_debug([&] {
    cout() << "Converting SYCLVector: " << &source << " to HypreVector " << &target;
  });


  if(target.getMemoryType()==Alien::BackEnd::Memory::Host)
  {
    UniqueArray<Arccore::Real> values(source.getAllocSize()) ;
    source.copyValuesTo(values.size(),values.data());
    target.setValues(values.size(), values.unguardedBasePointer());
    target.assemble() ;
  }
  else
  {
#ifdef ALIEN_USE_SYCL
      std::size_t alloc_size = source.getAllocSize();
      auto view = HVectorViewT<HypreVector>{&target,BackEnd::Memory::Device,alloc_size} ;
      source.copyValuesToDevice(view.m_values) ;
      target.setValues(alloc_size, view.m_values);
      target.assemble() ;
#endif
  }
}

void
SYCL_to_Hypre_VectorConverter::convert(const SYCLVector<double>& source,
                                       HypreVector& target) const
{

  alien_debug([&] {
    cout() << "Converting SYCLVector: " << &source << " to HypreVector " << &target;
  });


  if(target.getMemoryType()==Alien::BackEnd::Memory::Host)
  {
    UniqueArray<Arccore::Real> values(source.getAllocSize()) ;
    source.copyValuesTo(values.size(),values.data());
    target.setValues(values.size(), values.unguardedBasePointer());
    target.assemble() ;
  }
  else
  {
#ifdef ALIEN_USE_SYCL
    std::size_t alloc_size = source.getAllocSize();
    auto view = HVectorViewT<HypreVector>{&target,BackEnd::Memory::Device,alloc_size} ;
    source.copyValuesToDevice(view.m_values) ;
    target.setValues(alloc_size, view.m_values);
    target.assemble() ;
#endif
  }
}

/*---------------------------------------------------------------------------*/

REGISTER_VECTOR_CONVERTER(SYCL_to_Hypre_VectorConverter);
