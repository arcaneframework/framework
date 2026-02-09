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
#include <alien/kernels/hypre/data_structure/HypreVector.h>

#include <alien/handlers/accelerator/HCSRViewT.h>

#include <alien/kernels/hypre/HypreBackEnd.h>
#include <alien/kernels/hypre/data_structure/HypreVector.h>

#include <alien/kernels/sycl/SYCLBackEnd.h>

#include <alien/kernels/sycl/data/SYCLVector.h>

#include "Hypre_to_SYCL_VectorConverter.h"

using namespace Alien;
/*---------------------------------------------------------------------------*/


/*---------------------------------------------------------------------------*/

Hypre_to_SYCL_VectorConverter::Hypre_to_SYCL_VectorConverter()
{
  ;
}

/*---------------------------------------------------------------------------*/
void
Hypre_to_SYCL_VectorConverter::convert(
    const IVectorImpl* sourceImpl, IVectorImpl* targetImpl) const
{
  const HypreVector& source =
  cast<HypreVector>(sourceImpl, sourceBackend());
  auto& target = cast<SYCLVector<double>>(targetImpl, targetBackend());

  alien_debug([&] {
    cout() << "Converting HypreVector: " << &source << " to SYCLVector " << &target;
  });


  if(source.getMemoryType()==Alien::BackEnd::Memory::Device)
  {
#ifdef ALIEN_USE_SYCL
      Alien::HypreVector::IndexType* rows_d = nullptr;
      Alien::HypreVector::ValueType* values_d = nullptr ;
      std::size_t alloc_size = target.getAllocSize() ;
      Alien::SYCLVector<Arccore::Real>::allocateDevicePointers(alloc_size,
                                                               &rows_d,
                                                               &values_d) ;
      source.copyValuesToDevice(alloc_size, rows_d, values_d);
      target.setValuesFromDevice(alloc_size,values_d);
      Alien::SYCLVector<Arccore::Real>::freeDevicePointers(rows_d, values_d) ;
#endif
  }
  else
  {
    std::size_t alloc_size = target.getAllocSize() ;
    Arccore::UniqueArray<Arccore::Real> values(alloc_size);
    source.getValues(alloc_size, values.data());
    target.setValues(alloc_size, values.data());
  }
}

void
Hypre_to_SYCL_VectorConverter::convert(HypreVector const& source,
                                       SYCLVector<double>& target) const
{

  alien_debug([&] {
    cout() << "Converting HypreVector: " << &source << " to SYCLVector " << &target;
  });

  if(source.getMemoryType()==Alien::BackEnd::Memory::Device)
  {
#ifdef ALIEN_USE_SYCL

      std::size_t alloc_size = target.getAllocSize();
      auto view = HVectorViewT<HypreVector>{&source,BackEnd::Memory::Device,alloc_size} ;
      source.copyValuesToDevice(alloc_size,nullptr,view.m_values) ;
      target.setValuesFromDevice(alloc_size,view.m_values);
#endif
  }
  else
  {
    std::size_t alloc_size = target.getAllocSize() ;
    Arccore::UniqueArray<Arccore::Real> values(alloc_size);
    source.getValues(alloc_size, values.data());
    target.setValuesFromHost(alloc_size, values.data());
  }
}
/*---------------------------------------------------------------------------*/

REGISTER_VECTOR_CONVERTER(Hypre_to_SYCL_VectorConverter);
