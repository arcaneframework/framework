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

#include "SimpleCSR_to_Hypre_VectorConverter.h"

using namespace Alien;
using namespace Alien::SimpleCSRInternal;

/*---------------------------------------------------------------------------*/


/*---------------------------------------------------------------------------*/

SimpleCSR_to_Hypre_VectorConverter::SimpleCSR_to_Hypre_VectorConverter()
{
  ;
}

/*---------------------------------------------------------------------------*/

void
SimpleCSR_to_Hypre_VectorConverter::convert(
    const IVectorImpl* sourceImpl, IVectorImpl* targetImpl) const
{
  const SimpleCSRVector<Arccore::Real>& v =
      cast<SimpleCSRVector<Arccore::Real>>(sourceImpl, sourceBackend());
  auto& v2 = cast<HypreVector>(targetImpl, targetBackend());

  alien_debug([&] {
    cout() << "Converting SimpleCSRVector: " << &v << " to HypreVector " << &v2;
  });
  Arccore::ConstArrayView<Arccore::Real> values = v.values();
  if(v2.getMemoryType()==Alien::BackEnd::Memory::Host)
    v2.setValues(values.size(), values.unguardedBasePointer());
  else
  {
#ifdef ALIEN_USE_SYCL
      Alien::HypreVector::IndexType* rows_d = nullptr;
      Alien::HypreVector::ValueType* values_d = nullptr ;
      Alien::SYCLVector<Arccore::Real>::initDevicePointers(values.size(),
                                                           values.unguardedBasePointer(),&
                                                           rows_d,
                                                           &values_d) ;
      v2.setValues(v.getAllocSize(), rows_d, values_d);
      v2.assemble() ;
      Alien::SYCLVector<Arccore::Real>::freeDevicePointers(rows_d, values_d) ;
#else
      alien_fatal([&] {
        cout()<<"Error SYCL Support is required to Buid Hypre Vector on Device Memory";
      });
#endif
  }
}

void
SimpleCSR_to_Hypre_VectorConverter::convert(
    const SimpleCSRVector<Arccore::Real>& source, HypreVector& target) const
{
  alien_debug([&] {
    cout() << "Converting SimpleCSRVector: " << &source << " to HypreVector " << &target;
  });
  Arccore::ConstArrayView<Arccore::Real> values = source.values();
  if(target.getMemoryType()==Alien::BackEnd::Memory::Host)
    target.setValues(values.size(), values.unguardedBasePointer());
  else
  {
#ifdef ALIEN_USE_SYCL
      Alien::HypreVector::IndexType* rows_d = nullptr;
      Alien::HypreVector::ValueType* values_d = nullptr ;
      Alien::SYCLVector<Arccore::Real>::initDevicePointers(values.size(),
                                                           values.unguardedBasePointer(),&
                                                           rows_d,
                                                           &values_d) ;
      target.setValues(source.getAllocSize(), rows_d, values_d);
      target.assemble() ;
      Alien::SYCLVector<Arccore::Real>::freeDevicePointers(rows_d, values_d) ;
#else
      alien_fatal([&] {
        cout()<<"Error SYCL Support is required to Buid Hypre Vector on Device Memory";
      });
#endif
  }
}
/*---------------------------------------------------------------------------*/

REGISTER_VECTOR_CONVERTER(SimpleCSR_to_Hypre_VectorConverter);
