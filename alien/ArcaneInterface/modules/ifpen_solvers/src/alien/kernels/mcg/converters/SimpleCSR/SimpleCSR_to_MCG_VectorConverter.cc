// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2024 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------

#include <iostream>

#include "alien/core/backend/IVectorConverter.h"
#include "alien/core/backend/VectorConverterRegisterer.h"
#include "alien/kernels/simple_csr/SimpleCSRVector.h"
#include "alien/kernels/simple_csr/SimpleCSRBackEnd.h"

#include "alien/kernels/mcg/data_structure/MCGVector.h"
#include "alien/kernels/mcg/MCGBackEnd.h"

using namespace Alien;

class SimpleCSR_to_MCG_VectorConverter : public IVectorConverter
{
 public:
  SimpleCSR_to_MCG_VectorConverter();
  virtual ~SimpleCSR_to_MCG_VectorConverter() {}

 public:
  BackEndId sourceBackend() const
  {
    return AlgebraTraits<BackEnd::tag::simplecsr>::name();
  }
  BackEndId targetBackend() const
  {
    return AlgebraTraits<BackEnd::tag::mcgsolver>::name();
  }
  void convert(const IVectorImpl* sourceImpl, IVectorImpl* targetImpl) const;
  // void convert(const IVectorImpl * sourceImpl, IVectorImpl * targetImpl,int i) const;
};

SimpleCSR_to_MCG_VectorConverter::SimpleCSR_to_MCG_VectorConverter()
{
}

void
SimpleCSR_to_MCG_VectorConverter::convert(
    const IVectorImpl* sourceImpl, IVectorImpl* targetImpl) const
{
  const SimpleCSRVector<double>& v =
      cast<SimpleCSRVector<double>>(sourceImpl, sourceBackend());
  auto& v2 =
    cast<MCGVector<Real,MCGInternal::eMemoryDomain::Host>>(targetImpl, targetBackend());

  alien_debug([&] {
    cout() << "Converting SimpleCSRVector: " << &v << " to MCGVector " << &v2;
  });

  v2.setValues(v.values().data());
}

REGISTER_VECTOR_CONVERTER(SimpleCSR_to_MCG_VectorConverter);
