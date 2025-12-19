// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2024 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------

#include <iostream>

#include <alien/core/backend/IVectorConverter.h>
#include <alien/core/backend/VectorConverterRegisterer.h>
#include <alien/kernels/simple_csr/SimpleCSRVector.h>
#include <alien/kernels/simple_csr/SimpleCSRBackEnd.h>

#include "alien/kernels/mcg/data_structure/MCGVector.h"
#include "alien/kernels/mcg/MCGBackEnd.h"

using namespace Alien;

class MCG_to_SimpleCSR_VectorConverter : public IVectorConverter
{
 public:
  MCG_to_SimpleCSR_VectorConverter();
  virtual ~MCG_to_SimpleCSR_VectorConverter() {}

 public:
  BackEndId sourceBackend() const
  {
    return AlgebraTraits<BackEnd::tag::mcgsolver>::name();
  }
  BackEndId targetBackend() const
  {
    return AlgebraTraits<BackEnd::tag::simplecsr>::name();
  }
  void convert(const IVectorImpl* sourceImpl, IVectorImpl* targetImpl) const;
};

MCG_to_SimpleCSR_VectorConverter::MCG_to_SimpleCSR_VectorConverter()
{
  ;
}

void
MCG_to_SimpleCSR_VectorConverter::convert(
    const IVectorImpl* sourceImpl, IVectorImpl* targetImpl) const
{
  const auto& v = cast<MCGVector<Real,MCGInternal::eMemoryDomain::Host>>(sourceImpl, sourceBackend());
  SimpleCSRVector<double>& v2 =
      cast<SimpleCSRVector<double>>(targetImpl, targetBackend());

  alien_debug([&] {
    cout() << "Converting MCGVector: " << &v << " to SimpleCSRVector " << &v2;
  });

  v.getValues(v2.values().data());
}

REGISTER_VECTOR_CONVERTER(MCG_to_SimpleCSR_VectorConverter);
