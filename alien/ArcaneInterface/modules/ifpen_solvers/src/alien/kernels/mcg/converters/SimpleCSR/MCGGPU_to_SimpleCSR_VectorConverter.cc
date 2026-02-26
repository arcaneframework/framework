// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2024 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------

#include "alien/core/backend/IVectorConverter.h"
#include "alien/core/backend/VectorConverterRegisterer.h"
#include "alien/kernels/simple_csr/SimpleCSRVector.h"
#include "alien/kernels/simple_csr/SimpleCSRBackEnd.h"

#include "alien/kernels/mcg/data_structure/MCGVector.h"
#include "alien/kernels/mcg/MCGBackEnd.h"

using namespace Alien;

class MCGGPU_to_SimpleCSR_VectorConverter : public IVectorConverter
{
 public:
  MCGGPU_to_SimpleCSR_VectorConverter() = default;
  ~MCGGPU_to_SimpleCSR_VectorConverter() override = default;

  BackEndId sourceBackend() const override
  {
    return AlgebraTraits<BackEnd::tag::mcgsolver_gpu>::name();
  }
  BackEndId targetBackend() const override
  {
    return AlgebraTraits<BackEnd::tag::simplecsr>::name();
  }
  void convert(const IVectorImpl* sourceImpl, IVectorImpl* targetImpl) const override;
};

void
MCGGPU_to_SimpleCSR_VectorConverter::convert(
    const IVectorImpl* sourceImpl, IVectorImpl* targetImpl) const
{
  const auto& v =
    cast<MCGVector<Real,MCGInternal::eMemoryDomain::Device>>(sourceImpl, sourceBackend());
  auto& v2 =
      cast<SimpleCSRVector<double>>(targetImpl, targetBackend());

  alien_debug([this,&v,&v2] {
    cout() << "Converting MCGVector on Device: " << &v << " to SimpleCSRVector " << &v2;
  });

  v.getValues(v2.values().data());
}

REGISTER_VECTOR_CONVERTER(MCGGPU_to_SimpleCSR_VectorConverter);
