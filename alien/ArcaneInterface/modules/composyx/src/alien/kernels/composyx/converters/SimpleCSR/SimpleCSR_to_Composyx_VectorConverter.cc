// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2024 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
#include <alien/core/backend/IVectorConverter.h>
#include <alien/core/backend/VectorConverterRegisterer.h>

#include <iostream>

#include <alien/kernels/simple_csr/SimpleCSRVector.h>
#include <alien/kernels/simple_csr/SimpleCSRBackEnd.h>

#include <alien/kernels/composyx/ComposyxBackEnd.h>
#include <alien/kernels/composyx/data_structure/ComposyxVector.h>
using namespace Alien;

/*---------------------------------------------------------------------------*/
class SimpleCSR_to_Composyx_VectorConverter : public IVectorConverter
{
 public:
  SimpleCSR_to_Composyx_VectorConverter();
  virtual ~SimpleCSR_to_Composyx_VectorConverter() {}
 public:
  Alien::BackEndId sourceBackend() const
  {
    return AlgebraTraits<BackEnd::tag::simplecsr>::name();
  }
  Alien::BackEndId targetBackend() const { return AlgebraTraits<BackEnd::tag::composyx>::name(); }
  void convert(const IVectorImpl* sourceImpl, IVectorImpl* targetImpl) const;
};

/*---------------------------------------------------------------------------*/
SimpleCSR_to_Composyx_VectorConverter::SimpleCSR_to_Composyx_VectorConverter()
{
  ;
}

/*---------------------------------------------------------------------------*/
void
SimpleCSR_to_Composyx_VectorConverter::convert(
    const IVectorImpl* sourceImpl, IVectorImpl* targetImpl) const
{
  const SimpleCSRVector<double>& v =
      cast<SimpleCSRVector<double>>(sourceImpl, sourceBackend());
  ComposyxVector<double>& v2 =
      cast<ComposyxVector<double>>(targetImpl, targetBackend());

  alien_info([&] {
    cout() << "Converting SimpleCSRVector: " << &v << " to ComposyxVector " << &v2;
  });
  auto const& dist = targetImpl->distribution();
  v2.compute(dist.parallelMng(),v) ;
}

/*---------------------------------------------------------------------------*/
REGISTER_VECTOR_CONVERTER(SimpleCSR_to_Composyx_VectorConverter);
