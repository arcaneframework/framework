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
class Composyx_to_SimpleCSR_VectorConverter : public IVectorConverter
{
 public:
  Composyx_to_SimpleCSR_VectorConverter();
  virtual ~Composyx_to_SimpleCSR_VectorConverter() {}
 public:
  Alien::BackEndId sourceBackend() const { return AlgebraTraits<BackEnd::tag::composyx>::name(); }
  Alien::BackEndId targetBackend() const
  {
    return AlgebraTraits<BackEnd::tag::simplecsr>::name();
  }
  void convert(const IVectorImpl* sourceImpl, IVectorImpl* targetImpl) const;
};

/*---------------------------------------------------------------------------*/
Composyx_to_SimpleCSR_VectorConverter::Composyx_to_SimpleCSR_VectorConverter()
{
  ;
}

/*---------------------------------------------------------------------------*/
void
Composyx_to_SimpleCSR_VectorConverter::convert(
    const IVectorImpl* sourceImpl, IVectorImpl* targetImpl) const
{
  const ComposyxVector<double>& v =
      cast<ComposyxVector<double>>(sourceImpl, sourceBackend());
  SimpleCSRVector<double>& v2 =
      cast<SimpleCSRVector<double>>(targetImpl, targetBackend());

  alien_debug([&] {
    cout() << "Converting ComposyxVector: " << &v << " to SimpleCSRVector " << &v2;
  });

  v.getValues(v2.values().size(), v2.getDataPtr());
}

/*---------------------------------------------------------------------------*/
REGISTER_VECTOR_CONVERTER(Composyx_to_SimpleCSR_VectorConverter);
