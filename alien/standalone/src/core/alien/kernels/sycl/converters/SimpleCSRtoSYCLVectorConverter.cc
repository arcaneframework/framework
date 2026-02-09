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

#include <alien/kernels/sycl/data/SYCLVector.h>

#include <alien/kernels/sycl/SYCLBackEnd.h>
#include <alien/kernels/simple_csr/SimpleCSRVector.h>
#include <alien/kernels/simple_csr/SimpleCSRBackEnd.h>
using namespace Alien;

/*---------------------------------------------------------------------------*/

class SimpleCSRtoSYCLVectorConverter : public IVectorConverter
{
 public:
  SimpleCSRtoSYCLVectorConverter();
  virtual ~SimpleCSRtoSYCLVectorConverter() {}

 public:
  Alien::BackEndId sourceBackend() const
  {
    return AlgebraTraits<BackEnd::tag::simplecsr>::name();
  }
  Alien::BackEndId targetBackend() const
  {
    return AlgebraTraits<BackEnd::tag::sycl>::name();
  }
  void convert(const IVectorImpl* sourceImpl, IVectorImpl* targetImpl) const;
};

/*---------------------------------------------------------------------------*/

SimpleCSRtoSYCLVectorConverter::SimpleCSRtoSYCLVectorConverter()
{
  ;
}

/*---------------------------------------------------------------------------*/

void SimpleCSRtoSYCLVectorConverter::convert(
const IVectorImpl* sourceImpl, IVectorImpl* targetImpl) const
{
  const SimpleCSRVector<double>& v =
  cast<SimpleCSRVector<double>>(sourceImpl, sourceBackend());
  SYCLVector<double>& v2 =
  cast<SYCLVector<double>>(targetImpl, targetBackend());

  alien_debug(
  [&] { cout() << "Converting SimpleCSRVector: " << &v << " to SYCLVector " << &v2; });

  ConstArrayView<Real> values = v.values();
  v2.setValues(values.size(), values.data());
}

/*---------------------------------------------------------------------------*/

REGISTER_VECTOR_CONVERTER(SimpleCSRtoSYCLVectorConverter);
