// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2024 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------


#include <iostream>

#include <alien/core/backend/IVectorConverter.h>
#include <alien/core/backend/VectorConverterRegisterer.h>

#include <alien/kernels/sycl/data/SYCLVector.h>

#include <alien/kernels/sycl/SYCLBackEnd.h>
#include <alien/kernels/simple_csr/SimpleCSRVector.h>
#include <alien/kernels/simple_csr/SimpleCSRBackEnd.h>
using namespace Alien;

/*---------------------------------------------------------------------------*/

class SYCLtoSimpleCSRVectorConverter : public IVectorConverter
{
 public:
  SYCLtoSimpleCSRVectorConverter();
  virtual ~SYCLtoSimpleCSRVectorConverter() {}

 public:
  Alien::BackEndId sourceBackend() const
  {
    return AlgebraTraits<BackEnd::tag::sycl>::name();
  }
  Alien::BackEndId targetBackend() const
  {
    return AlgebraTraits<BackEnd::tag::simplecsr>::name();
  }
  void convert(const IVectorImpl* sourceImpl, IVectorImpl* targetImpl) const;
};

/*---------------------------------------------------------------------------*/

SYCLtoSimpleCSRVectorConverter::SYCLtoSimpleCSRVectorConverter()
{
  ;
}

/*---------------------------------------------------------------------------*/

void SYCLtoSimpleCSRVectorConverter::convert(
const IVectorImpl* sourceImpl, IVectorImpl* targetImpl) const
{
  const SYCLVector<double>& sycl_v =
  cast<SYCLVector<double>>(sourceImpl, sourceBackend());
  SimpleCSRVector<double>& csr_v =
  cast<SimpleCSRVector<double>>(targetImpl, targetBackend());

  alien_debug([&] { cout() << "Converting SYCLVector: " << &sycl_v << " to SimpleCSRVector " << &csr_v; });
  sycl_v.copyValuesTo(csr_v.values().size(), csr_v.getDataPtr());
}

/*---------------------------------------------------------------------------*/

REGISTER_VECTOR_CONVERTER(SYCLtoSimpleCSRVectorConverter);
