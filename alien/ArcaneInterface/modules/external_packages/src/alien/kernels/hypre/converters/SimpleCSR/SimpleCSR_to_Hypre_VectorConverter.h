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

using namespace Alien;
using namespace Alien::SimpleCSRInternal;

/*---------------------------------------------------------------------------*/

class SimpleCSR_to_Hypre_VectorConverter
: public Alien::IVectorConverter
, public Alien::VectorConverterT<Alien::BackEnd::tag::simplecsr,Alien::BackEnd::tag::hypre>
{
 public:
  using ConcreteBaseType = Alien::VectorConverterT<Alien::BackEnd::tag::simplecsr,Alien::BackEnd::tag::hypre> ;

  SimpleCSR_to_Hypre_VectorConverter();
  virtual ~SimpleCSR_to_Hypre_VectorConverter() {}
 public:
  Alien::BackEndId sourceBackend() const
  {
    return AlgebraTraits<Alien::BackEnd::tag::simplecsr>::name();
  }
  Alien::BackEndId targetBackend() const
  {
    return AlgebraTraits<Alien::BackEnd::tag::hypre>::name();
  }
  void convert(const Alien::IVectorImpl* sourceImpl, Alien::IVectorImpl* targetImpl) const;

  void convert(const ConcreteBaseType::SourceVectorType& source, ConcreteBaseType::TargetVectorType& target) const;
};

