// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2024 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
#pragma once
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

/*---------------------------------------------------------------------------*/

class Hypre_to_SimpleCSR_VectorConverter
: public Alien::IVectorConverter
, public Alien::VectorConverterT<Alien::BackEnd::tag::hypre,Alien::BackEnd::tag::simplecsr>
{
 public:
  using ConcreteBaseType = Alien::VectorConverterT<Alien::BackEnd::tag::hypre,Alien::BackEnd::tag::simplecsr> ;

  Hypre_to_SimpleCSR_VectorConverter();
  virtual ~Hypre_to_SimpleCSR_VectorConverter() {}
 public:
  Alien::BackEndId sourceBackend() const
  {
    return Alien::AlgebraTraits<Alien::BackEnd::tag::hypre>::name();
  }
  Alien::BackEndId targetBackend() const
  {
    return Alien::AlgebraTraits<Alien::BackEnd::tag::simplecsr>::name();
  }
  void convert(const Alien::IVectorImpl* sourceImpl, Alien::IVectorImpl* targetImpl) const;

  void convert(const ConcreteBaseType::SourceVectorType& source, ConcreteBaseType::TargetVectorType& target) const;
};

/*---------------------------------------------------------------------------*/
