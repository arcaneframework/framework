// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
#pragma once
#include <iostream>
#include <vector>

#include <alien/core/backend/IVectorConverter.h>
#include <alien/core/backend/VectorConverterRegisterer.h>
#include <alien/kernels/hypre/data_structure/HypreVector.h>

#include <alien/kernels/hypre/HypreBackEnd.h>
#include <alien/kernels/hypre/data_structure/HypreVector.h>

#include <alien/kernels/sycl/SYCLBackEnd.h>

#include <alien/kernels/sycl/data/SYCLVector.h>


using namespace Alien;

/*---------------------------------------------------------------------------*/
class SYCL_to_Hypre_VectorConverter
: public IVectorConverter
, public Alien::VectorConverterT<Alien::BackEnd::tag::sycl,Alien::BackEnd::tag::hypre>
{
 public:
  SYCL_to_Hypre_VectorConverter();
  virtual ~SYCL_to_Hypre_VectorConverter() {}

 public:
  using ConcreteBaseType = Alien::VectorConverterT<Alien::BackEnd::tag::sycl,
                                                   Alien::BackEnd::tag::hypre> ;

  Alien::BackEndId sourceBackend() const
  {
    return AlgebraTraits<BackEnd::tag::sycl>::name();
  }

  Alien::BackEndId targetBackend() const
  {
    return AlgebraTraits<BackEnd::tag::hypre>::name();
  }

  void convert(const IVectorImpl* sourceImpl, IVectorImpl* targetImpl) const;

  void convert(const ConcreteBaseType::SourceVectorType& source, ConcreteBaseType::TargetVectorType& target) const;

};

/*---------------------------------------------------------------------------*/
