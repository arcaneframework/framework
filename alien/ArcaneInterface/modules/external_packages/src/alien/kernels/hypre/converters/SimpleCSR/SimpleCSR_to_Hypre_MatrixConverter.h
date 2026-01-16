// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
#pragma once
#include <iostream>

#include <alien/utils/Precomp.h>
#include <alien/core/backend/IMatrixConverter.h>
#include <alien/core/backend/MatrixConverterRegisterer.h>

#include <alien/kernels/hypre/data_structure/HypreMatrix.h>

#include <alien/kernels/hypre/HypreBackEnd.h>
#include <alien/kernels/simple_csr/CSRStructInfo.h>
#include <alien/kernels/simple_csr/SimpleCSRMatrix.h>
#include <alien/kernels/simple_csr/SimpleCSRBackEnd.h>
#include <alien/distribution/MatrixDistribution.h>

#include <arccore/collections/Array2.h>

/*---------------------------------------------------------------------------*/

class SimpleCSR_to_Hypre_MatrixConverter
: public Alien::IMatrixConverter
, public Alien::MatrixConverterT<Alien::BackEnd::tag::simplecsr,Alien::BackEnd::tag::hypre>
{
 public:
  using ConcreteBaseType = Alien::MatrixConverterT<Alien::BackEnd::tag::simplecsr,Alien::BackEnd::tag::hypre> ;

  SimpleCSR_to_Hypre_MatrixConverter();
  virtual ~SimpleCSR_to_Hypre_MatrixConverter() {}
 public:
  BackEndId sourceBackend() const
  {
    return Alien::AlgebraTraits<Alien::BackEnd::tag::simplecsr>::name();
  }

  BackEndId targetBackend() const
  {
    return Alien::AlgebraTraits<Alien::BackEnd::tag::hypre>::name();
  }
  void convert(const Alien::IMatrixImpl* sourceImpl, Alien::IMatrixImpl* targetImpl) const;
  void convert(const ConcreteBaseType::SourceMatrixType& source, ConcreteBaseType::TargetMatrixType& target) const;
  void _build(const Alien::SimpleCSRMatrix<Arccore::Real>& sourceImpl,
             Alien::HypreMatrix& targetImpl) const;
  void _buildBlock(const Alien::SimpleCSRMatrix<Arccore::Real>& sourceImpl,
                   Alien::HypreMatrix& targetImpl) const;
};

