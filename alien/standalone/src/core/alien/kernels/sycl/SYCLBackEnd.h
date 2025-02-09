// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
#pragma once

#include <arccore/message_passing/MessagePassingGlobal.h>
#include <alien/core/backend/BackEnd.h>
#include <alien/utils/Precomp.h>

#include <alien/core/backend/IInternalLinearAlgebraExprT.h>
#include <alien/core/backend/IInternalLinearAlgebraT.h>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Alien
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class MultiVectorImpl;

template <typename T>
class SYCLBEllPackMatrix;

template <typename T>
class SYCLVector;

namespace SYCLInternal
{

  template <typename MatrixT>
  class SYCLSendRecvOp ;

  template <typename MatrixT>
  class SYCLLUSendRecvOp ;
}

template <typename T>
struct SYCLBEllPackTraits
{
  // clang-format off
  typedef SYCLBEllPackMatrix<T>                              MatrixType;
  typedef SYCLVector<T>                                      VectorType;
  typedef IInternalLinearAlgebra<MatrixType, VectorType>     AlgebraType;
  typedef IInternalLinearAlgebraExpr<MatrixType, VectorType> AlgebraExprType;
  // clang-format on
};

extern SYCLBEllPackTraits<Real>::AlgebraType* SYCLInternalLinearAlgebraFactory();
extern SYCLBEllPackTraits<Real>::AlgebraExprType*
SYCLInternalLinearAlgebraExprFactory();


template <typename T>
class HCSRMatrix;

template <typename T>
class HCSRVector;


template <typename T>
struct HCSRTraits
{
  // clang-format off
  typedef HCSRMatrix<T>                              MatrixType;
  typedef HCSRVector<T>                              VectorType;
  // clang-format on
};

/*---------------------------------------------------------------------------*/

namespace BackEnd
{
  namespace tag
  {
    struct sycl
    {};

    struct hcsr
    {};
  } // namespace tag
} // namespace BackEnd

template <>
struct AlgebraTraits<BackEnd::tag::sycl>
{
  // clang-format off
  typedef Real                                      value_type;
  typedef SYCLBEllPackTraits<Real>::MatrixType      matrix_type;
  typedef SYCLBEllPackTraits<Real>::VectorType      vector_type;
  typedef SYCLBEllPackTraits<Real>::AlgebraType     algebra_type;
  typedef SYCLBEllPackTraits<Real>::AlgebraExprType algebra_expr_type;
  // clang-format on

  static algebra_type* algebra_factory(
  IMessagePassingMng* p_mng ALIEN_UNUSED_PARAM = nullptr)
  {
    return SYCLInternalLinearAlgebraFactory();
  }
  static algebra_expr_type* algebra_expr_factory(
  IMessagePassingMng* p_mng ALIEN_UNUSED_PARAM = nullptr)
  {
    return SYCLInternalLinearAlgebraExprFactory();
  }

  static BackEndId name() { return "sycl"; }
};


template <>
struct AlgebraTraits<BackEnd::tag::hcsr>
{
  // clang-format off
  typedef Real                              value_type;
  typedef HCSRTraits<Real>::MatrixType      matrix_type;
  typedef HCSRTraits<Real>::VectorType      vector_type;
  // clang-format on

  static BackEndId name() { return "hcsr"; }
};

template <>
struct LUSendRecvTraits<BackEnd::tag::sycl>
{
  // clang-format off
  typedef AlgebraTraits<BackEnd::tag::sycl>::matrix_type      matrix_type ;
  typedef AlgebraTraits<BackEnd::tag::sycl>::vector_type      vector_type ;
  typedef AlgebraTraits<BackEnd::tag::sycl>::value_type       value_type ;
  typedef SYCLInternal::SYCLLUSendRecvOp<matrix_type>         matrix_op_type ;
  typedef SYCLInternal::SYCLSendRecvOp<value_type>            vector_op_type ;
  // clang-format on
};

/*---------------------------------------------------------------------------*/

} // namespace Alien

/*---------------------------------------------------------------------------*/
