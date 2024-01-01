// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2024 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* DataOperation.h                                             (C) 2000-2024 */
/*                                                                           */
/* Opération sur une donnée.                                                 */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_IMPL_DATAOPERATION_H
#define ARCANE_IMPL_DATAOPERATION_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/base/ArrayView.h"
#include "arccore/base/Span.h"

#include "arcane/utils/Real2.h"
#include "arcane/utils/Real3.h"
#include "arcane/utils/Real2x2.h"
#include "arcane/utils/Real3x3.h"
#include "arcane/utils/Float16.h"
#include "arcane/utils/BFloat16.h"

#include "arcane/datatype/IDataOperation.h"

#include "arcane/Parallel.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<typename DataOperator>
class DataOperationT
: public IDataOperation
{
 public:

  DataOperationT() {}
  DataOperationT(const DataOperator& op) : m_operator(op) {}

 private:

  template<typename DataType> void
  _applyView(ArrayView<DataType> output,ConstArrayView<DataType> input)
  {
    for( Int32 i=0, n=input.size(); i<n; ++i )
      output[i] = m_operator(output[i],input[i]);
  }

 public:

  void apply(ByteArrayView output,ByteConstArrayView input) override
  {
    _applyView(output,input);
  }
  void apply(RealArrayView output,RealConstArrayView input) override
  {
    _applyView(output,input);
  }
  void apply(Int32ArrayView output,Int32ConstArrayView input) override
  {
    _applyView(output,input);
  }
  void apply(Int16ArrayView output,Int16ConstArrayView input) override
  {
    _applyView(output,input);
  }
  void apply(Int64ArrayView output,Int64ConstArrayView input) override
  {
    _applyView(output,input);
  }
  void apply(Real2ArrayView output,Real2ConstArrayView input) override
  {
    _applyView(output,input);
  }
  void apply(Real3ArrayView output,Real3ConstArrayView input) override
  {
    _applyView(output,input);
  }
  void apply(Real2x2ArrayView output,Real2x2ConstArrayView input) override
  {
    _applyView(output,input);
  }
  void apply(Real3x3ArrayView output,Real3x3ConstArrayView input) override
  {
    _applyView(output,input);
  }
  void apply(ArrayView<Int8> output,ConstArrayView<Int8> input) override
  {
    _applyView(output,input);
  }
  void apply(ArrayView<Float16> output,ConstArrayView<Float16> input) override
  {
    _applyView(output,input);
  }
  void apply(ArrayView<BFloat16> output,ConstArrayView<BFloat16> input) override
  {
    _applyView(output,input);
  }
  void apply(ArrayView<Float32> output,ConstArrayView<Float32> input) override
  {
    _applyView(output,input);
  }

 private:

  template<typename DataType> void
  _applySpan(Span<DataType> output,Span<const DataType> input)
  {
    for( Int64 i=0, n=input.size(); i<n; ++i )
      output[i] = m_operator(output[i],input[i]);
  }

 public:

  void applySpan(Span<Byte> output,Span<const Byte> input) override
  {
    _applySpan(output,input);
  }
  void applySpan(Span<Real> output,Span<const Real> input) override
  {
    _applySpan(output,input);
  }
  void applySpan(Span<Int16> output,Span<const Int16> input) override
  {
    _applySpan(output,input);
  }
  void applySpan(Span<Int32> output,Span<const Int32> input) override
  {
    _applySpan(output,input);
  }
  void applySpan(Span<Int64> output,Span<const Int64> input) override
  {
    _applySpan(output,input);
  }
  void applySpan(Span<Real2> output,Span<const Real2> input) override
  {
    _applySpan(output,input);
  }
  void applySpan(Span<Real3> output,Span<const Real3> input) override
  {
    _applySpan(output,input);
  }
  void applySpan(Span<Real2x2> output,Span<const Real2x2> input) override
  {
    _applySpan(output,input);
  }
  void applySpan(Span<Real3x3> output,Span<const Real3x3> input) override
  {
    _applySpan(output,input);
  }
  void applySpan(Span<Int8> output,Span<const Int8> input) override
  {
    _applySpan(output,input);
  }
  void applySpan(Span<Float16> output,Span<const Float16> input) override
  {
    _applySpan(output,input);
  }
  void applySpan(Span<BFloat16> output,Span<const BFloat16> input) override
  {
    _applySpan(output,input);
  }
  void applySpan(Span<Float32> output,Span<const Float32> input) override
  {
    _applySpan(output,input);
  }

 private:

  DataOperator m_operator;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

extern "C++" ARCANE_IMPL_EXPORT IDataOperation*
arcaneCreateDataOperation(Parallel::eReduceType rt);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
#endif
