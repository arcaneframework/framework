// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2021 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* IDataOperation.h                                            (C) 2000-2020 */
/*                                                                           */
/* Interface d'une opération sur une donnée.                                 */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_DATATYPE_IDATAOPERATION_H
#define ARCANE_DATATYPE_IDATAOPERATION_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/datatype/DataTypes.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class IDataOperation;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \internal
 * \brief Interface d'une opération sur une donnée.
 */
class IDataOperation
{
 public:
  
  virtual ~IDataOperation() {} //!< Libère les ressources

 public:

  virtual void apply(ByteArrayView output,ByteConstArrayView input) =0;
  virtual void apply(RealArrayView output,RealConstArrayView input) =0;
  virtual void apply(Int16ArrayView output,Int16ConstArrayView input) =0;
  virtual void apply(Int32ArrayView output,Int32ConstArrayView input) =0;
  virtual void apply(Int64ArrayView output,Int64ConstArrayView input) =0;
  virtual void apply(Real2ArrayView output,Real2ConstArrayView input) =0;
  virtual void apply(Real3ArrayView output,Real3ConstArrayView input) =0;
  virtual void apply(Real2x2ArrayView output,Real2x2ConstArrayView input) =0;
  virtual void apply(Real3x3ArrayView output,Real3x3ConstArrayView input) =0;

  virtual void applySpan(Span<Byte> output,Span<const Byte> input) =0;
  virtual void applySpan(Span<Real> output,Span<const Real> input) =0;
  virtual void applySpan(Span<Int16> output,Span<const Int16> input) =0;
  virtual void applySpan(Span<Int32> output,Span<const Int32> input) =0;
  virtual void applySpan(Span<Int64> output,Span<const Int64> input) =0;
  virtual void applySpan(Span<Real2> output,Span<const Real2> input) =0;
  virtual void applySpan(Span<Real3> output,Span<const Real3> input) =0;
  virtual void applySpan(Span<Real2x2> output,Span<const Real2x2> input) =0;
  virtual void applySpan(Span<Real3x3> output,Span<const Real3x3> input) =0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  
