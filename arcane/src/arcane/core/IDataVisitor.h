// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2024 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* IDataVisitor.h                                              (C) 2000-2024 */
/*                                                                           */
/* Interface du pattern visitor pour une donnée.                             */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_IDATAVISITOR_H
#define ARCANE_CORE_IDATAVISITOR_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/core/ArcaneTypes.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class IData;
class IScalarData;
class IArrayData;
class IArray2Data;
class IMultiArray2Data;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Interface du pattern visitor pour une donnée.
 *
 * Ce pattern se contente de transférer le visiteur sur le
 * visiteur IScalarDataVisitor, IArrayDataVisitor et IArray2DataVisitor.
 */
class ARCANE_CORE_EXPORT IDataVisitor
{
 public:

  virtual ~IDataVisitor() {}

 public:

  virtual void applyDataVisitor(IScalarData* data) = 0;
  virtual void applyDataVisitor(IArrayData* data) = 0;
  virtual void applyDataVisitor(IArray2Data* data) = 0;
  // No longer used. Throw NotSupportedException
  virtual void applyDataVisitor(IMultiArray2Data*);
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Interface du pattern visitor pour une donnée scalaire.
 */
class ARCANE_CORE_EXPORT IScalarDataVisitor
{
 public:

  virtual ~IScalarDataVisitor() {}

 public:

  virtual void applyVisitor(IScalarDataT<Byte>* data) = 0;
  virtual void applyVisitor(IScalarDataT<Real>* data) = 0;
  virtual void applyVisitor(IScalarDataT<Int8>* data) = 0;
  virtual void applyVisitor(IScalarDataT<Int16>* data) = 0;
  virtual void applyVisitor(IScalarDataT<Int32>* data) = 0;
  virtual void applyVisitor(IScalarDataT<Int64>* data) = 0;
  virtual void applyVisitor(IScalarDataT<Real2>* data) = 0;
  virtual void applyVisitor(IScalarDataT<Real3>* data) = 0;
  virtual void applyVisitor(IScalarDataT<Real2x2>* data) = 0;
  virtual void applyVisitor(IScalarDataT<Real3x3>* data) = 0;
  virtual void applyVisitor(IScalarDataT<Float16>* data) = 0;
  virtual void applyVisitor(IScalarDataT<BFloat16>* data) = 0;
  virtual void applyVisitor(IScalarDataT<Float32>* data) = 0;
  virtual void applyVisitor(IScalarDataT<String>* data) = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Interface du pattern visitor pour une donnée tableau.
 */
class ARCANE_CORE_EXPORT IArrayDataVisitor
{
 public:

  virtual ~IArrayDataVisitor() {}

 public:

  virtual void applyVisitor(IArrayDataT<Byte>* data) = 0;
  virtual void applyVisitor(IArrayDataT<Real>* data) = 0;
  virtual void applyVisitor(IArrayDataT<Int8>* data) = 0;
  virtual void applyVisitor(IArrayDataT<Int16>* data) = 0;
  virtual void applyVisitor(IArrayDataT<Int32>* data) = 0;
  virtual void applyVisitor(IArrayDataT<Int64>* data) = 0;
  virtual void applyVisitor(IArrayDataT<Real2>* data) = 0;
  virtual void applyVisitor(IArrayDataT<Real3>* data) = 0;
  virtual void applyVisitor(IArrayDataT<Real2x2>* data) = 0;
  virtual void applyVisitor(IArrayDataT<Real3x3>* data) = 0;
  virtual void applyVisitor(IArrayDataT<Float16>* data) = 0;
  virtual void applyVisitor(IArrayDataT<BFloat16>* data) = 0;
  virtual void applyVisitor(IArrayDataT<Float32>* data) = 0;
  virtual void applyVisitor(IArrayDataT<String>* data) = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Interface du pattern visitor pour une donnée tableau 2D.
 */
class ARCANE_CORE_EXPORT IArray2DataVisitor
{
 public:

  virtual ~IArray2DataVisitor() {}

 public:

  virtual void applyVisitor(IArray2DataT<Byte>* data) = 0;
  virtual void applyVisitor(IArray2DataT<Real>* data) = 0;
  virtual void applyVisitor(IArray2DataT<Int8>* data) = 0;
  virtual void applyVisitor(IArray2DataT<Int16>* data) = 0;
  virtual void applyVisitor(IArray2DataT<Int32>* data) = 0;
  virtual void applyVisitor(IArray2DataT<Int64>* data) = 0;
  virtual void applyVisitor(IArray2DataT<Real2>* data) = 0;
  virtual void applyVisitor(IArray2DataT<Real3>* data) = 0;
  virtual void applyVisitor(IArray2DataT<Real2x2>* data) = 0;
  virtual void applyVisitor(IArray2DataT<Real3x3>* data) = 0;
  virtual void applyVisitor(IArray2DataT<Float16>* data) = 0;
  virtual void applyVisitor(IArray2DataT<BFloat16>* data) = 0;
  virtual void applyVisitor(IArray2DataT<Float32>* data) = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Interface du pattern visitor pour une donnée tableau 2D à taille variable
 *
 * \deprecated Ne plus utiliser car le type sous-jacent n'est plus utilisé
 */
class ARCANE_CORE_EXPORT IMultiArray2DataVisitor
{
 public:

  virtual ~IMultiArray2DataVisitor() {}

 public:

  virtual void applyVisitor(IMultiArray2DataT<Byte>*) {}
  virtual void applyVisitor(IMultiArray2DataT<Real>*) {}
  virtual void applyVisitor(IMultiArray2DataT<Int16>*) {}
  virtual void applyVisitor(IMultiArray2DataT<Int32>*) {}
  virtual void applyVisitor(IMultiArray2DataT<Int64>*) {}
  virtual void applyVisitor(IMultiArray2DataT<Real2>*) {}
  virtual void applyVisitor(IMultiArray2DataT<Real3>*) {}
  virtual void applyVisitor(IMultiArray2DataT<Real2x2>*) {}
  virtual void applyVisitor(IMultiArray2DataT<Real3x3>*) {}
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
