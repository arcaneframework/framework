// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* AbstractDataVisitor.h                                       (C) 2000-2025 */
/*                                                                           */
/* Visiteur abstrait pour une donnée.                                        */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_ABSTRACTDATAVISITOR_H
#define ARCANE_CORE_ABSTRACTDATAVISITOR_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/core/IDataVisitor.h"

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

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Visiteur abstrait pour une donnée scalaire.
 *
 * Ce visiteur lève une exception pour chaque fonction applyVisitor()
 * non réimplémentée.
 */
class ARCANE_CORE_EXPORT AbstractScalarDataVisitor
: public IScalarDataVisitor
{
 public:

  virtual void applyVisitor(IScalarDataT<Byte>* data);
  virtual void applyVisitor(IScalarDataT<Real>* data);
  virtual void applyVisitor(IScalarDataT<Int8>* data);
  virtual void applyVisitor(IScalarDataT<Int16>* data);
  virtual void applyVisitor(IScalarDataT<Int32>* data);
  virtual void applyVisitor(IScalarDataT<Int64>* data);
  virtual void applyVisitor(IScalarDataT<Real2>* data);
  virtual void applyVisitor(IScalarDataT<Real3>* data);
  virtual void applyVisitor(IScalarDataT<Real2x2>* data);
  virtual void applyVisitor(IScalarDataT<Real3x3>* data);
  virtual void applyVisitor(IScalarDataT<Float16>* data);
  virtual void applyVisitor(IScalarDataT<BFloat16>* data);
  virtual void applyVisitor(IScalarDataT<Float32>* data);
  virtual void applyVisitor(IScalarDataT<String>* data);

 protected:
  
  void _throwException(eDataType dt);
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Visiteur abstrait pour une donnée tableau.
 *
 * Ce visiteur lève une exception pour chaque fonction applyVisitor()
 * non réimplémentée.
 */
class ARCANE_CORE_EXPORT AbstractArrayDataVisitor
: public IArrayDataVisitor
{
 public:

  virtual void applyVisitor(IArrayDataT<Byte>* data);
  virtual void applyVisitor(IArrayDataT<Real>* data);
  virtual void applyVisitor(IArrayDataT<Int8>* data);
  virtual void applyVisitor(IArrayDataT<Int16>* data);
  virtual void applyVisitor(IArrayDataT<Int32>* data);
  virtual void applyVisitor(IArrayDataT<Int64>* data);
  virtual void applyVisitor(IArrayDataT<Real2>* data);
  virtual void applyVisitor(IArrayDataT<Real3>* data);
  virtual void applyVisitor(IArrayDataT<Real2x2>* data);
  virtual void applyVisitor(IArrayDataT<Real3x3>* data);
  virtual void applyVisitor(IArrayDataT<Float16>* data);
  virtual void applyVisitor(IArrayDataT<BFloat16>* data);
  virtual void applyVisitor(IArrayDataT<Float32>* data);
  virtual void applyVisitor(IArrayDataT<String>* data);

 protected:
  
  void _throwException(eDataType dt);
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Visiteur abstrait pour une donnée tableau 2D.
 *
 * Ce visiteur lève une exception pour chaque fonction applyVisitor()
 * non réimplémentée.
 */
class ARCANE_CORE_EXPORT AbstractArray2DataVisitor
: public IArray2DataVisitor
{
 public:

  virtual void applyVisitor(IArray2DataT<Byte>* data);
  virtual void applyVisitor(IArray2DataT<Real>* data);
  virtual void applyVisitor(IArray2DataT<Int8>* data);
  virtual void applyVisitor(IArray2DataT<Int16>* data);
  virtual void applyVisitor(IArray2DataT<Int32>* data);
  virtual void applyVisitor(IArray2DataT<Int64>* data);
  virtual void applyVisitor(IArray2DataT<Real2>* data);
  virtual void applyVisitor(IArray2DataT<Real3>* data);
  virtual void applyVisitor(IArray2DataT<Real2x2>* data);
  virtual void applyVisitor(IArray2DataT<Real3x3>* data);
  virtual void applyVisitor(IArray2DataT<Float16>* data);
  virtual void applyVisitor(IArray2DataT<BFloat16>* data);
  virtual void applyVisitor(IArray2DataT<Float32>* data);

 protected:
  
  void _throwException(eDataType dt);
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Visiteur abstrait pour une donnée tableau 2D à taille variable.
 *
 * Ce visiteur lève une exception pour chaque fonction applyVisitor()
 * non réimplémentée.
 */
class ARCANE_CORE_EXPORT AbstractMultiArray2DataVisitor
: public IMultiArray2DataVisitor
{
  void _throwException(eDataType dt);
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Visiteur abstrait pour une donnée.
 *
 * Ce visiteur se contente de rediriger vers le visiteur scalaire,
 * tableau 1D ou tableau 2D concerné.
 */
class ARCANE_CORE_EXPORT AbstractDataVisitor
: public AbstractScalarDataVisitor
, public AbstractArrayDataVisitor
, public AbstractArray2DataVisitor
, public AbstractMultiArray2DataVisitor
, public IDataVisitor
{
 public:

  virtual void applyDataVisitor(IScalarData* data);
  virtual void applyDataVisitor(IArrayData* data);
  virtual void applyDataVisitor(IArray2Data* data);
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  

