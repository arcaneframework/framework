// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* IDataTracer.h                                               (C) 2000-2005 */
/*                                                                           */
/* Interface d'un traceur.                                                   */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_DATATYPE_IDATATRACER_H
#define ARCANE_DATATYPE_IDATATRACER_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/


#include "arcane/utils/ArcaneGlobal.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_BEGIN_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

//! Point d'arrêt utilisable par un débuggeur pour une trace
extern "C" ARCANE_DATATYPE_EXPORT 
void arcaneTraceBreakpoint();

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \internal
 * \brief Interface d'un tracer.
 */
template<typename DataType>
class IDataTracerT
{
 protected:
  IDataTracerT() {}
 public:
  virtual ~IDataTracerT() {}
 public:
  //! Message de trace pour un accès (lecture ou écriture) à une valeur
  virtual void traceAccess(const DataType& value) =0;
  //! Message de trace pour un accès en lecture d'une valeur
  virtual void traceRead(const DataType& value) =0;
  //! Message de trace pour un accès en écriture d'une valeur
  virtual void traceWrite(const DataType& old_value,const DataType& new_value) =0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_END_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  

