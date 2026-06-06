// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* IDataTracer.h                                               (C) 2000-2005 */
/*                                                                           */
/* Interface of a tracer.                                                    */
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

//! Breakpoint usable by a debugger for tracing
extern "C" ARCANE_DATATYPE_EXPORT 
void arcaneTraceBreakpoint();

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \internal
 * \brief Interface of a tracer.
 */
template<typename DataType>
class IDataTracerT
{
 protected:
  IDataTracerT() {}
 public:
  virtual ~IDataTracerT() {}
 public:
  //! Trace message for an access (read or write) to a value
  virtual void traceAccess(const DataType& value) =0;
  //! Trace message for a read access of a value
  virtual void traceRead(const DataType& value) =0;
  //! Trace message for a write access of a value
  virtual void traceWrite(const DataType& old_value,const DataType& new_value) =0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_END_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
