// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* DataTracer.h                                                (C) 2000-2007 */
/*                                                                           */
/* Traceur d'une donnée.                                                     */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_DATATYPE_DATATRACER_H
#define ARCANE_DATATYPE_DATATRACER_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/ITraceMng.h"
#include "arcane/utils/String.h"

#include "arcane/core/datatype/IDataTracer.h"
#include "arcane/core/datatype/DataTypes.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_BEGIN_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \internal
 * \brief DataTracer pour une donées d'un type \a DataType.
 */
template<typename DataType>
class DataTracerT
: public IDataTracerT<DataType>
{
 public:
  DataTracerT(ITraceMng* msg,Integer index,eTraceType trace_type,const String& var_name)
    : m_msg(msg), m_index(index), m_trace_type(trace_type), m_var_name(var_name)
    {
    }
  virtual ~DataTracerT() {}
 public:
  virtual void traceAccess(const DataType& value)
    {
      m_msg->info() << "TraceAccess " << m_var_name << " i=" << m_index
                    << " V=" << value;
      arcaneTraceBreakpoint();
   }
  virtual void traceRead(const DataType& value)
    {
      if (m_trace_type==TT_Read)
        m_msg->info() << "TraceRead " << m_var_name << " i=" << m_index
                      << " V=" << value;
      arcaneTraceBreakpoint();
   }
  virtual void traceWrite(const DataType& old_value,const DataType& new_value)
    {
      if (m_trace_type==TT_Write)
        m_msg->info() << "TraceWrite " << m_var_name << " i=" << m_index
                      << " V=" << old_value << ' ' << new_value;
      arcaneTraceBreakpoint();
   }
 private:
  ITraceMng* m_msg;
  Integer m_index;
  eTraceType m_trace_type;
  String m_var_name;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_END_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  

