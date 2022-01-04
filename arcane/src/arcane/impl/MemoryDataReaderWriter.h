// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* MemoryDataReaderWriter.h                                    (C) 2000-2009 */
/*                                                                           */
/* Lecture/ecriture des données en mémoire.                                  */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_IMPL_MEMORYDATAREADERWRITER_H
#define ARCANE_IMPL_MEMORYDATAREADERWRITER_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/TraceAccessor.h"
#include "arcane/IDataReaderWriter.h"

#include <map>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_BEGIN_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class IVariable;
class IData;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \ingroup IO
 * \brief Lecture/ecriture des données en mémoire.
 *
 * Cette classe sert par exemple pour sauvegarder et restaurer les données
 * pour le retour-arrière.
 */
class  ARCANE_IMPL_EXPORT MemoryDataReaderWriter
: public TraceAccessor
, public IDataReaderWriter
{
 private:

  typedef std::map<String,Ref<IData>> VarToDataMap;

 public:

  MemoryDataReaderWriter(ITraceMng* tm)
  : TraceAccessor(tm)
  {
  }
  virtual ~MemoryDataReaderWriter();

 public:
  
  virtual void beginWrite(const VariableCollection& vars);
  virtual void endWrite(){}
  virtual void setMetaData(const String& meta_data) { m_meta_data = meta_data; }
  virtual void write(IVariable* var,IData* data);

  virtual void beginRead(const VariableCollection&){}
  virtual void endRead(){}
  virtual String metaData() { return m_meta_data; }
  virtual void read(IVariable* var,IData* data);

  void free();

 private:

  Ref<IData> _findData(IVariable* var);

 private:

  String m_meta_data;
  VarToDataMap m_vars_to_data;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_END_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  
