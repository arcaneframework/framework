// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* LimaCutInfosReader.h                                        (C) 2000-2018 */
/*                                                                           */
/* Lecteur des informations de découpages avec les fichiers Lima.            */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CEA_LIMACUTINFOSREADER_H
#define ARCANE_CEA_LIMACUTINFOSREADER_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/TraceAccessor.h"
#include "arcane/ArcaneTypes.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_BEGIN_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class XmlNode;
class IParallelMng;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Construction d'un maillage 3D.
 */
class LimaCutInfosReader
: public TraceAccessor
{
 public:

  LimaCutInfosReader(IParallelMng* parallel_mng);
  virtual ~LimaCutInfosReader() {}

 public:

 public:

  void readItemsUniqueId(Int64ArrayView nodes_id,Int64ArrayView cells_id,
                         const String& dir_name);

 private:

  IParallelMng* m_parallel_mng; 

 private:

  void _readUniqueIndex(Int64ArrayView nodes_id,Int64ArrayView cells_id,const String& dir_name);
  void _readUniqueIndexFromXml(Int64ArrayView nodes_id,Int64ArrayView cells_id,
                               XmlNode root_element,Int32 rank);
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_END_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  

