// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2021 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* MemoryAccessInfo.h                                          (C) 2000-2006 */
/*                                                                           */
/* Informations sur un accès mémoire.                                        */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_UTILS_MEMORYACCESSINFO_H
#define ARCANE_UTILS_MEMORYACCESSINFO_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/Numeric.h"
#include "arcane/utils/UtilsTypes.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_BEGIN_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

enum eMemoryAccessMessage
{
  MAM_UnitializedMemoryRead,
  MAM_MayBeUnitializedMemoryRead,
  MAM_NotSyncRead
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class ARCANE_UTILS_EXPORT IMemoryAccessTrace
{
 public:
  virtual ~IMemoryAccessTrace() {}
 public:
  virtual void notify(eMemoryAccessMessage message,Integer handle) =0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class ARCANE_UTILS_EXPORT MemoryAccessInfo
{
 public:
 
  MemoryAccessInfo(Byte* info,IMemoryAccessTrace* trace,Integer handle)
  : m_info(info), m_trace(trace), m_handle(handle) {}
  MemoryAccessInfo(const MemoryAccessInfo& rhs)
  : m_info(rhs.m_info), m_trace(rhs.m_trace), m_handle(rhs.m_handle) {}

 public:

  void setRead() const;
  void setWrite() const;
  void setWriteAndSync() const;
  void setSync() const;
  void setNeedSync(bool) const;
  void setReadOrWrite() const;
  void setCreate() const;

 private:

  Byte* m_info;
  IMemoryAccessTrace* m_trace;
  Integer m_handle;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_END_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
