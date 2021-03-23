// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2021 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* MemoryAccessInfo.cc                                         (C) 2000-2006 */
/*                                                                           */
/* Informations sur un accès mémoire.                                        */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/ArcanePrecomp.h"

#include "arcane/utils/MemoryAccessInfo.h"
#include "arcane/utils/Iostream.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_BEGIN_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

static Byte MA_HasValue = 1 << 1;
static Byte MA_IsSync = 1 << 2;
static Byte MA_NeedSync = 1 << 3;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MemoryAccessInfo::
setCreate() const
{
  *m_info = 0;
}

void MemoryAccessInfo::
setRead() const
{
  Byte v = *m_info;
  if (v & MA_HasValue){
    if ( (v & MA_NeedSync) && !(v & MA_IsSync) ){
      if (m_trace)
        m_trace->notify(MAM_NotSyncRead,m_handle);
      else
        std::cerr << "Not Sync Read\n";
    }
    return;
  }
  if (m_trace)
    m_trace->notify(MAM_UnitializedMemoryRead,m_handle);
  else
    std::cerr << "Unitialized Memory Read\n";
}

void MemoryAccessInfo::
setWrite() const
{
  Byte v = *m_info;
  *m_info |= MA_HasValue;
  if (v & MA_NeedSync)
    *m_info = static_cast<Byte>(v & (~MA_IsSync));
}

void MemoryAccessInfo::
setWriteAndSync() const
{
  *m_info |= MA_HasValue;
  setSync();
}

void MemoryAccessInfo::
setSync() const
{
  bool v = *m_info;
  if (v & MA_NeedSync)
    *m_info |= MA_IsSync;
}

void MemoryAccessInfo::
setNeedSync(bool need_sync) const
{
  Byte v = *m_info;
  if (need_sync)
    *m_info |= MA_NeedSync;
  else
    *m_info = static_cast<Byte>(v & (~MA_NeedSync));
}

void MemoryAccessInfo::
setReadOrWrite() const
{
  Byte v = *m_info;
  if (v & MA_HasValue)
    return;
  if (m_trace)
    m_trace->notify(MAM_MayBeUnitializedMemoryRead,m_handle);
  else
    std::cerr << "Value may be used unitialized\n";
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_END_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

