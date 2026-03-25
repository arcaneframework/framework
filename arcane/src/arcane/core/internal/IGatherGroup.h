// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* IGatherGroup.h                                              (C) 2000-2026 */
/*                                                                           */
/* TODO .                                                   */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_INTERNAL_IGATHERGROUP_H
#define ARCANE_CORE_INTERNAL_IGATHERGROUP_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/core/ArcaneTypes.h"
#include "arccore/base/BaseTypes.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/



/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class ARCANE_CORE_EXPORT IGatherGroup
{
 public:
  class ARCANE_CORE_EXPORT IGatherGroupInfo
  {
  public:

    virtual ~IGatherGroupInfo() = default;

  public:

    virtual void computeSize(Int32 nb_elem_in) = 0;
    virtual void needRecompute() = 0;
    virtual bool isComputed() = 0;
    virtual Int32 nbElemOutput() = 0;
    virtual Int32 sizeOfOutput(Int32 sizeof_type) = 0;
    virtual SmallSpan<Int32> nbElemRecvGatherToMasterIO() = 0;
    virtual Int32 nbWriterGlobal() = 0;
  };

 public:

  virtual ~IGatherGroup() = default;

 public:

  virtual bool needGather() = 0;
  virtual void gatherToMasterIO(Int64 sizeof_elem, Span<const Byte> in, Span<Byte> out) = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
