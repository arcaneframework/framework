// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* LimaUtils.h                                                 (C) 2000-2026 */
/*                                                                           */
/* Utility functions for Lima.                                               */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_INTERNAL_LIMAUTILS_H
#define ARCANE_CORE_INTERNAL_LIMAUTILS_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/lima/ArcaneLimaGlobal.h"
#include "arcane/core/IMeshReader.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class IItemFamily;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class ARCANE_LIMA_EXPORT LimaUtils
{
 public:

  LimaUtils() = default;

  ~LimaUtils() = default;

 public:

  /*!
   * \brief Creates a group of entities.
   *
   * To ensure reproducibility, ensures that the entities are
   * sorted according to their localid. Also ensures there are no duplicates
   * in the list because Lima allows it but Arcane does not.
   */
  static void createGroup(IItemFamily* family, const String& name, Int32ArrayView local_ids);

  static IMeshReader::eReturnType _directLimaPartitionMalipp(ITimerMng* timer_mng, IPrimaryMesh* mesh, const String& filename, Real length_multiplier);
  static IMeshReader::eReturnType _directLimaPartitionMalipp2(ITimerMng* timer_mng, IPrimaryMesh* mesh, const String& filename, Real length_multiplier);
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
