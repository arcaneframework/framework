// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2024 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ParallelDataWriter.h                                        (C) 2000-2024 */
/*                                                                           */
/* Ecrivain de IData en parallèle.                                           */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_STD_PARALLELDATAWRITER_H
#define ARCANE_STD_PARALLELDATAWRITER_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/ArcaneTypes.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{
class IParallelMng;
class IData;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class ParallelDataWriter
{
  class Impl;

 public:

  explicit ParallelDataWriter(IParallelMng* pm);
  ParallelDataWriter(const ParallelDataWriter& rhs) = delete;
  ~ParallelDataWriter();

 public:

  Int64ConstArrayView sortedUniqueIds() const;
  void setGatherAll(bool v);
  void sort(Int32ConstArrayView local_ids,Int64ConstArrayView items_uid);
  Ref<IData> getSortedValues(IData* data);

 private:

  Impl* m_p;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
