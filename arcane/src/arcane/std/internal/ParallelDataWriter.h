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
#ifndef ARCANE_STD_INTERNAL_PARALLELDATAWRITER_H
#define ARCANE_STD_INTERNAL_PARALLELDATAWRITER_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/ArcaneTypes.h"

#include <map>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Écrivain parallèle pour faire des sorties par uniqueId() croissant.
 *
 * Une instance de cette classe est associée à un groupe du maillage.
 */
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
  void sort(Int32ConstArrayView local_ids, Int64ConstArrayView items_uid);
  Ref<IData> getSortedValues(IData* data);

 private:

  Impl* m_p;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief liste de 'ParallelDataWriter'.
 */
class ParallelDataWriterList
{
 public:

  Ref<ParallelDataWriter> getOrCreateWriter(const ItemGroup& group);

 private:

  std::map<ItemGroup, Ref<ParallelDataWriter>> m_data_writers;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
