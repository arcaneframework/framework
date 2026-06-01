// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ParallelDataReader.h                                        (C) 2000-2024 */
/*                                                                           */
/* Parallel IData Reader.                                                    */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_STD_INTERNAL_PARALLELDATAREADER_H
#define ARCANE_STD_INTERNAL_PARALLELDATAREADER_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/UtilsTypes.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{
class IParallelMng;
class IData;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Parallel reading.
 *
 * An instance of this class is associated with a mesh group.
 *
 * To use it, each rank of IParallelMng must specify:
 * - the list of uids it wants, to be filled in wantedUniqueIds()
 * - the list of uids managed by this rank, sorted in ascending order, to be filled
 * in writtenUniqueIds().
 * Once this is done, the sort() method must be called to calculate
 * the information needed for sending and receiving values.
 *
 * The instance is then usable for all variables that rely on this group, and getSortedValues()
 * must be called to retrieve the values for a variable.
 *
 */
class ParallelDataReader
{
  class Impl;

 public:

  explicit ParallelDataReader(IParallelMng* pm);
  ParallelDataReader(const ParallelDataReader& rhs) = delete;
  ~ParallelDataReader();

 public:

  Array<Int64>& writtenUniqueIds();
  Array<Int64>& wantedUniqueIds();
  void sort();
  void getSortedValues(IData* written_data, IData* data);

 private:

  Impl* m_p;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
