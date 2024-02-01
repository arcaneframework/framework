// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2024 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* BasicReader.h                                               (C) 2000-2024 */
/*                                                                           */
/* Lecteur simple.                                                           */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_STD_INTERNAL_BASICREADER_H
#define ARCANE_STD_INTERANL_BASICREADER_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/std/internal/BasicReaderWriter.h"

#include <map>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::impl
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Lecteur simple.
 */
class BasicReader
: public BasicReaderWriterCommon
, public IDataReader
, public IDataReader2
{
 public:

  /*!
    \brief  Interface pour retrouver le groupe associée à une variable à partir
    de ces meta-données.
  */
  class IItemGroupFinder
  {
   public:

    virtual ~IItemGroupFinder() = default;
    virtual ItemGroup getWantedGroup(VariableMetaData* vmd) = 0;
  };

 public:

  BasicReader(IApplication* app, IParallelMng* pm, Int32 forced_rank_to_read,
              const String& path, bool want_parallel);

 public:

  void beginRead(const VariableCollection& vars) override;
  void endRead() override {}
  String metaData() override;
  void read(IVariable* v, IData* data) override;

  void fillMetaData(ByteArray& bytes) override;
  void beginRead(const DataReaderInfo& infos) override;
  void read(const VariableDataReadInfo& infos) override;

 public:

  void initialize();
  void setItemGroupFinder(IItemGroupFinder* group_finder)
  {
    m_item_group_finder = group_finder;
  }
  void fillComparisonHash(std::map<String, String>& comparison_hash_map);

 private:

  bool m_want_parallel = false;
  Integer m_nb_written_part = 0;
  Int32 m_version = -1;

  Int32 m_first_rank_to_read = -1;
  Int32 m_nb_rank_to_read = -1;
  Int32 m_forced_rank_to_read = -1;

  std::map<String, Ref<ParallelDataReader>> m_parallel_data_readers;
  UniqueArray<Ref<IGenericReader>> m_global_readers;
  IItemGroupFinder* m_item_group_finder;
  Ref<KeyValueTextReader> m_forced_rank_to_read_text_reader; //!< Lecteur pour le premier rang à lire.
  Ref<IDataCompressor> m_data_compressor;
  Ref<IHashAlgorithm> m_comparison_hash_algorithm;

 private:

  void _directReadVal(VariableMetaData* varmd, IData* data);

  Ref<ParallelDataReader> _getReader(VariableMetaData* varmd);
  void _setRanksToRead();
  Ref<IGenericReader> _readOwnMetaDataAndCreateReader(Int32 rank);
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::impl

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
