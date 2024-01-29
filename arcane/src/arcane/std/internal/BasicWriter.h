// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2024 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* BasicWriter.h                                               (C) 2000-2024 */
/*                                                                           */
/* Ecrivain simple.                                                          */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_STD_INTERNAL_BASICWRITER_H
#define ARCANE_STD_INTERANL_BASICWRITER_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/std/internal/BasicReaderWriter.h"

#include <map>
#include <set>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::impl
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Lecture/Ecriture simple.
 */
class BasicWriter
: public BasicReaderWriterCommon
, public IDataWriter
{
 public:

  BasicWriter(IApplication* app, IParallelMng* pm, const String& path,
              eOpenMode open_mode, Integer version, bool want_parallel);

 public:

  //!@{ Implémentation de IDataWriter
  void beginWrite(const VariableCollection& vars) override;
  void endWrite() override;
  void setMetaData(const String& meta_data) override;
  void write(IVariable* v, IData* data) override;
  //@}

 public:

  //! Positionne le service de compression. Doit être appelé avant initialize()
  void setDataCompressor(Ref<IDataCompressor> data_compressor)
  {
    _checkNoInit();
    m_data_compressor = data_compressor;
  }
  //! Positionne le service de calcul de hash pour la comparaison. Doit être appelé avant initialize()
  void setCompareHashAlgorithm(Ref<IHashAlgorithm> hash_algo)
  {
    _checkNoInit();
    m_compare_hash_algorithm = hash_algo;
  }
  void initialize();

 private:

  bool m_want_parallel = false;
  bool m_is_gather = false;
  bool m_is_init = false;
  Int32 m_version = -1;

  Ref<IDataCompressor> m_data_compressor;
  Ref<IHashAlgorithm> m_compare_hash_algorithm;
  Ref<IHashAlgorithm> m_hash_algorithm;
  Ref<KeyValueTextWriter> m_text_writer;

  std::map<ItemGroup, Ref<ParallelDataWriter>> m_parallel_data_writers;
  std::set<ItemGroup> m_written_groups;

  ScopedPtrT<IGenericWriter> m_global_writer;

 private:

  void _directWriteVal(IVariable* v, IData* data);
  String _computeCompareHash(IVariable* var, IData* write_data);
  Ref<ParallelDataWriter> _getWriter(IVariable* var);
  void _endWriteV3();
  void _checkNoInit();
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::impl

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
