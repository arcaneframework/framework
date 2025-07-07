// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2024 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* BasicWriter.cc                                              (C) 2000-2024 */
/*                                                                           */
/* Ecriture simple pour les protections/reprises.                            */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/std/internal/BasicWriter.h"

#include "arcane/utils/FatalErrorException.h"
#include "arcane/utils/StringBuilder.h"
#include "arcane/utils/PlatformUtils.h"
#include "arcane/utils/JSONWriter.h"
#include "arcane/utils/IDataCompressor.h"
#include "arcane/utils/MemoryView.h"
#include "arcane/utils/Ref.h"
#include "arcane/utils/IHashAlgorithm.h"

#include "arcane/core/IParallelMng.h"
#include "arcane/core/ItemGroup.h"
#include "arcane/core/IVariable.h"
#include "arcane/core/IItemFamily.h"
#include "arcane/core/IData.h"
#include "arcane/core/internal/IVariableInternal.h"

#include "arcane/std/internal/ParallelDataWriter.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::impl
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

BasicWriter::
BasicWriter(IApplication* app, IParallelMng* pm, const String& path,
            eOpenMode open_mode, Int32 version, bool want_parallel)
: BasicReaderWriterCommon(app, pm, path, open_mode)
, m_want_parallel(want_parallel)
, m_version(version)
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void BasicWriter::
initialize()
{
  _checkNoInit();

  Int32 rank = m_parallel_mng->commRank();
  if (m_open_mode == OpenModeTruncate && m_parallel_mng->isMasterIO())
    platform::recursiveCreateDirectory(m_path);
  m_parallel_mng->barrier();
  String filename = _getBasicVariableFile(m_version, m_path, rank);
  m_text_writer = makeRef(new KeyValueTextWriter(traceMng(), filename, m_version));
  m_text_writer->setDataCompressor(m_data_compressor);
  m_text_writer->setHashAlgorithm(m_hash_algorithm);

  // Permet de surcharger le service utilisé pour la compression par une
  // variable d'environnement si aucun n'est positionné
  if (!m_data_compressor.get()) {
    String data_compressor_name = platform::getEnvironmentVariable("ARCANE_DEFLATER");
    if (!data_compressor_name.null()) {
      data_compressor_name = data_compressor_name + "DataCompressor";
      auto bc = _createDeflater(m_application, data_compressor_name);
      info() << "Use data_compressor from environment variable ARCANE_DEFLATER name=" << data_compressor_name;
      m_data_compressor = bc;
      m_text_writer->setDataCompressor(bc);
    }
  }

  // Idem pour le service de calcul de hash
  if (!m_hash_algorithm.get()) {
    String hash_algorithm_name = platform::getEnvironmentVariable("ARCANE_HASHALGORITHM");
    if (hash_algorithm_name.null())
      hash_algorithm_name = "SHA3_256";
    else
      info() << "Use hash algorithm from environment variable ARCANE_HASHALGORITHM name=" << hash_algorithm_name;
    hash_algorithm_name = hash_algorithm_name + "HashAlgorithm";
    auto v = _createHashAlgorithm(m_application, hash_algorithm_name);
    m_hash_algorithm = v;
    m_text_writer->setHashAlgorithm(v);
  }

  // Pour test, permet de spécifier un service pour le calcul du hash global.
  if (!m_compare_hash_algorithm.get()) {
    String algo_name = platform::getEnvironmentVariable("ARCANE_COMPAREHASHALGORITHM");
    if (!algo_name.empty()) {
      info() << "Use global hash algorithm from environment variable ARCANE_COMPAREHASHALGORITHM name=" << algo_name;
      algo_name = algo_name + "HashAlgorithm";
      auto v = _createHashAlgorithm(m_application, algo_name);
      m_compare_hash_algorithm = v;
    }
  }

  m_global_writer = new BasicGenericWriter(m_application, m_version, m_text_writer);
  if (m_verbose_level > 0)
    info() << "** OPEN MODE = " << m_open_mode;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void BasicWriter::
_checkNoInit()
{
  if (m_is_init)
    ARCANE_FATAL("initialize() has already been called");
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Ref<ParallelDataWriter> BasicWriter::
_getWriter(IVariable* var)
{
  return m_parallel_data_writers.getOrCreateWriter(var->itemGroup());
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void BasicWriter::
_directWriteVal(IVariable* var, IData* data)
{
  info(4) << "DIRECT WRITE VAL v=" << var->fullName();

  IData* write_data = data;
  Int64ConstArrayView written_unique_ids;
  Int64UniqueArray wanted_unique_ids;
  Int64UniqueArray sequential_written_unique_ids;

  Ref<IData> allocated_write_data;
  const bool is_mesh_variable = (var->itemKind() != IK_Unknown);
  if (is_mesh_variable) {
    ItemGroup group = var->itemGroup();
    if (m_want_parallel) {
      Ref<ParallelDataWriter> writer = _getWriter(var);
      written_unique_ids = writer->sortedUniqueIds();
      allocated_write_data = writer->getSortedValues(data);
      write_data = allocated_write_data.get();
    }
    else {
      // TODO vérifier que les uniqueId() sont bien triés.
      // Normalement c'est toujours le cas.
      _fillUniqueIds(group, sequential_written_unique_ids);
      written_unique_ids = sequential_written_unique_ids.view();
    }
    // Ecrit les informations du groupe si c'est la première fois qu'on accède à ce groupe.
    if (m_written_groups.find(group) == m_written_groups.end()) {
      info(5) << "WRITE GROUP " << group.name();
      const IItemFamily* item_family = group.itemFamily();
      const String& gname = group.name();
      String group_full_name = item_family->fullName() + "_" + gname;
      _fillUniqueIds(group, wanted_unique_ids);
      if (m_is_save_values)
        m_global_writer->writeItemGroup(group_full_name, written_unique_ids, wanted_unique_ids.view());
      m_written_groups.insert(group);
    }
  }

  Ref<ISerializedData> sdata(write_data->createSerializedDataRef(false));
  String compare_hash;
  if (is_mesh_variable) {
    compare_hash = _computeCompareHash(var, write_data);
  }
  m_global_writer->writeData(var->fullName(), sdata.get(), compare_hash, m_is_save_values);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Calcul un hash de comparaison pour la variable.
 *
 * Le rang maitre récupère un tableau contenant la concaténation des valeurs
 * de la variable pour tous les rangs et calcul un hash sur ce tableau.
 *
 * Comme ce tableau est trié suivant les uniqueId(), il peut servir à
 * comparer directement la valeur de la variable.
 *
 * \return le hash sous-forme de chaîne de caratères si un algorithme de hash
 * est spécifié.
 */
String BasicWriter::
_computeCompareHash(IVariable* var, IData* write_data)
{
  IHashAlgorithm* hash_algo = m_compare_hash_algorithm.get();
  if (!hash_algo)
    return {};
  return var->_internalApi()->computeComparisonHashCollective(hash_algo, write_data);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void BasicWriter::
write(IVariable* var, IData* data)
{
  if (var->isPartial()) {
    info() << "** WARNING: partial variable not implemented in BasicWriter";
    return;
  }
  _directWriteVal(var, data);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void BasicWriter::
setMetaData(const String& meta_data)
{
  // Dans la version 3, les méta-données de la protection sont dans la
  // base de données.
  if (m_version >= 3) {
    Span<const Byte> bytes = meta_data.utf8();
    Int64 length = bytes.length();
    String key_name = "Global:CheckpointMetadata";
    m_text_writer->setExtents(key_name, Int64ConstArrayView(1, &length));
    m_text_writer->write(key_name, asBytes(bytes));
  }
  else {
    Int32 my_rank = m_parallel_mng->commRank();
    String filename = _getMetaDataFileName(my_rank);
    std::ofstream ofile(filename.localstr(), ios::binary);
    meta_data.writeBytes(ofile);
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void BasicWriter::
beginWrite(const VariableCollection& vars)
{
  ARCANE_UNUSED(vars);
  Int32 my_rank = m_parallel_mng->commRank();
  m_global_writer->initialize(m_path, my_rank);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void BasicWriter::
_endWriteV3()
{
  const Int64 nb_part = m_parallel_mng->commSize();

  // Sauvegarde les informations au format JSON
  JSONWriter jsw;

  {
    JSONWriter::Object main_object(jsw);
    jsw.writeKey(_getArcaneDBTag());
    {
      JSONWriter::Object db_object(jsw);
      jsw.write("Version", (Int64)m_version);
      jsw.write("NbPart", nb_part);
      jsw.write("HasValues", m_is_save_values);

      String data_compressor_name;
      Int64 data_compressor_min_size = 0;
      if (m_data_compressor.get()) {
        data_compressor_name = m_data_compressor->name();
        data_compressor_min_size = m_data_compressor->minCompressSize();
      }
      jsw.write("DataCompressor", data_compressor_name);
      jsw.write("DataCompressorMinSize", String::fromNumber(data_compressor_min_size));

      // Sauve le nom de l'algorithme de hash
      {
        String name;
        if (m_hash_algorithm.get())
          name = m_hash_algorithm->name();
        jsw.write("HashAlgorithm", name);
      }

      // Sauve le nom de l'algorithme de hash pour les comparaisons
      {
        String name;
        if (m_compare_hash_algorithm.get())
          name = m_compare_hash_algorithm->name();
        jsw.write("ComparisonHashAlgorithm", name);
      }
    }
  }

  StringBuilder filename = m_path;
  filename += "/arcane_acr_db.json";
  String fn = filename.toString();
  std::ofstream ofile(fn.localstr());
  ofile << jsw.getBuffer();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void BasicWriter::
endWrite()
{
  const IParallelMng* pm = m_parallel_mng;
  if (pm->isMasterIO()) {
    if (m_version >= 3) {
      _endWriteV3();
    }
    else {
      Int64 nb_part = pm->commSize();
      StringBuilder filename = m_path;
      filename += "/infos.txt";
      String fn = filename.toString();
      std::ofstream ofile(fn.localstr());
      ofile << nb_part << '\n';
    }
  }
  m_global_writer->endWrite();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::impl

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
