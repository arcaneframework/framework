// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2024 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* BasicReader.cc                                              (C) 2000-2024 */
/*                                                                           */
/* Lecture simple pour les protections/reprises.                             */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/std/internal/BasicReader.h"

#include "arcane/utils/PlatformUtils.h"
#include "arcane/utils/JSONReader.h"
#include "arcane/utils/Ref.h"

#include "arcane/core/IParallelMng.h"
#include "arcane/core/IIOMng.h"
#include "arcane/core/IData.h"
#include "arcane/core/ItemGroup.h"
#include "arcane/core/IVariable.h"
#include "arcane/core/SerializeBuffer.h"

#include "arcane/std/ParallelDataReader.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::impl
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

BasicReader::
BasicReader(IApplication* app, IParallelMng* pm, Int32 forced_rank_to_read,
            const String& path, bool want_parallel)
: BasicReaderWriterCommon(app, pm, path, BasicReaderWriterCommon::OpenModeRead)
, m_want_parallel(want_parallel)
, m_nb_written_part(0)
, m_version(-1)
, m_first_rank_to_read(0)
, m_nb_rank_to_read(0)
, m_forced_rank_to_read(forced_rank_to_read)
, m_item_group_finder(nullptr)
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void BasicReader::
initialize()
{
  info() << "BasicReader::initialize()";

  IParallelMng* pm = m_parallel_mng;
  // Si un fichier 'arcane_acr_db.json' existe alors on lit les informations
  // de ce fichier pour détecter entre autre le numéro de version. Il faut
  // le faire avant de lire les informations telles que les méta-données
  // de la protection car l'emplacement des ces dernières dépend de la version
  String db_filename = String::concat(m_path, "/arcane_acr_db.json");
  Int32 has_db_file = 0;
  {
    if (pm->isMasterIO())
      has_db_file = platform::isFileReadable(db_filename) ? 1 : 0;
    pm->broadcast(Int32ArrayView(1, &has_db_file), pm->masterIORank());
  }
  String data_compressor_name;
  String hash_algorithm_name;
  String comparison_hash_algorithm_name;
  if (has_db_file) {
    UniqueArray<Byte> bytes;
    pm->ioMng()->collectiveRead(db_filename, bytes, false);
    JSONDocument json_doc;
    json_doc.parse(bytes, db_filename);
    JSONValue root = json_doc.root();
    JSONValue jv_arcane_db = root.expectedChild(_getArcaneDBTag());
    m_version = jv_arcane_db.expectedChild("Version").valueAsInt32();
    m_nb_written_part = jv_arcane_db.expectedChild("NbPart").valueAsInt32();
    data_compressor_name = jv_arcane_db.child("DataCompressor").value();
    hash_algorithm_name = jv_arcane_db.child("HashAlgorithm").value();
    comparison_hash_algorithm_name = jv_arcane_db.child("ComparisonHashAlgorithm").value();
    info() << "**--** Begin read using database version=" << m_version
           << " nb_part=" << m_nb_written_part
           << " compressor=" << data_compressor_name
           << " hash_algorithm=" << hash_algorithm_name
           << " comparison_hash_algorithm=" << comparison_hash_algorithm_name;
  }
  else {
    // Ancien format
    // Le proc maitre lit le fichier 'infos.txt' et envoie les informations
    // aux autres. Ce format ne permet d'avoir que le nombre de parties
    // comme information.
    if (pm->isMasterIO()) {
      Integer nb_part = 0;
      String filename = String::concat(m_path, "/infos.txt");
      std::ifstream ifile(filename.localstr());
      ifile >> nb_part;
      info(4) << "** NB PART=" << nb_part;
      m_nb_written_part = nb_part;
    }
    pm->broadcast(Int32ArrayView(1, &m_nb_written_part), pm->masterIORank());
  }
  if (m_version >= 3) {
    Int32 rank_to_read = m_forced_rank_to_read;
    if (rank_to_read < 0) {
      if (m_nb_written_part > 1)
        rank_to_read = pm->commRank();
      else
        rank_to_read = 0;
    }
    String main_filename = _getBasicVariableFile(m_version, m_path, rank_to_read);
    m_forced_rank_to_read_text_reader = makeRef(new KeyValueTextReader(traceMng(), main_filename, m_version));
    if (!data_compressor_name.empty()) {
      Ref<IDataCompressor> dc = _createDeflater(m_application, data_compressor_name);
      m_forced_rank_to_read_text_reader->setDataCompressor(dc);
    }
    if (!hash_algorithm_name.empty()) {
      Ref<IHashAlgorithm> v = _createHashAlgorithm(m_application, hash_algorithm_name);
      m_forced_rank_to_read_text_reader->setHashAlgorithm(v);
    }
    if (!comparison_hash_algorithm_name.empty()) {
      Ref<IHashAlgorithm> v = _createHashAlgorithm(m_application, comparison_hash_algorithm_name);
      m_comparison_hash_algorithm = v;
    }
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void BasicReader::
_directReadVal(VariableMetaData* varmd, IData* data)
{
  info(4) << "DIRECT READ VAL v=" << varmd->fullName();
    
  bool is_item_variable = !varmd->itemFamilyName().null();
  Int32 nb_rank_to_read = m_nb_rank_to_read;
  // S'il s'agit d'une variable qui n'est pas sur le maillage,
  // il ne faut lire qu'un seul rang car il n'est pas
  // certain que cette variable soit definie partout
  if (!is_item_variable)
    if (nb_rank_to_read > 1)
      nb_rank_to_read = 1;

  UniqueArray<Ref<IData>> allocated_data;
  UniqueArray<IData*> written_data(nb_rank_to_read);

  for (Integer i = 0; i < nb_rank_to_read; ++i) {
    written_data[i] = data;
    if ((nb_rank_to_read > 1 || m_want_parallel) && is_item_variable) {
      Ref<IData> new_data = data->cloneEmptyRef();
      written_data[i] = new_data.get();
      allocated_data.add(new_data);
    }
    String vname = varmd->fullName();
    info(4) << " TRY TO READ var_full_name=" << vname;
    m_global_readers[i]->readData(vname, written_data[i]);
    if (i==0 && m_comparison_hash_algorithm.get() )
      info(5) << "COMPARISON_HASH =" << m_global_readers[i]->comparisonHashValue(vname);
  }

  if (is_item_variable) {
    Ref<ParallelDataReader> parallel_data_reader = _getReader(varmd);

    Int64UniqueArray full_written_unique_ids;
    IData* full_written_data = nullptr;

    if (nb_rank_to_read == 0) {
      // Rien à lire
      // Il faut tout de même passer dans le reader parallèle
      // pour assurer les opérations collectives
      full_written_data = nullptr;
    }
    else if (nb_rank_to_read == 1) {
      //full_written_unique_ids = written_unique_ids[0];
      full_written_data = written_data[0];
    }
    else {
      // Il faut créer une donnée qui contient l'union des written_data
      Ref<IData> allocated_written_data = data->cloneEmptyRef();
      allocated_data.add(allocated_written_data);
      full_written_data = allocated_written_data.get();
      SerializeBuffer sbuf;
      sbuf.setMode(ISerializer::ModeReserve);
      for (Int32 i = 0; i < nb_rank_to_read; ++i)
        written_data[i]->serialize(&sbuf, nullptr);
      sbuf.allocateBuffer();
      sbuf.setMode(ISerializer::ModePut);
      for (Int32 i = 0; i < nb_rank_to_read; ++i)
        written_data[i]->serialize(&sbuf, nullptr);
      sbuf.setMode(ISerializer::ModeGet);
      sbuf.setReadMode(ISerializer::ReadAdd);
      for (Int32 i = 0; i < nb_rank_to_read; ++i)
        full_written_data->serialize(&sbuf, nullptr);
    }
    if (data != full_written_data) {
      info(5) << "PARALLEL READ";
      parallel_data_reader->getSortedValues(full_written_data, data);
    }
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Ref<ParallelDataReader> BasicReader::
_getReader(VariableMetaData* varmd)
{
  Int32 nb_to_read = m_nb_rank_to_read;

  // Pour la lecture, lors d'une reprise, le groupe (var->itemGroup())
  // associé à la variable ainsi que la famille (var->itemFamily())
  // n'existe pas encore. Il ne faut donc pas l'utiliser
  const String& var_group_name = varmd->itemGroupName();
  String group_full_name = varmd->meshName() + "_" + varmd->itemFamilyName() + "_" + var_group_name;
  auto ix = m_parallel_data_readers.find(group_full_name);
  if (ix != m_parallel_data_readers.end())
    return ix->second;

  IParallelMng* pm = m_parallel_mng;
  Ref<ParallelDataReader> reader = makeRef(new ParallelDataReader(pm));
  {
    UniqueArray<SharedArray<Int64>> written_unique_ids(nb_to_read);
    Int64Array& wanted_unique_ids = reader->wantedUniqueIds();
    for (Integer i = 0; i < nb_to_read; ++i) {
      m_global_readers[i]->readItemGroup(group_full_name, written_unique_ids[i], wanted_unique_ids);
    }
    // Cela ne doit être actif que pour les comparaisons (pas en reprise car les groupes
    // ne sont pas valides)
    if (m_item_group_finder) {
      ItemGroup ig = m_item_group_finder->getWantedGroup(varmd);
      _fillUniqueIds(ig, wanted_unique_ids);
    }
    for (Integer i = 0; i < nb_to_read; ++i) {
      Integer nb_uid = written_unique_ids[i].size();
      if (nb_uid >= 1)
        info(5) << "PART I=" << i
                << " nb_uid=" << nb_uid
                << " min_uid=" << written_unique_ids[i][0]
                << " max_uid=" << written_unique_ids[i][nb_uid - 1];
    }
    Int64Array& full_written_unique_ids = reader->writtenUniqueIds();
    for (Integer i = 0; i < nb_to_read; ++i)
      full_written_unique_ids.addRange(written_unique_ids[i]);
    info(5) << "FULL UID SIZE=" << full_written_unique_ids.size();
    if (m_want_parallel)
      reader->sort();
  }
  m_parallel_data_readers.insert(std::make_pair(group_full_name, reader));
  return reader;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Remplit l'argument avec des couples (nom_de_variable,valeur du hash).
 *
 * Cela n'est valide que pour le rang 0
 */
void BasicReader::
fillComparisonHash(std::map<String, String>& comparison_hash_map)
{
  comparison_hash_map.clear();
  if (m_nb_rank_to_read==0)
    return;
  if (m_parallel_mng->commRank()!=m_parallel_mng->masterIORank())
    return;
  const VariableDataInfoMap& var_map = m_global_readers[0]->variablesDataInfoMap();
  for( auto v : var_map){
    VariableDataInfo* vd = v.second.get();
    comparison_hash_map.try_emplace(vd->fullName(),vd->comparisonHashValue());
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void BasicReader::
read(IVariable* var, IData* data)
{
  info(4) << "MASTER READ var=" << var->fullName() << " data=" << data;
  if (var->isPartial()) {
    info() << "** WARNING: partial variable not implemented in BasicReaderWriter";
    return;
  }
  ScopedPtrT<VariableMetaData> vmd(var->createMetaData());
  _directReadVal(vmd.get(), data);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void BasicReader::
read(const VariableDataReadInfo& infos)
{
  VariableMetaData* vmd = infos.variableMetaData();
  IData* data = infos.data();
  info(4) << "MASTER2 READ var=" << vmd->fullName() << " data=" << data;
  if (vmd->isPartial()) {
    info() << "** WARNING: partial variable not implemented in BasicReaderWriter";
    return;
  }
  _directReadVal(vmd, data);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

String BasicReader::
metaData()
{
  ByteUniqueArray bytes;
  fillMetaData(bytes);
  String s(bytes);
  info(5) << " S=" << s << '\n';
  return s;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void BasicReader::
fillMetaData(ByteArray& bytes)
{
  Int32 rank = m_parallel_mng->commRank();
  if (m_forced_rank_to_read >= 0)
    rank = m_forced_rank_to_read;
  if (m_version >= 3) {
    Int64 meta_data_size = 0;
    String key_name = "Global:CheckpointMetadata";
    info(4) << "Reading checkpoint metadata from database";
    m_forced_rank_to_read_text_reader->getExtents(key_name, Int64ArrayView(1, &meta_data_size));
    bytes.resize(meta_data_size);
    m_forced_rank_to_read_text_reader->read(key_name, asWritableBytes(bytes.span()));
  }
  else {
    String filename = _getMetaDataFileName(rank);
    info(4) << "Reading checkpoint metadata file=" << filename;
    platform::readAllFile(filename, false, bytes);
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void BasicReader::
_setRanksToRead()
{
  IParallelMng* pm = m_parallel_mng;
  Int32 my_rank = pm->commRank();
  Int32 nb_rank = pm->commSize();
  Int32 first_to_read = my_rank;
  Int32 nb_to_read = 1;

  if (m_forced_rank_to_read >= 0) {
    m_first_rank_to_read = m_forced_rank_to_read;
    m_nb_rank_to_read = 1;
    return;
  }
  if (nb_rank == 1) {
    nb_to_read = m_nb_written_part;
  }
  else if (nb_rank < m_nb_written_part) {
    // Il y a plus de fichiers que de processeurs.
    // Un des processeurs doit donc lire au moins deux fichiers.
    // Pour ceux dont c'est le cas, il faut que ces fichiers
    // soient consécutifs pour que l'intervalle des uniqueId()
    // des entités du processeur soient consécutifs
    Int32 nb_part_per_rank = m_nb_written_part / nb_rank;
    Int32 remaining_nb_part = m_nb_written_part - (nb_part_per_rank * nb_rank);
    first_to_read = nb_part_per_rank * my_rank;
    nb_to_read = nb_part_per_rank;
    info(4) << "NB_PART_PER_RANK = " << nb_part_per_rank
            << " REMAINING=" << remaining_nb_part;
    if (my_rank >= remaining_nb_part) {
      first_to_read += remaining_nb_part;
    }
    else {
      first_to_read += my_rank;
      ++nb_to_read;
    }
  }
  else {
    if (my_rank >= m_nb_written_part) {
      // Aucune partie à lire
      nb_to_read = 0;
    }
  }

  m_first_rank_to_read = first_to_read;
  m_nb_rank_to_read = nb_to_read;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Ref<IGenericReader> BasicReader::
_readOwnMetaDataAndCreateReader(Int32 rank)
{
  String main_filename = _getBasicVariableFile(m_version, m_path, rank);
  Ref<KeyValueTextReader> text_reader;
  if (m_version >= 3) {
    // Si le rang est le même que m_forced_rank_to_read, alors on peut réutiliser
    // le lecteur déjà créé.
    if (rank == m_forced_rank_to_read)
      text_reader = m_forced_rank_to_read_text_reader;
    else {
      text_reader = makeRef(new KeyValueTextReader(traceMng(), main_filename, m_version));
      // Il faut que ce lecteur ait le même gestionnaire de compression
      // que celui déjà créé
      text_reader->setDataCompressor(m_forced_rank_to_read_text_reader->dataCompressor());
      text_reader->setHashAlgorithm(m_forced_rank_to_read_text_reader->hashAlgorithm());
    }
  }

  auto r = makeRef<IGenericReader>(new BasicGenericReader(m_application, m_version, text_reader));
  r->initialize(m_path, rank);
  return r;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void BasicReader::
beginRead(const VariableCollection& vars)
{
  ARCANE_UNUSED(vars);
  beginRead(DataReaderInfo());
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void BasicReader::
beginRead(const DataReaderInfo& infos)
{
  ARCANE_UNUSED(infos);
  info(4) << "** ** BEGIN READ";

  _setRanksToRead();
  info(4) << "RanksToRead: FIRST TO READ =" << m_first_rank_to_read
          << " nb=" << m_nb_rank_to_read << " version=" << m_version;
  m_global_readers.resize(m_nb_rank_to_read);
  for (Integer i = 0; i < m_nb_rank_to_read; ++i)
    m_global_readers[i] = _readOwnMetaDataAndCreateReader(i + m_first_rank_to_read);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::impl

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
