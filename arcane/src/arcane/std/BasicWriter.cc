// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2023 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* BasicWriter.cc                                              (C) 2000-2023 */
/*                                                                           */
/* Ecriture simple pour les protections/reprises.                            */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/std/internal/BasicReaderWriter.h"

#include "arcane/utils/FatalErrorException.h"
#include "arcane/utils/StringBuilder.h"
#include "arcane/utils/PlatformUtils.h"
#include "arcane/utils/JSONWriter.h"
#include "arcane/utils/IDataCompressor.h"
#include "arcane/utils/IHashAlgorithm.h"

#include "arcane/core/IXmlDocumentHolder.h"
#include "arcane/core/IParallelMng.h"
#include "arcane/core/IIOMng.h"
#include "arcane/core/IApplication.h"
#include "arcane/core/ISerializedData.h"
#include "arcane/core/ItemGroup.h"
#include "arcane/core/IRessourceMng.h"
#include "arcane/core/ServiceBuilder.h"
#include "arcane/core/IVariable.h"
#include "arcane/core/IItemFamily.h"
#include "arcane/core/IData.h"

#include "arcane/std/ParallelDataWriter.h"
#include "arcane/std/TextWriter.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::impl
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

BasicGenericWriter::
BasicGenericWriter(IApplication* app, Int32 version,
                   Ref<KeyValueTextWriter> text_writer)
: TraceAccessor(app->traceMng())
, m_application(app)
, m_version(version)
, m_rank(A_NULL_RANK)
, m_text_writer(text_writer)
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

BasicGenericWriter::
~BasicGenericWriter()
{
  for (const auto& x : m_variables_data_info) {
    delete x.second;
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void BasicGenericWriter::
initialize(const String& path, Int32 rank)
{
  if (!m_text_writer)
    ARCANE_FATAL("Null text writer");
  m_path = path;
  m_rank = rank;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void BasicGenericWriter::
writeData(const String& var_full_name, const ISerializedData* sdata)
{
  //TODO: Verifier que initialize() a bien été appelé.
  auto var_data_info = new VariableDataInfo(var_full_name, sdata);
  KeyValueTextWriter* writer = m_text_writer.get();
  var_data_info->setFileOffset(writer->fileOffset());
  m_variables_data_info.insert(std::make_pair(var_full_name, var_data_info));
  info(4) << " SDATA name=" << var_full_name << " nb_element=" << sdata->nbElement()
          << " dim=" << sdata->nbDimension() << " datatype=" << sdata->baseDataType()
          << " nb_basic_element=" << sdata->nbBaseElement()
          << " is_multi=" << sdata->isMultiSize()
          << " dimensions_size=" << sdata->extents().size()
          << " memory_size=" << sdata->memorySize()
          << " bytes_size=" << sdata->constBytes().size();

  const void* ptr = sdata->constBytes().data();

  // Si la variable est de type tableau à deux dimensions, sauve les
  // tailles de la deuxième dimension par élément.
  Int64ConstArrayView extents = sdata->extents();
  writer->setExtents(var_full_name, extents);

  // Maintenant, sauve les valeurs si necessaire
  Int64 nb_base_element = sdata->nbBaseElement();
  if (nb_base_element != 0 && ptr) {
    writer->write(var_full_name, asBytes(sdata->constBytes()));
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void BasicGenericWriter::
writeItemGroup(const String& group_full_name, SmallSpan<const Int64> written_unique_ids,
               SmallSpan<const Int64> wanted_unique_ids)
{
  if (m_version >= 3) {
    // Sauve les informations du groupe la base de données (clé,valeur)
    {
      String written_uid_name = String("GroupWrittenUid:") + group_full_name;
      Int64 nb_written_uid = written_unique_ids.size();
      m_text_writer->setExtents(written_uid_name, Int64ConstArrayView(1, &nb_written_uid));
      m_text_writer->write(written_uid_name, asBytes(written_unique_ids));
    }
    {
      String wanted_uid_name = String("GroupWantedUid:") + group_full_name;
      Int64 nb_wanted_uid = wanted_unique_ids.size();
      m_text_writer->setExtents(wanted_uid_name, Int64ConstArrayView(1, &nb_wanted_uid));
      m_text_writer->write(wanted_uid_name, asBytes(wanted_unique_ids));
    }
    return;
  }

  String filename = BasicReaderWriterCommon::_getBasicGroupFile(m_path, group_full_name, m_rank);
  TextWriter writer(filename);

  // Sauve la liste des unique_ids écrits
  {
    Integer nb_unique_id = written_unique_ids.size();
    writer.write(asBytes(Span<const Int32>(&nb_unique_id, 1)));
    writer.write(asBytes(written_unique_ids));
  }

  // Sauve la liste des unique_ids souhaités par ce sous-domaine
  {
    Integer nb_unique_id = wanted_unique_ids.size();
    writer.write(asBytes(Span<const Int32>(&nb_unique_id, 1)));
    writer.write(asBytes(wanted_unique_ids));
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void BasicGenericWriter::
endWrite()
{
  IApplication* app = m_application;
  ScopedPtrT<IXmlDocumentHolder> xdoc(app->ressourceMng()->createXmlDocument());
  XmlNode doc = xdoc->documentNode();
  XmlElement root(doc, "variables-data");
  IDataCompressor* dc = m_text_writer->dataCompressor().get();
  if (dc) {
    root.setAttrValue("deflater-service", dc->name());
    root.setAttrValue("min-compress-size", String::fromNumber(dc->minCompressSize()));
  }
  root.setAttrValue("version", String::fromNumber(m_version));
  for (const auto& i : m_variables_data_info) {
    VariableDataInfo* vdi = i.second;
    XmlNode e = root.createAndAppendElement("variable-data");
    e.setAttrValue("full-name", vdi->fullName());
    vdi->write(e);
  }
  if (m_version >= 3) {
    // Sauve les méta-données dans la base de données.
    UniqueArray<Byte> bytes;
    xdoc->save(bytes);
    Int64 length = bytes.length();
    String key_name = "Global:OwnMetadata";
    m_text_writer->setExtents(key_name, Int64ConstArrayView(1, &length));
    m_text_writer->write(key_name, asBytes(bytes.span()));
  }
  else {
    String filename = BasicReaderWriterCommon::_getOwnMetatadaFile(m_path, m_rank);
    app->ioMng()->writeXmlFile(xdoc.get(), filename);
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

BasicWriter::
BasicWriter(IApplication* app, IParallelMng* pm, const String& path,
            eOpenMode open_mode, Integer version, bool want_parallel)
: BasicReaderWriterCommon(app, pm, path, open_mode)
, m_want_parallel(want_parallel)
, m_is_gather(false)
, m_version(version)
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

BasicWriter::
~BasicWriter()
{
  for (const auto& i : m_parallel_data_writers)
    delete i.second;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void BasicWriter::
initialize()
{
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

  m_global_writer = new BasicGenericWriter(m_application, m_version, m_text_writer);
  if (m_verbose_level > 0)
    info() << "** OPEN MODE = " << m_open_mode;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ParallelDataWriter* BasicWriter::
_getWriter(IVariable* var)
{
  ItemGroup group = var->itemGroup();
  auto i = m_parallel_data_writers.find(group);
  if (i != m_parallel_data_writers.end())
    return i->second;
  ParallelDataWriter* writer = new ParallelDataWriter(m_parallel_mng);
  writer->setGatherAll(m_is_gather);
  {
    Int64UniqueArray items_uid;
    ItemGroup own_group = var->itemGroup().own();
    _fillUniqueIds(own_group, items_uid);
    Int32ConstArrayView local_ids = own_group.internal()->itemsLocalId();
    writer->sort(local_ids, items_uid);
  }
  m_parallel_data_writers.insert(std::make_pair(group, writer));
  return writer;
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
  if (var->itemKind() != IK_Unknown) {
    ItemGroup group = var->itemGroup();
    if (m_want_parallel) {
      ParallelDataWriter* writer = _getWriter(var);
      written_unique_ids = writer->sortedUniqueIds();
      allocated_write_data = writer->getSortedValues(data);
      write_data = allocated_write_data.get();
    }
    else {
      //TODO il faut trier les uniqueId
      _fillUniqueIds(group, sequential_written_unique_ids);
      written_unique_ids = sequential_written_unique_ids.view();
    }
    _fillUniqueIds(group, wanted_unique_ids);
    if (m_written_groups.find(group) == m_written_groups.end()) {
      info(5) << "WRITE GROUP " << group.name();
      IItemFamily* item_family = group.itemFamily();
      String gname = group.name();
      String group_full_name = item_family->fullName() + "_" + gname;
      m_global_writer->writeItemGroup(group_full_name, written_unique_ids, wanted_unique_ids.view());
      m_written_groups.insert(group);
    }
  }
  Ref<ISerializedData> sdata(write_data->createSerializedDataRef(false));
  m_global_writer->writeData(var->fullName(), sdata.get());
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
    std::ofstream ofile(filename.localstr());
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
endWrite()
{
  IParallelMng* pm = m_parallel_mng;
  if (pm->isMasterIO()) {
    Int64 nb_part = pm->commSize();
    if (m_version >= 3) {
      // Sauvegarde les informations au format JSON
      JSONWriter jsw;
      {
        JSONWriter::Object main_object(jsw);
        jsw.writeKey(_getArcaneDBTag());
        {
          JSONWriter::Object db_object(jsw);
          jsw.write("Version", (Int64)m_version);
          jsw.write("NbPart", nb_part);

          String data_compressor_name;
          Int64 data_compressor_min_size = 0;
          if (m_data_compressor.get()) {
            data_compressor_name = m_data_compressor->name();
            data_compressor_min_size = m_data_compressor->minCompressSize();
          }
          jsw.write("DataCompressor", data_compressor_name);
          jsw.write("DataCompressorMinSize", String::fromNumber(data_compressor_min_size));

          String hash_algorithm_name;
          if (m_hash_algorithm.get())
            hash_algorithm_name = m_hash_algorithm->name();
          jsw.write("HashAlgorithm", hash_algorithm_name);
        }
      }
      StringBuilder filename = m_path;
      filename += "/arcane_acr_db.json";
      String fn = filename.toString();
      std::ofstream ofile(fn.localstr());
      ofile << jsw.getBuffer();
    }
    else {
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
