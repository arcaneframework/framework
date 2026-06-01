// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* BasicGenericReader.cc                                       (C) 2000-2024 */
/*                                                                           */
/* Simple reading for protections/recoveries.                                */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/std/internal/BasicReader.h"

#include "arcane/utils/FatalErrorException.h"
#include "arcane/utils/StringBuilder.h"
#include "arcane/utils/IDataCompressor.h"
#include "arcane/utils/JSONReader.h"
#include "arcane/utils/Ref.h"

#include "arcane/core/IApplication.h"
#include "arcane/core/IXmlDocumentHolder.h"
#include "arcane/core/IIOMng.h"
#include "arcane/core/IData.h"
#include "arcane/core/ArcaneException.h"
#include "arcane/core/ISerializedData.h"
#include "arcane/core/XmlNodeList.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::impl
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

BasicGenericReader::
BasicGenericReader(IApplication* app, Int32 version, Ref<KeyValueTextReader> text_reader)
: TraceAccessor(app->traceMng())
, m_application(app)
, m_text_reader(text_reader)
, m_version(version)
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void BasicGenericReader::
initialize(const String& path, Int32 rank)
{
  // In the case of version 1 or 2, we cannot create the KeyValueTextReader
  // before reading 'OwnMetaData' because they contain
  // the version number.

  m_path = path;
  m_rank = rank;

  info(4) << "BasicGenericReader::initialize known_version=" << m_version;

  ScopedPtrT<IXmlDocumentHolder> xdoc;

  if (m_version >= 3) {
    if (!m_text_reader.get())
      ARCANE_FATAL("Null text reader");
    String dc_name;
    if (m_text_reader->dataCompressor().get())
      dc_name = m_text_reader->dataCompressor()->name();
    info(4) << "BasicGenericReader::initialize data_compressor=" << dc_name;

    // If we already know the version and it is greater than or equal to 3
    // then the information is in the database. In this case we read
    // the info directly from this database.
    String main_filename = BasicReaderWriterCommon::_getBasicVariableFile(m_version, m_path, rank);
    Int64 meta_data_size = 0;
    String key_name = "Global:OwnMetadata";
    m_text_reader->getExtents(key_name, Int64ArrayView(1, &meta_data_size));
    UniqueArray<std::byte> bytes(meta_data_size);
    m_text_reader->read(key_name, bytes);
    info(4) << "Reading own metadata rank=" << rank << " from database";
    xdoc = IXmlDocumentHolder::loadFromBuffer(bytes, "OwnMetadata", traceMng());
  }
  else {
    StringBuilder filename = BasicReaderWriterCommon::_getOwnMetatadaFile(m_path, m_rank);
    info(4) << "Reading own metadata rank=" << rank << " file=" << filename;
    IApplication* app = m_application;
    xdoc = app->ioMng()->parseXmlFile(filename);
  }
  XmlNode root = xdoc->documentNode().documentElement();
  XmlNodeList variables_elem = root.children("variable-data");
  String deflater_name = root.attrValue("deflater-service");
  String hash_algorithm_name = root.attrValue("hash-algorithm-service");
  String version_id = root.attrValue("version", false);
  info(4) << "Infos from metadata deflater-service=" << deflater_name
          << " hash-algorithm-service=" << hash_algorithm_name
          << " version=" << version_id;
  if (version_id.null() || version_id == "1")
    // Version 1:
    // - dimension size in 32 bits
    m_version = 1;
  else if (version_id == "2")
    // Version 2:
    // - dimension size in 64 bits
    m_version = 2;
  else if (version_id == "3")
    // Version 3:
    // - dimension size in 64 bits
    // - only 1 file for all metadata
    m_version = 3;
  else
    ARCANE_FATAL("Unsupported version '{0}' (max=3)", version_id);

  Ref<IDataCompressor> deflater;
  if (!deflater_name.null())
    deflater = BasicReaderWriterCommon::_createDeflater(m_application, deflater_name);

  Ref<IHashAlgorithm> hash_algorithm;
  if (!hash_algorithm_name.null())
    hash_algorithm = BasicReaderWriterCommon::_createHashAlgorithm(m_application, hash_algorithm_name);

  // If available, try to reread the variable information in JSON format
  bool do_json = true;
  String json_variables_elem = root.child("variables-data-json").value();
  if (do_json && !json_variables_elem.empty()) {
    JSONDocument json_doc;
    json_doc.parse(json_variables_elem.bytes(), "Internal variables data");
    JSONValue json_root = json_doc.root();
    JSONValue json_vars = json_root.expectedChild("Variables");
    for (JSONKeyValue kv : json_vars.keyValueChildren()) {
      String var_full_name = kv.name();
      m_variables_data_info.add(var_full_name, kv.value());
    }
  }
  else {
    for (const XmlNode& n : variables_elem) {
      String var_full_name = n.attrValue("full-name");
      m_variables_data_info.add(var_full_name, n);
    }
  }

  if (!m_text_reader.get()) {
    String main_filename = BasicReaderWriterCommon::_getBasicVariableFile(m_version, m_path, rank);
    m_text_reader = makeRef(new KeyValueTextReader(traceMng(), main_filename, m_version));
  }

  // There might already be a compression algorithm specified.
  // It should not be overwritten if none is specified in 'OwnMetadata'.
  // (Normally this should not happen unless there is an inconsistency).
  if (deflater.get())
    m_text_reader->setDataCompressor(deflater);
  if (hash_algorithm.get())
    m_text_reader->setHashAlgorithm(hash_algorithm);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Ref<VariableDataInfo> BasicGenericReader::
_getVarInfo(const String& full_name)
{
  Ref<VariableDataInfo> vdi = m_variables_data_info.find(full_name);
  if (!vdi.get())
    ARCANE_THROW(ReaderWriterException,
                 "Can not find own metadata infos for data var={0} rank={1}", full_name, m_rank);
  return vdi;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void BasicGenericReader::
readData(const String& var_full_name, IData* data)
{
  KeyValueTextReader* reader = m_text_reader.get();
  String vname = var_full_name;
  Ref<VariableDataInfo> vdi = _getVarInfo(vname);
  if (m_version < 3)
    reader->setFileOffset(vdi->fileOffset());

  eDataType data_type = vdi->baseDataType();
  Int64 memory_size = vdi->memorySize();
  Integer dimension_array_size = vdi->dimensionArraySize();
  Int64 nb_element = vdi->nbElement();
  Integer nb_dimension = vdi->nbDimension();
  Int64 nb_base_element = vdi->nbBaseElement();
  bool is_multi_size = vdi->isMultiSize();
  UniqueArray<Int64> extents(dimension_array_size);
  reader->getExtents(var_full_name, extents.view());
  ArrayShape shape = vdi->shape();

  Ref<ISerializedData> sd(arcaneCreateSerializedDataRef(data_type, memory_size, nb_dimension, nb_element,
                                                        nb_base_element, is_multi_size, extents, shape));

  Int64 storage_size = sd->memorySize();
  info(4) << " READ DATA storage_size=" << storage_size << " DATA=" << data;

  data->allocateBufferForSerializedData(sd.get());

  if (storage_size != 0)
    reader->read(var_full_name, asWritableBytes(sd->writableBytes()));

  data->assignSerializedData(sd.get());
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void BasicGenericReader::
readItemGroup(const String& group_full_name, Int64Array& written_unique_ids,
              Int64Array& wanted_unique_ids)
{
  if (m_version >= 3) {
    {
      String written_uid_name = String("GroupWrittenUid:") + group_full_name;
      Int64 nb_written_uid = 0;
      m_text_reader->getExtents(written_uid_name, Int64ArrayView(1, &nb_written_uid));
      written_unique_ids.resize(nb_written_uid);
      m_text_reader->read(written_uid_name, asWritableBytes(written_unique_ids.span()));
    }
    {
      String wanted_uid_name = String("GroupWantedUid:") + group_full_name;
      Int64 nb_wanted_uid = 0;
      m_text_reader->getExtents(wanted_uid_name, Int64ArrayView(1, &nb_wanted_uid));
      wanted_unique_ids.resize(nb_wanted_uid);
      m_text_reader->read(wanted_uid_name, asWritableBytes(wanted_unique_ids.span()));
    }
    return;
  }

  info(5) << "READ GROUP " << group_full_name;
  String filename = BasicReaderWriterCommon::_getBasicGroupFile(m_path, group_full_name, m_rank);
  std::ifstream reader(filename.localstr(), std::ios::in | std::ios::binary);

  {
    Integer nb_unique_id = 0;
    binaryRead(reader, asWritableBytes(SmallSpan<Integer>(&nb_unique_id, 1)));
    info(5) << "NB_WRITTEN_UNIQUE_ID = " << nb_unique_id;
    written_unique_ids.resize(nb_unique_id);
    binaryRead(reader, asWritableBytes(written_unique_ids.span()));
  }

  {
    Integer nb_unique_id = 0;
    binaryRead(reader, asWritableBytes(SmallSpan<Integer>(&nb_unique_id, 1)));
    info(5) << "NB_WANTED_UNIQUE_ID = " << nb_unique_id;
    wanted_unique_ids.resize(nb_unique_id);
    binaryRead(reader, asWritableBytes(wanted_unique_ids.span()));
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

String BasicGenericReader::
comparisonHashValue(const String& var_full_name) const
{
  Ref<VariableDataInfo> vdi = m_variables_data_info.find(var_full_name);
  if (vdi.get())
    return vdi->comparisonHashValue();
  return {};
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::impl

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
