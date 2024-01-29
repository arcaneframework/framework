// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2024 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* BasicGenericReader.cc                                       (C) 2000-2024 */
/*                                                                           */
/* Lecture simple pour les protections/reprises.                             */
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

#include "arcane/std/TextReader.h"

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
  // Dans le cas de la version 1 ou 2, on ne peut pas créer le KeyValueTextReader
  // avant de lire les 'OwnMetaData' car ce sont ces dernières qui contiennent
  // le numéro de version.

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

    // Si on connait déjà la version et qu'elle est supérieure ou égale à 3
    // alors les informations sont dans la base de données. Dans ce cas on lit
    // directement les infos depuis cette base.
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
    // - taille des dimensions sur 32 bits
    m_version = 1;
  else if (version_id == "2")
    // Version 2:
    // - taille des dimensions sur 64 bits
    m_version = 2;
  else if (version_id == "3")
    // Version 3:
    // - taille des dimensions sur 64 bits
    // - 1 seul fichier pour toutes les meta-données
    m_version = 3;
  else
    ARCANE_FATAL("Unsupported version '{0}' (max=3)", version_id);

  Ref<IDataCompressor> deflater;
  if (!deflater_name.null())
    deflater = BasicReaderWriterCommon::_createDeflater(m_application, deflater_name);

  Ref<IHashAlgorithm> hash_algorithm;
  if (!hash_algorithm_name.null())
    hash_algorithm = BasicReaderWriterCommon::_createHashAlgorithm(m_application, hash_algorithm_name);

  // Si disponible, essaie de relire les informations des variables au format JSON
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

  // Il est possible qu'il y ait déjà un algorithme de compression spécifié.
  // Il ne faut pas l'écraser si aucun n'est spécifié dans 'OwnMetadata'.
  // (Normalement cela ne devrait pas arriver sauf incohérence).
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
  TextReader reader(filename);

  {
    Integer nb_unique_id = 0;
    reader.readIntegers(IntegerArrayView(1, &nb_unique_id));
    info(5) << "NB_WRITTEN_UNIQUE_ID = " << nb_unique_id;
    written_unique_ids.resize(nb_unique_id);
    reader.read(asWritableBytes(written_unique_ids.span()));
  }

  {
    Integer nb_unique_id = 0;
    reader.readIntegers(IntegerArrayView(1, &nb_unique_id));
    info(5) << "NB_WANTED_UNIQUE_ID = " << nb_unique_id;
    wanted_unique_ids.resize(nb_unique_id);
    reader.read(asWritableBytes(wanted_unique_ids.span()));
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
