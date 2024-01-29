// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2024 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* BasicGenericWriter.cc                                       (C) 2000-2024 */
/*                                                                           */
/* Ecriture simple pour les protections/reprises.                            */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/std/internal/BasicWriter.h"

#include "arcane/utils/FatalErrorException.h"
#include "arcane/utils/IDataCompressor.h"
#include "arcane/utils/Ref.h"

#include "arcane/core/IXmlDocumentHolder.h"
#include "arcane/core/IParallelMng.h"
#include "arcane/core/IIOMng.h"
#include "arcane/core/IApplication.h"
#include "arcane/core/ISerializedData.h"
#include "arcane/core/IRessourceMng.h"

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
, m_text_writer(text_writer)
{
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
  auto var_data_info = m_variables_data_info.add(var_full_name, sdata);
  KeyValueTextWriter* writer = m_text_writer.get();
  var_data_info->setFileOffset(writer->fileOffset());
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
  const IDataCompressor* dc = m_text_writer->dataCompressor().get();
  if (dc) {
    root.setAttrValue("deflater-service", dc->name());
    root.setAttrValue("min-compress-size", String::fromNumber(dc->minCompressSize()));
  }
  root.setAttrValue("version", String::fromNumber(m_version));
  for (const auto& i : m_variables_data_info) {
    Ref<VariableDataInfo> vdi = i.second;
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

} // namespace Arcane::impl

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
