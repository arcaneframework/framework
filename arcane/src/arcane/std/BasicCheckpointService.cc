// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2024 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* BasicCheckpointService.cc                                   (C) 2000-2024 */
/*                                                                           */
/* Service basique de protection/reprise.                                    */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/std/internal/BasicReader.h"
#include "arcane/std/internal/BasicWriter.h"

#include "arcane/utils/StringBuilder.h"
#include "arcane/utils/OStringStream.h"
#include "arcane/utils/PlatformUtils.h"

#include "arcane/core/IXmlDocumentHolder.h"
#include "arcane/core/IParallelMng.h"
#include "arcane/core/CheckpointService.h"
#include "arcane/core/Directory.h"
#include "arcane/core/IParallelReplication.h"
#include "arcane/core/IVariableUtilities.h"
#include "arcane/core/VerifierService.h"
#include "arcane/core/IVariableMng.h"
#include "arcane/core/CheckpointInfo.h"

#include "arcane/std/ArcaneBasicCheckpoint_axl.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{
using namespace Arcane::impl;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Protection/reprise basique (version 1).
 */
class ArcaneBasicCheckpointService
: public ArcaneArcaneBasicCheckpointObject
{
 public:

  struct MetaData
  {
    int m_version = -1;
    static MetaData parse(const String& meta_data, ITraceMng* tm)
    {
      auto doc_ptr = IXmlDocumentHolder::loadFromBuffer(meta_data.bytes(), "MetaData", tm);
      ScopedPtrT<IXmlDocumentHolder> xml_doc(doc_ptr);
      XmlNode root = xml_doc->documentNode().documentElement();
      Integer version = root.attr("version").valueAsInteger();
      if (version != 1)
        ARCANE_THROW(ReaderWriterException, "Bad checkpoint metadata version '{0}' (expected 1)", version);
      MetaData md;
      md.m_version = version;
      return md;
    }
  };

 public:

  explicit ArcaneBasicCheckpointService(const ServiceBuildInfo& sbi)
  : ArcaneArcaneBasicCheckpointObject(sbi)
  , m_write_index(0)
  , m_writer(nullptr)
  , m_reader(nullptr)
  {}
  IDataWriter* dataWriter() override { return m_writer; }
  IDataReader* dataReader() override { return m_reader; }

  void notifyBeginWrite() override;
  void notifyEndWrite() override;
  void notifyBeginRead() override;
  void notifyEndRead() override;
  void close() override {}
  String readerServiceName() const override { return "ArcaneBasicCheckpointReader"; }

 private:

  Integer m_write_index;
  BasicWriter* m_writer;
  BasicReader* m_reader;

 private:

  String _defaultFileName()
  {
    info() << "USE DEFAULT FILE NAME index=" << currentIndex();
    IParallelReplication* pr = subDomain()->parallelMng()->replication();

    String buf = "arcanedump";
    if (pr->hasReplication()) {
      buf = buf + "_r";
      buf = buf + pr->replicationRank();
    }
    info() << "FILE_NAME is " << buf;
    return buf;
  }
  Directory _defaultDirectory()
  {
    return Directory(baseDirectoryName());
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Protection/reprise basique (version 2).
 *
 * Idem ArcaneBasicCheckpointService sauf qu'on utilise le
 * service ArcaneBasic2CheckpointReader pour la relecture.
 */
class ArcaneBasic2CheckpointService
: public ArcaneBasicCheckpointService
{
 public:

  explicit ArcaneBasic2CheckpointService(const ServiceBuildInfo& sbi)
  : ArcaneBasicCheckpointService(sbi)
  {}
  String readerServiceName() const override { return "ArcaneBasic2CheckpointReader"; }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ArcaneBasicCheckpointService::
notifyBeginRead()
{
  String meta_data_str = readerMetaData();
  MetaData md = MetaData::parse(meta_data_str, traceMng());

  info() << " GET META DATA READER " << readerMetaData()
         << " version=" << md.m_version
         << " filename=" << fileName();

  String filename = fileName();
  if (filename.null()) {
    Directory dump_dir(_defaultDirectory());
    filename = dump_dir.file(_defaultFileName());
    setFileName(filename);
  }
  filename = filename + "_n" + currentIndex();
  info() << " READ CHECKPOINT FILENAME = " << filename;
  IParallelMng* pm = subDomain()->parallelMng();
  IApplication* app = subDomain()->application();
  bool want_parallel = false;
  m_reader = new BasicReader(app, pm, A_NULL_RANK, filename, want_parallel);
  m_reader->initialize();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ArcaneBasicCheckpointService::
notifyEndRead()
{
  delete m_reader;
  m_reader = nullptr;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ArcaneBasicCheckpointService::
notifyBeginWrite()
{
  auto open_mode = BasicReaderWriterCommon::OpenModeAppend;
  Integer write_index = checkpointTimes().size();
  --write_index;
  if (write_index == 0)
    open_mode = BasicReaderWriterCommon::OpenModeTruncate;

  String filename = fileName();
  if (filename.null()) {
    Directory dump_dir(_defaultDirectory());
    filename = dump_dir.file(_defaultFileName());
    setFileName(filename);
  }
  filename = filename + "_n" + write_index;

  Int32 version = 2;
  Ref<IDataCompressor> data_compressor;
  if (options()) {
    version = options()->formatVersion();
    // N'utilise la compression qu'à partir de la version 3 car cela est
    // incompatible avec les anciennes versions
    if (version >= 3) {
      data_compressor = options()->dataCompressor.instanceRef();
    }
  }

  info() << "Writing checkpoint with 'ArcaneBasicCheckpointService'"
         << " version=" << version
         << " filename='" << filename << "'\n";

  platform::recursiveCreateDirectory(filename);

  IParallelMng* pm = subDomain()->parallelMng();
  IApplication* app = subDomain()->application();
  bool want_parallel = pm->isParallel();
  want_parallel = false;
  m_writer = new BasicWriter(app, pm, filename, open_mode, version, want_parallel);
  m_writer->setDataCompressor(data_compressor);
  m_writer->initialize();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ArcaneBasicCheckpointService::
notifyEndWrite()
{
  OStringStream ostr;
  ostr() << "<infos";
  const int meta_data_version = 1;
  ostr() << " version='" << meta_data_version << "'";
  ostr() << "/>\n";
  setReaderMetaData(ostr.str());
  ++m_write_index;
  delete m_writer;
  m_writer = nullptr;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Relecture de protection (version 2).
 */
class ArcaneBasic2CheckpointReaderService
: public AbstractService
, public ICheckpointReader2
{
 public:

  explicit ArcaneBasic2CheckpointReaderService(const ServiceBuildInfo& sbi)
  : AbstractService(sbi)
  , m_application(sbi.application())
  , m_reader(nullptr)
  {}
  IDataReader2* dataReader() override { return m_reader; }

  void notifyBeginRead(const CheckpointReadInfo& cri) override;
  void notifyEndRead() override;

 private:

  IApplication* m_application;
  BasicReader* m_reader;

 private:

  String _defaultFileName(const CheckpointInfo& ci)
  {
    Integer index = ci.checkpointIndex();
    bool has_replication = ci.nbReplication() > 1;
    Int32 replication_rank = ci.replicationRank();
    info() << "USE DEFAULT FILE NAME index=" << index;

    String buf = "arcanedump";
    if (has_replication) {
      buf = buf + "_r";
      buf = buf + replication_rank;
    }
    info() << "FILE_NAME is " << buf;
    return buf;
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ArcaneBasic2CheckpointReaderService::
notifyBeginRead(const CheckpointReadInfo& cri)
{
  const CheckpointInfo& ci = cri.checkpointInfo();
  String reader_meta_data_str = ci.readerMetaData();
  auto md = ArcaneBasicCheckpointService::MetaData::parse(reader_meta_data_str, traceMng());

  info() << "Basic2CheckpointReader GET META DATA READER " << reader_meta_data_str
         << " version=" << md.m_version;

  Directory dump_dir(ci.directory());
  String filename = dump_dir.file(_defaultFileName(ci));
  filename = filename + "_n" + ci.checkpointIndex();
  ;
  info() << " READ CHECKPOINT FILENAME = " << filename;
  IParallelMng* pm = cri.parallelMng();
  bool want_parallel = false;
  m_reader = new BasicReader(m_application, pm, ci.subDomainRank(), filename, want_parallel);
  m_reader->initialize();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ArcaneBasic2CheckpointReaderService::
notifyEndRead()
{
  delete m_reader;
  m_reader = nullptr;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_REGISTER_SERVICE(ArcaneBasicCheckpointService,
                        ServiceProperty("ArcaneBasicCheckpointWriter", ST_SubDomain | ST_CaseOption),
                        ARCANE_SERVICE_INTERFACE(Arcane::ICheckpointWriter));

ARCANE_REGISTER_SERVICE(ArcaneBasicCheckpointService,
                        ServiceProperty("ArcaneBasicCheckpointReader", ST_SubDomain),
                        ARCANE_SERVICE_INTERFACE(Arcane::ICheckpointReader));

ARCANE_REGISTER_SERVICE(ArcaneBasic2CheckpointService,
                        ServiceProperty("ArcaneBasic2CheckpointWriter", ST_SubDomain | ST_CaseOption),
                        ARCANE_SERVICE_INTERFACE(Arcane::ICheckpointWriter));

ARCANE_REGISTER_SERVICE(ArcaneBasic2CheckpointReaderService,
                        ServiceProperty("ArcaneBasic2CheckpointReader", ST_Application),
                        ARCANE_SERVICE_INTERFACE(Arcane::ICheckpointReader2));

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
