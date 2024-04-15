// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2024 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------

#include <arcane/utils/Collection.h>

#include <arcane/core/Directory.h>
#include <arcane/core/PostProcessorWriterBase.h>
#include <arcane/core/IDataWriter.h>
#include <arcane/core/IMesh.h>
#include <arcane/core/ItemGroup.h>
#include <arcane/core/IParallelMng.h>

#include <arcane/core/materials/IMeshMaterialMng.h>
#include <arcane/core/materials/IMeshEnvironment.h>
#include <arcane/core/materials/IMeshMaterial.h>

#include "SimplePostProcessor_axl.h"

using namespace Arcane;
using namespace Arcane::Materials;

namespace SimpleTest
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class SimplePostProcessorDataWriter
: public TraceAccessor
, public IDataWriter
{
 public:

  SimplePostProcessorDataWriter(IMesh* mesh)
  : TraceAccessor(mesh->traceMng())
  , m_mesh(mesh)
  {}

 public:

  void beginWrite(const VariableCollection& vars) override;
  void endWrite() override {}
  void setMetaData(const String& meta_data) override {}
  void write(IVariable* var, IData* data) override;

 public:

  void setTimes(SmallSpan<const Real> times) { m_times = times; }
  void setDirectoryName(const String& v) { m_directory_name = v; }

 private:

  UniqueArray<Real> m_times;
  String m_directory_name;
  IMesh* m_mesh = nullptr;
  IMeshMaterialMng* m_material_mng = nullptr;

 private:

  void _printMaterialsInfos();
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void SimplePostProcessorDataWriter::
beginWrite(const VariableCollection& vars)
{
  IParallelMng* pm = m_mesh->parallelMng();
  const Int32 nb_rank = pm->commSize();
  bool is_master_io = pm->isMasterIO();

  Directory dir(m_directory_name);

  // Seul le proc maitre créé le répertoire
  {
    if (is_master_io)
      dir.createDirectory();
    pm->barrier();
  }
  if (is_master_io) {
    // Écrit juste un fichier pour le test
    String filename = dir.file("testout");
    std::ofstream ofile(filename.localstr());
    ofile << "Hello\n";
  }

  m_material_mng = IMeshMaterialMng::getReference(m_mesh, false);
  _printMaterialsInfos();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void SimplePostProcessorDataWriter::
_printMaterialsInfos()
{
  if (!m_material_mng) {
    info() << "No IMeshMaterialMng associated to the mesh";
    return;
  }
  // Liste des milieux:
  for (IMeshEnvironment* env : m_material_mng->environments()) {
    info() << "Environment name=" << env->name();
    for (IMeshMaterial* mat : env->materials()) {
      info() << "  Material name=" << mat->name();
    }
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Ecrit les données \a data de la variable \a var.
 */
void SimplePostProcessorDataWriter::
write(IVariable* var, IData* data)
{
  info() << "Write variable name=" << var->name();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class SimplePostProcessorService
: public ArcaneSimplePostProcessorObject
{
 public:

  explicit SimplePostProcessorService(const ServiceBuildInfo& sbi);

 public:

  IDataWriter* dataWriter() override
  {
    return m_writer.get();
  }
  void notifyBeginWrite() override
  {
    auto w = std::make_unique<SimplePostProcessorDataWriter>(mesh());
    w->setTimes(times());
    Directory dir(baseDirectoryName());
    w->setDirectoryName(dir.file("simple_post_processor"));
    m_writer = std::move(w);
  }
  void notifyEndWrite() override {}
  void close() override {}

 private:

  std::unique_ptr<IDataWriter> m_writer;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

SimplePostProcessorService::
SimplePostProcessorService(const ServiceBuildInfo& sbi)
: ArcaneSimplePostProcessorObject(sbi)
{
  info() << "Create SimplePostProcessorService";
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_REGISTER_SERVICE_SIMPLEPOSTPROCESSOR(SimplePostProcessor,
                                            SimplePostProcessorService);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace SimpleTest

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
