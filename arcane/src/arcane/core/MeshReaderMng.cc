// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2024 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* MeshReaderMng.h                                             (C) 2000-2024 */
/*                                                                           */
/* Gestionnaire de lecteurs de maillage.                                     */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/core/MeshReaderMng.h"

#include "arcane/utils/UniqueArray.h"
#include "arcane/utils/ScopedPtr.h"

#include "arcane/core/ISubDomain.h"
#include "arcane/core/IMainFactory.h"
#include "arcane/core/IMeshReader.h"
#include "arcane/core/ServiceBuilder.h"
#include "arcane/core/IPrimaryMesh.h"
#include "arcane/core/XmlNode.h"
#include "arcane/core/Properties.h"
#include "arcane/core/IXmlDocumentHolder.h"
#include "arcane/core/IParallelMng.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class IMeshReader;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class MeshReaderMng::Impl
{
 public:

  explicit Impl(ISubDomain* sd)
  : m_sub_domain(sd)
  {}
  ~Impl() = default;

 public:

  void checkInit()
  {
    if (m_is_init)
      return;
    m_is_init = true;
    ServiceBuilder<IMeshReader> builder(m_sub_domain);
    m_mesh_readers = builder.createAllInstances();
  }

 public:

  ConstArrayView<Ref<IMeshReader>> readers() const { return m_mesh_readers; }

 public:

  ISubDomain* m_sub_domain = nullptr;
  UniqueArray<Ref<IMeshReader>> m_mesh_readers;
  bool m_is_init = false;
  bool m_is_use_unit = true;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

MeshReaderMng::
MeshReaderMng(ISubDomain* sd)
: m_p(new Impl(sd))
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

MeshReaderMng::
~MeshReaderMng()
{
  delete m_p;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

IMesh* MeshReaderMng::
readMesh(const String& mesh_name,const String& file_name)
{
  ISubDomain* sd = m_p->m_sub_domain;
  IParallelMng* pm = sd->parallelMng()->sequentialParallelMng();
  return readMesh(mesh_name,file_name,pm);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
// TODO: fusionner cette méthode avec cell de ISubDomain.
IMesh* MeshReaderMng::
readMesh(const String& mesh_name,const String& file_name,IParallelMng* parallel_mng)
{
  m_p->checkInit();
  String extension;
  {
    // Cherche l'extension du fichier et la conserve dans \a extension
    std::string_view fview = file_name.toStdStringView();
    std::size_t extension_pos = fview.find_last_of('.');
    if (extension_pos==std::string_view::npos)
      ARCANE_FATAL("file name '{0}' has no extension",file_name);
    fview.remove_prefix(extension_pos+1);
    extension = fview;

  }
  // TODO: à terme, créer le maillage par le lecteur.
  ISubDomain* sd = m_p->m_sub_domain;
  IParallelMng* pm = parallel_mng;
  IPrimaryMesh* mesh = sd->mainFactory()->createMesh(sd,pm,mesh_name);

  // Créé le maillage.
  // Le maillage peut déjà exister.
  // Dans notre cas, c'est une erreur s'il est déjà alloué.
  if (mesh->isAllocated())
    ARCANE_FATAL("Mesh '{0}' already exists and is allocated", mesh_name);

  mesh->properties()->setBool("dump", false);

  String use_unit_str = (m_p->m_is_use_unit) ? "true" : "false";
  String use_unit_xml = "<?xml version=\"1.0\"?><file use-unit='"+use_unit_str+"' />";

  ITraceMng* tm = sd->traceMng();
  ScopedPtrT<IXmlDocumentHolder> xml_doc(IXmlDocumentHolder::loadFromBuffer(use_unit_xml.bytes(), String(),tm));
  XmlNode mesh_xml_node = xml_doc->documentNode().documentElement();

  String dir_name;
  bool is_bad = true;
  bool use_internal_partition = pm->isParallel();
  for( auto& reader_ref : m_p->readers() ){
    IMeshReader* reader = reader_ref.get();
    if (!reader->allowExtension(extension))
      continue;

    auto ret = reader->readMeshFromFile(mesh,mesh_xml_node,
                                        file_name,dir_name,
                                        use_internal_partition);
    if (ret==IMeshReader::RTOk){
      is_bad = false;
      break;
    }
    if (ret==IMeshReader::RTError){
      ARCANE_FATAL("Can not read mesh file '{0}'",file_name);
    }
  }

  if (is_bad)
    ARCANE_FATAL("No mesh reader is available for mesh file '{0}'",file_name);

  return mesh;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MeshReaderMng::
setUseMeshUnit(bool v)
{
  m_p->m_is_use_unit = v;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

bool MeshReaderMng::
isUseMeshUnit() const
{
  return m_p->m_is_use_unit;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

