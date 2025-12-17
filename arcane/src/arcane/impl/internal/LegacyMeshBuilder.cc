// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* LegacyMeshBuilder.cc                                        (C) 2000-2025 */
/*                                                                           */
/* Construction du maillage via la méthode "historique".                     */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/impl/internal/LegacyMeshBuilder.h"

#include "arcane/utils/PlatformUtils.h"
#include "arcane/utils/StringBuilder.h"

#include "arcane/core/MeshKind.h"
#include "arcane/core/ISubDomain.h"
#include "arcane/core/CaseNodeNames.h"
#include "arcane/core/ICaseDocument.h"
#include "arcane/core/IParallelMng.h"
#include "arcane/core/ServiceBuilder.h"
#include "arcane/core/IMainFactory.h"
#include "arcane/core/IPrimaryMesh.h"
#include "arcane/core/IGhostLayerMng.h"
#include "arcane/core/IMeshReader.h"
#include "arcane/core/IMeshMng.h"
#include "arcane/core/IMeshUniqueIdMng.h"

#include "arcane_internal_config.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

LegacyMeshBuilder::
LegacyMeshBuilder(ISubDomain* sd,MeshHandle default_mesh_handle)
: TraceAccessor(sd->traceMng())
, m_sub_domain(sd)
, m_default_mesh_handle(default_mesh_handle)
, m_internal_partitioner_name(ARCANE_DEFAULT_PARTITIONER_STR)
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void LegacyMeshBuilder::
readCaseMeshes()
{
  ISubDomain* sd = m_sub_domain;
  Integer sub_domain_id = sd->subDomainId();
  ICaseDocument* case_doc = sd->caseDocument();
  CaseNodeNames* cnn = case_doc->caseNodeNames();
  XmlNodeList mesh_elems(case_doc->meshElements());
  bool has_mesh_file = true;
  if (mesh_elems.empty()){
    info() << "No mesh in the input data";
    has_mesh_file = false;
  }
  Integer nb_mesh = mesh_elems.size();
  m_meshes_build_info.resize(nb_mesh);
  for( Integer i=0; i<nb_mesh; ++i ){
    MeshBuildInfo& mbi = m_meshes_build_info[i];
    mbi.m_dir_name = ".";
    mbi.m_xml_node = mesh_elems[i];
    XmlNodeList partitioner_elems = mesh_elems[i].children(cnn->mesh_partitioner) ;
    m_use_partitioner_tester = false;
    if(partitioner_elems.empty()) {
      m_use_partitioner_tester = true;
      m_internal_partitioner_name = ARCANE_DEFAULT_PARTITIONER_STR;
    }
    else{
      m_internal_partitioner_name = partitioner_elems[0].value() ;
      //check if the partitioner is parallel
      m_use_partitioner_tester = partitioner_elems[0].attr("need-basic-partition-first").valueAsBoolean();
    }
      
    XmlNode meshfile_elem = mesh_elems[i].child(cnn->mesh_file);
    String smesh_file = meshfile_elem.value();
    StringBuilder mesh_file = smesh_file;
    mbi.m_orig_file_name = smesh_file;
    if (smesh_file.null()){
      info() << "No mesh in the input data";
      has_mesh_file = false;
    }

    String file_format = meshfile_elem.attrValue("format");
    bool internal_cut = meshfile_elem.attr("internal-partition").valueAsBoolean();
    String internal_partitioner = meshfile_elem.attr("partitioner").value();

    // Si la variable d'environnement est définie, force le repartitionnement
    // initial avec le service dont le nom est spécifié dans cette variable.
    String internal_partitioner_env = platform::getEnvironmentVariable("ARCANE_INTERNAL_PARTITIONER");
    if (!internal_partitioner_env.null()){
      info() << "Forcing internal partitioner from environment variable";
      internal_cut = true;
      internal_partitioner = internal_partitioner_env;
    }
    IParallelMng* pm = sd->parallelMng();
    // Dans le cas où Arcane est retranché à un coeur, on ne va pas chercher les CPU*.mli2
    if (pm->isParallel() && (pm->commSize()>1)){
      m_use_internal_mesh_partitioner = internal_cut;
        
      if (!internal_partitioner.empty())
        m_internal_partitioner_name = internal_partitioner;
      Integer nb_sub_domain = pm->commSize();
      info() << "Subdomain number is " << sub_domain_id << '/' << nb_sub_domain;
        
      //check if the file mesh reader need a unique or multi file
      bool use_unique_file = meshfile_elem.attr("unique").valueAsBoolean();
      if (!use_unique_file){
        //Integer nb_sub_domain = m_parallel_mng->commSize();
        StringBuilder cut_dir_str("cut_");
        cut_dir_str += nb_sub_domain;
        String mesh_cut_dir = meshfile_elem.attrValue(cut_dir_str);
        debug() << "MESH CUT DIR " << mesh_cut_dir << ' ' << cut_dir_str;
        if (has_mesh_file && !internal_cut){
          char buf[128];
          String file_format_str = "mli2";
          if (!file_format.null())
            file_format_str = file_format;
          sprintf(buf,"CPU%05d.%s",(int)sub_domain_id,file_format_str.localstr());
          log() << "The original mesh file is " << mesh_file;
          if (mesh_cut_dir.empty())
            mesh_file = String(std::string_view(buf));
          else{
            mbi.m_dir_name = mesh_cut_dir;
            mesh_file = mesh_cut_dir;
            mesh_file += "/";
            mesh_file += buf;
          }
        }
      }
    }
    log() << "The mesh file is " << mesh_file;
    mbi.m_file_name = mesh_file;
    // Cette partie de la configuration doit être lue avant
    // la création du maillage car elle peut contenir des options
    // dont le générateur de maillage a besoin.
    //m_case_config->read();
  }

  // Créé les MeshHandle pour les maillages.
  // Cela permettra de les récupérer dans les points d'entrée 'Build'
  _createMeshesHandle();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void LegacyMeshBuilder::
readMeshes()
{
  // Construit les services de lecture de maillage. Ce sont
  // ceux qui implémentent IMeshReader.
  ServiceBuilder<IMeshReader> builder(m_sub_domain);
  UniqueArray<Ref<IMeshReader>> mesh_readers(builder.createAllInstances());

  for( const MeshBuildInfo& mbi : m_meshes_build_info ){
    _readMesh(mesh_readers,mbi);
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void LegacyMeshBuilder::
createDefaultMesh()
{
  ISubDomain* sd = m_sub_domain;
  String mesh_name = m_default_mesh_handle.meshName();
  ICaseDocument* case_doc = sd->caseDocument();
  if (!case_doc){
    // Si aucun jeu de données n'est spécifié, créé un maillage
    m_default_mesh_handle._setMesh(sd->mainFactory()->createMesh(sd,mesh_name));
    return;
  }

  CaseNodeNames* cnn = case_doc->caseNodeNames();
  XmlNodeList mesh_elems(case_doc->meshElements());
  if (mesh_elems.empty()){
    info() << "No mesh in the input data";
  }
  Integer nb_mesh = mesh_elems.size();
  for( Integer i=0; i<nb_mesh; ++i ){
    XmlNode meshfile_elem = mesh_elems[i].child(cnn->mesh_file);
    String mesh_file = meshfile_elem.value();
    if (mesh_file.null()){
      info() << "No mesh in the input data";
    }
  }
  // default_mesh is the mesh described by mesh_elems[0]
  // Now that amr flag has to be known at mesh creation, check-it for default mesh
  bool is_amr = mesh_elems[0].attr("amr").valueAsBoolean();
  eMeshAMRKind amr_type = static_cast<eMeshAMRKind>(mesh_elems[0].attr("amr-type").valueAsInteger());
  if(is_amr && amr_type == eMeshAMRKind::None) {
    amr_type = eMeshAMRKind::Cell;
  }

  m_default_mesh_handle._setMesh(sd->mainFactory()->createMesh(sd,mesh_name, amr_type));
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void LegacyMeshBuilder::
_createMeshesHandle()
{
  IMeshMng* mesh_mng = m_sub_domain->meshMng();

  // Le premier maillage est toujours celui par défaut
  Integer nb_build_mesh = m_meshes_build_info.size();
  if (nb_build_mesh>0){
    m_meshes_build_info[0].m_mesh_handle = m_default_mesh_handle;
  }

  // Créé les autres maillages spécifiés dans le jeu de données
  for( Integer z=1; z<nb_build_mesh; ++z ){
    String name;
    if(m_meshes_build_info[z].m_xml_node.attr("dual").valueAsBoolean())
      name = "DualMesh";
    else
      name = "Mesh";
    name = name + z;
    MeshHandle handle = mesh_mng->createMeshHandle(name);
    m_meshes_build_info[z].m_mesh_handle = handle;
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void LegacyMeshBuilder::
allocateMeshes()
{
  ISubDomain* sd = m_sub_domain;

  // Le premier maillage est toujours celui par défaut
  Integer nb_build_mesh = m_meshes_build_info.size();
  if (nb_build_mesh>0){
    m_meshes_build_info[0].m_mesh = m_default_mesh_handle.mesh()->toPrimaryMesh();
  }

  // Créé les autres maillages spécifiés dans le jeu de données
  for( Integer z=1; z<nb_build_mesh; ++z ){
    MeshHandle handle = m_meshes_build_info[z].m_mesh_handle;
    if (handle.isNull())
      ARCANE_FATAL("Invalid null MeshHandle for mesh index={0}",z);
    // Depuis la 1.8.0 (modif IFP), cette methode
    // appelle this->addMesh()
    bool is_amr = m_meshes_build_info[z].m_xml_node.attr("amr").valueAsBoolean();
    eMeshAMRKind amr_type = static_cast<eMeshAMRKind>(m_meshes_build_info[z].m_xml_node.attr("amr-type").valueAsInteger());
    if(is_amr && amr_type == eMeshAMRKind::None) {
      amr_type = eMeshAMRKind::Cell;
    }

    IPrimaryMesh* mesh = sd->mainFactory()->createMesh(sd,handle.meshName(),amr_type);
    m_meshes_build_info[z].m_mesh = mesh;
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void LegacyMeshBuilder::
initializeMeshVariablesFromCaseFile()
{
  info() << "Initialization of the variable from the configuration file";
  CaseNodeNames* cnn = m_sub_domain->caseDocument()->caseNodeNames();
  for( const LegacyMeshBuilder::MeshBuildInfo& mbi : m_meshes_build_info ){
    IMesh* mesh = mbi.m_mesh;
    XmlNode node = mbi.m_xml_node;
    XmlNode init_node = node.child(cnn->mesh_initialisation);
    if (!init_node.null())
      mesh->initializeVariables(init_node);
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void LegacyMeshBuilder::
_readMesh(ConstArrayView<Ref<IMeshReader>> mesh_readers,const MeshBuildInfo& mbi)
{
  IPrimaryMesh* mesh = mbi.m_mesh;
  ARCANE_CHECK_POINTER(mesh);

  String mesh_file_name = mbi.m_file_name;
  // Si un service de partitionnement est spécifié, il faut utiliser le fichier specifié
  // dans le JDD et pas le nom du fichier éventuellement transformé dans readCaseMeshes()
  bool use_internal_partitioner = m_use_internal_mesh_partitioner;
  if (m_initial_partitioner.get()){
    mesh_file_name = mbi.m_orig_file_name;
    use_internal_partitioner = true;
  }
  // Permet de forcer la dimension au cas ou le format ne peut pas la reconnaitre.
  Integer wanted_dimension = mbi.m_xml_node.attr("dimension").valueAsInteger();
  if (wanted_dimension!=0){
    info() << "Force mesh dimension to " << wanted_dimension;
    mesh->setDimension(wanted_dimension);
  }
  log() << "Mesh file: " << mesh_file_name;

  Integer nb_ghost_layer = -1;
  XmlNode nbGhostLayerNode = mbi.m_xml_node.attr("nb-ghostlayer");
  if (!nbGhostLayerNode.null()){
    nb_ghost_layer = nbGhostLayerNode.valueAsInteger();
    if (nb_ghost_layer>=0){
      info() << "Set number of ghost layers to '" << nb_ghost_layer << "' from caseoption";
      mesh->ghostLayerMng()->setNbGhostLayer(nb_ghost_layer);
    }
  }

  Integer builder_version = mbi.m_xml_node.attr("ghostlayer-builder-version").valueAsInteger();
  if (nb_ghost_layer>=0 && builder_version>=0){
    info() << "Set ghostlayer builder version to '" << builder_version << "' from caseoption";
    mesh->ghostLayerMng()->setBuilderVersion(builder_version);
  }

  XmlNode face_numbering_version_node = mbi.m_xml_node.attr("face-numbering-version");
  if (!face_numbering_version_node.null()) {
    Int32 v = face_numbering_version_node.valueAsInteger();
    if (v >= 0) {
      info() << "Set face numbering version to '" << v << "' from caseoption";
      mesh->meshUniqueIdMng()->setFaceBuilderVersion(v);
    }
  }
  bool is_bad = true;
  String extension;
  {
    // Cherche l'extension du fichier et la conserve dans \a case_ext
    std::string_view fview = mesh_file_name.toStdStringView();
    debug() << " MF=" << fview;
    std::size_t extension_pos = fview.find_last_of('.');
    if (extension_pos!=std::string_view::npos){
      fview.remove_prefix(extension_pos+1);
      extension = fview;
    }
  }

  for( auto& mesh_reader_ref : mesh_readers ){
    IMeshReader* mesh_reader = mesh_reader_ref.get();
    if (!mesh_reader->allowExtension(extension))
      continue;

    IMeshReader::eReturnType ret = mesh_reader->readMeshFromFile(mesh,
                                                                 mbi.m_xml_node,
                                                                 mesh_file_name,
                                                                 mbi.m_dir_name,
                                                                 use_internal_partitioner);
    if (ret==IMeshReader::RTOk){
      is_bad = false;
      break;
    }
    if (ret==IMeshReader::RTError){
      ARCANE_FATAL("Error while generating the mesh");
    }
  }

  if (is_bad){
    ARCANE_FATAL("Internal error: no mesh loaded or generated. \n",
                 "The mesh reader or generator required isn't available ",
                 "Recompile with the relevant options");
  }

  mesh->computeTiedInterfaces(mbi.m_xml_node);
  //! AMR
  //mesh->readAmrActivator(mbi.m_xml_node);

#if 0
  IMeshWriter* writer = ServiceFinderT<IMeshWriter>::find(serviceMng(),"VtkLegacyMeshWriter");
  if (writer){
    writer->writeMeshToFile(mesh,"test.vtk");
  }
  if (!parallelMng()->isParallel()){
    mesh_utils::writeMeshConnectivity(mesh,"mesh-reference.xml");
  }
#endif
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
