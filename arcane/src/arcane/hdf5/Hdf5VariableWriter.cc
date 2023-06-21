// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2023 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* Hdf5VariableWriter.cc                                       (C) 2000-2023 */
/*                                                                           */
/* Ecriture de variables au format HDF5.                                     */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/ArcanePrecomp.h"

#include "arcane/utils/ScopedPtr.h"
#include "arcane/utils/Array.h"
#include "arcane/utils/StringBuilder.h"
#include "arcane/utils/NotImplementedException.h"
#include "arcane/utils/StringBuilder.h"

#include "arcane/AbstractService.h"
#include "arcane/IRessourceMng.h"
#include "arcane/BasicTimeLoopService.h"
#include "arcane/IVariableWriter.h"
#include "arcane/IXmlDocumentHolder.h"
#include "arcane/IIOMng.h"
#include "arcane/IMesh.h"
#include "arcane/PostProcessorWriterBase.h"
#include "arcane/IDataWriter.h"
#include "arcane/IParallelMng.h"
#include "arcane/Directory.h"
#include "arcane/VariableCollection.h"
#include "arcane/IMeshMng.h"

#include "arcane/utils/Collection.h"

#include "arcane/hdf5/Hdf5VariableWriter_axl.h"
#include "arcane/hdf5/Hdf5VariableInfoBase.h"

#include <set>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_BEGIN_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

using namespace Hdf5Utils;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class Hdf5VariableWriterHelper
: public TraceAccessor
{
 public:
  Hdf5VariableWriterHelper(IMesh* mesh,const String& xml_filename)
  : TraceAccessor(mesh->traceMng()), m_mesh(mesh), m_xml_file_name(xml_filename)
  {
  }
  virtual ~Hdf5VariableWriterHelper(){}
 public:
  virtual void open();
  virtual void notifyRestore(){}
  virtual void writeOnExit();
 private:
  IMesh* m_mesh;
  String m_xml_file_name;
  String m_hdf5_file_name;
  Hdf5Utils::StandardTypes m_types;
  HFile m_file_id; //!< Identifiant HDF du fichier 
  ScopedPtrT<IXmlDocumentHolder> m_xml_document_holder;
  UniqueArray<Hdf5VariableInfoBase*> m_exit_variables;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void Hdf5VariableWriterHelper::
open()
{
  IIOMng* io_mng = m_mesh->parallelMng()->ioMng();
  m_xml_document_holder = io_mng->parseXmlFile(m_xml_file_name);
  if (!m_xml_document_holder.get())
    fatal() << "Can not read file '" << m_xml_file_name << "'";

  XmlNode root_element = m_xml_document_holder->documentNode().documentElement();
  m_hdf5_file_name = root_element.attrValue("file-name",true);

  // Lecture des variables pour les sorties finales
  {
    XmlNodeList variables_elem = root_element.children("exit-variable");
    for( XmlNode elem : variables_elem ){
      String var_name = elem.attrValue("name",true);
      String var_family = elem.attrValue("family",true);
      String var_path = elem.attrValue("path",true);
      info() << "VARIABLE: name=" << var_name << " path=" << var_path
             << " family=" << var_family;
      Hdf5VariableInfoBase* var_info = Hdf5VariableInfoBase::create(m_mesh,var_name,var_family);
      var_info->setPath(var_path);
      m_exit_variables.add(var_info);
    }
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void Hdf5VariableWriterHelper::
writeOnExit()
{
  //TODO lancer exception en cas d'erreur.
  HFile hfile;

  if (m_mesh->parallelMng()->isMasterIO()){
    hfile.openTruncate(m_hdf5_file_name);
  }

  std::set<ItemGroup> groups_to_write;
  for( Integer iz=0, izs=m_exit_variables.size(); iz<izs; ++iz ){
    Hdf5VariableInfoBase* vi = m_exit_variables[iz];
    groups_to_write.insert(vi->variable()->itemGroup());
    vi->writeVariable(hfile,m_types);
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Ecriture de variables au format HDF5.
 */
class Hdf5VariableWriter
: public ArcaneHdf5VariableWriterObject
{
 public:

  Hdf5VariableWriter(const ServiceBuildInfo& sbi);
  ~Hdf5VariableWriter();

 public:

  void build() override {}
  void onTimeLoopStartInit() override
  {
    IMeshMng* mm = subDomain()->meshMng();
    for( Integer i=0, is=options()->write.size(); i<is; ++i ){
      String file_name = options()->write[i]->fileName();
      String mesh_name = options()->write[i]->meshName();
      info() << "Hdf5VariableWriter: FILE_INFO: mesh=" << mesh_name << " file_name=" << file_name;
      {
        MeshHandle mesh_handle = mm->findMeshHandle(mesh_name);
        Hdf5VariableWriterHelper* sd = new Hdf5VariableWriterHelper(mesh_handle.mesh(),file_name);
        m_writers.add(sd);
      }
    }
    info() << "Hdf5VariableWriter: Nb writer =" << m_writers.size();
    for( Integer i=0, is=m_writers.size(); i<is; ++i ){
      m_writers[i]->open();
    }
  }
  void onTimeLoopExit() override
  {
    info() << "Hdf5VariableWriter: End loop";
    for( Integer i=0, is=m_writers.size(); i<is; ++i ){
      m_writers[i]->writeOnExit();
    }
  }
  void onTimeLoopRestore() override
  {
    for( Integer i=0, is=m_writers.size(); i<is; ++i ){
      m_writers[i]->notifyRestore();
    }
  }
  void onTimeLoopBeginLoop() override {}
 private:
  UniqueArray<Hdf5VariableWriterHelper*> m_writers;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Hdf5VariableWriter::
Hdf5VariableWriter(const ServiceBuildInfo& sbi)
: ArcaneHdf5VariableWriterObject(sbi)
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Hdf5VariableWriter::
~Hdf5VariableWriter()
{
  for( Integer i=0, is=m_writers.size(); i<is; ++i )
    delete m_writers[i];
  m_writers.clear();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class ManualHdf5DataWriter
: public TraceAccessor
, public IDataWriter
{
  typedef std::set<ItemGroup> ItemGroupSet;
 public:
  ManualHdf5DataWriter(IParallelMng* pm,Integer index,const String& directory_name,
                       const String& file_name)
  : TraceAccessor(pm->traceMng()), m_parallel_mng(pm), m_index(index),
    m_directory_name(directory_name), m_file_name(file_name)
  {
  }
 public:
  
  virtual void beginWrite(const VariableCollection& vars)
  {
    info(4) << "BEGIN WRITE N=" << vars.count() << " INDEX=" << m_index << " directory=" << m_directory_name;
    Directory out_dir(m_directory_name);
    String full_filename = out_dir.file(m_file_name);
    info(4) << "OUT FILE_NAME=" << full_filename;
    if (m_parallel_mng->isMasterIO()){
      if (m_index<=1){
        m_hdf_file.openTruncate(full_filename);
      }
      else
        m_hdf_file.openAppend(full_filename);
    }
    m_saved_groups.clear();
  }
  virtual void endWrite()
  {
  }
  virtual void setMetaData(const String& meta_data)
  {
    ARCANE_UNUSED(meta_data);
  }
  virtual void write(IVariable* var,IData* data)
  {
    ARCANE_UNUSED(data);
    info(4) << "WRITE VAR name=" << var->fullName();
    String index_path = String("Index") + m_index;
    StringBuilder path = index_path;
    path += "/Variables/";
    path += var->fullName();

    ScopedPtrT<Hdf5VariableInfoBase> var_info(Hdf5VariableInfoBase::create(var));
    var_info->setPath(path);
    var_info->writeVariable(m_hdf_file,m_hdf5_types);
    if (m_index==1){
      // HACK: pour l'instant, sauve uniquement les infos de groupe lors de la premiere protection
      // Normalement, cela devrait etre specifié par l'API.
      ItemGroup group = var->itemGroup();
      if (!group.null() && m_saved_groups.find(group)==m_saved_groups.end()){
        Integer save_type = Hdf5VariableInfoBase::SAVE_IDS + Hdf5VariableInfoBase::SAVE_COORDS;
        String group_path = index_path + "/Groups/" + group.name();
        var_info->writeGroup(m_hdf_file,m_hdf5_types,group_path,save_type);
      }
      m_saved_groups.insert(group);
    }
  }

  void writeInfos(ByteConstArrayView bytes)
  {
    Hdf5Utils::StandardArrayT<Byte> v(m_hdf_file.id(),"Infos");
    v.write(m_hdf5_types,bytes);
  }

 private:
  Hdf5Utils::StandardTypes m_hdf5_types;
  IParallelMng* m_parallel_mng;
  Integer m_index;
  String m_directory_name;
  String m_file_name;
  HFile m_hdf_file;
  std::set<ItemGroup> m_saved_groups;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Ecriture de variables au format HDF5.
 */
class ManualHdf5VariableWriter
: public PostProcessorWriterBase
{
 public:

  ManualHdf5VariableWriter(const ServiceBuildInfo& sbi);
  ~ManualHdf5VariableWriter();

 public:

  virtual void build()
  {
  }

  //! Retourne l'écrivain associé à ce post-processeur
  virtual IDataWriter* dataWriter() { return m_writer; }

  //! Notifie qu'une sortie va être effectuée avec les paramètres courants.
  virtual void notifyBeginWrite()
  {
    _setFileName();
    VariableCollection variables = this->variables();
    Integer index = times().size();
    info() << "Hdf5VariableWriter: nb_vars=" << variables.count() << " index=" << index;
    m_writer = new ManualHdf5DataWriter(subDomain()->parallelMng(),index,baseDirectoryName(),m_file_name);
  }

  //! Notifie qu'une sortie vient d'être effectuée.
  virtual void notifyEndWrite()
  {
    // Ecrit le fichier XML contenant les infos des temps et des variables
    bool is_master = subDomain()->parallelMng()->isMasterIO();
    if (is_master){
      IApplication* app = subDomain()->application();
      ScopedPtrT<IXmlDocumentHolder> info_doc(app->ressourceMng()->createXmlDocument());
      XmlNode doc = info_doc->documentNode();
      XmlElement root_element = XmlElement(doc,"hdf5");
      root_element.setAttrValue("file-name",m_file_name);
      VariableCollection variables = this->variables();
      RealConstArrayView my_times = times();
      for( VariableCollection::Enumerator ivar(variables); ++ivar; ){
        IVariable* var = *ivar;
        // Ne gère que les variables sur des entités du maillage.
        if (var->itemKind()==IK_Unknown)
          continue;
        XmlNode x = root_element.createAndAppendElement("time-variable");
        x.setAttrValue("name",var->name());
        x.setAttrValue("family",var->itemFamilyName());
        for( Integer i=0, n=my_times.size(); i<n; ++i ){
          XmlNode tx = x.createAndAppendElement("time-value");
          String index_path = String::format("Index{0}/Variables/{1}",i+1,var->fullName());
          tx.setAttrValue("path",index_path);
          tx.setAttrValue("global-time",String::fromNumber(my_times[i]));
        }
      }
      ByteUniqueArray xml_bytes;
      info_doc->save(xml_bytes);
      if (m_writer)
        m_writer->writeInfos(xml_bytes);
      app->ioMng()->writeXmlFile(info_doc.get(),"toto.xml");
      // Il faut écrire le fichier XML dans le fichier hdf5.
    }
    delete m_writer;
    m_writer = 0;
  }

  //! Ferme l'écrivain. Après fermeture, il ne peut plus être utilisé
  virtual void close() {}

 private:

  ManualHdf5DataWriter* m_writer;
  String m_file_name;

 private:

  void _setFileName()
  {
    String base_file_name = baseFileName();
    if (base_file_name.null())
      base_file_name = "data";
    m_file_name = base_file_name + ".h5";
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ManualHdf5VariableWriter::
ManualHdf5VariableWriter(const ServiceBuildInfo& sbi)
: PostProcessorWriterBase(sbi)
, m_writer(0)
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ManualHdf5VariableWriter::
~ManualHdf5VariableWriter()
{
  delete m_writer;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_REGISTER_SERVICE_HDF5VARIABLEWRITER(Hdf5VariableWriter,
                                           Hdf5VariableWriter);

ARCANE_REGISTER_SUB_DOMAIN_FACTORY(ManualHdf5VariableWriter,
                                   IPostProcessorWriter,
                                   Hdf5VariableWriter);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_END_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
