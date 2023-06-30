// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* VariableMng.cc                                              (C) 2000-2023 */
/*                                                                           */
/* Classe gérant l'ensemble des variables.                                   */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/impl/internal/VariableMng.h"

#include "arcane/utils/Iterator.h"
#include "arcane/utils/Iostream.h"
#include "arcane/utils/Deleter.h"
#include "arcane/utils/StringBuilder.h"
#include "arcane/utils/OStringStream.h"
#include "arcane/utils/ITraceMng.h"
#include "arcane/utils/ScopedPtr.h"
#include "arcane/utils/NotImplementedException.h"
#include "arcane/utils/FatalErrorException.h"
#include "arcane/utils/ParallelFatalErrorException.h"
#include "arcane/utils/MD5HashAlgorithm.h"
#include "arcane/utils/Convert.h"
#include "arcane/utils/PlatformUtils.h"
#include "arcane/utils/Math.h"
#include "arcane/utils/JSONWriter.h"
#include "arcane/utils/Event.h"

#include "arcane/core/IVariableMng.h"
#include "arcane/core/IMeshMng.h"
#include "arcane/core/IApplication.h"
#include "arcane/core/IRessourceMng.h"
#include "arcane/core/Variable.h"
#include "arcane/core/VariableRef.h"
#include "arcane/core/VarRefEnumerator.h"
#include "arcane/core/IModule.h"
#include "arcane/core/ISubDomain.h"
#include "arcane/core/ArcaneException.h"
#include "arcane/core/Directory.h"
#include "arcane/core/IItemFamily.h"
#include "arcane/core/IObservable.h"
#include "arcane/core/ServiceUtils.h"
#include "arcane/core/ICheckpointReader.h"
#include "arcane/core/ICheckpointWriter.h"
#include "arcane/core/IPostProcessorWriter.h"
#include "arcane/core/VariableInfo.h"
#include "arcane/core/VariableBuildInfo.h"
#include "arcane/core/VariableFactoryRegisterer.h"
#include "arcane/core/IVariableFactory.h"
#include "arcane/core/IIOMng.h"
#include "arcane/core/XmlNode.h"
#include "arcane/core/XmlNodeList.h"
#include "arcane/core/IXmlDocumentHolder.h"
#include "arcane/core/ItemGroup.h"
#include "arcane/core/IMesh.h"
#include "arcane/core/IDataReader.h"
#include "arcane/core/IDataReader2.h"
#include "arcane/core/IDataWriter.h"
#include "arcane/core/Timer.h"
#include "arcane/core/IParallelMng.h"
#include "arcane/core/ISerializedData.h"
#include "arcane/core/IModuleMng.h"
#include "arcane/core/ITimeLoopMng.h"
#include "arcane/core/IEntryPoint.h"
#include "arcane/core/IApplication.h"
#include "arcane/core/IMainFactory.h"
#include "arcane/core/Properties.h"
#include "arcane/core/VariableStatusChangedEventArgs.h"
#include "arcane/core/VariableMetaData.h"
#include "arcane/core/CheckpointInfo.h"
#include "arcane/core/IMeshFactoryMng.h"
#include "arcane/core/MeshBuildInfo.h"

#include "arcane/impl/VariableUtilities.h"

#include <exception>
#include <set>
#include <vector>

// TODO: gérer le hash en version 64 bits.

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

extern "C++" IVariableMng*
arcaneCreateVariableMng(ISubDomain* sd)
{
  auto vm = new VariableMng(sd);
  vm->build();
  vm->initialize();
  return vm;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Interface pour utiliser IDataReader ou IDataReader2.
 */
class VariableMng::IDataReaderWrapper
{
 public:
  virtual ~IDataReaderWrapper(){}
 public:
  virtual void beginRead(const VariableCollection& vars) =0;
  virtual void read(VariableMetaData* vmd,IVariable* var,IData* data) =0;
  virtual void endRead() =0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Wrapper pour IDataReader.
 */
class VariableMng::OldDataReaderWrapper
: public IDataReaderWrapper
{
 public:
  OldDataReaderWrapper(IDataReader* reader) : m_reader(reader){}
  void beginRead(const VariableCollection& vars) override
  {
    return m_reader->beginRead(vars);
  }
  void read(VariableMetaData* vmd,IVariable* var,IData* data) override
  {
    ARCANE_UNUSED(vmd);
    m_reader->read(var,data);
  }
  void endRead() override
  {
    return m_reader->endRead();
  }
 private:
  IDataReader* m_reader;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Wrapper pour IDataReader2.
 */
class VariableMng::DataReaderWrapper
: public IDataReaderWrapper
{
 public:
  DataReaderWrapper(IDataReader2* reader) : m_reader(reader){}
  void beginRead(const VariableCollection& vars) override
  {
    ARCANE_UNUSED(vars);
    return m_reader->beginRead(DataReaderInfo());
  }
  void read(VariableMetaData* vmd,IVariable* var,IData* data) override
  {
    ARCANE_UNUSED(var);
    m_reader->read(VariableDataReadInfo(vmd,data));
  }
  void endRead() override
  {
    return m_reader->endRead();
  }
 private:
  IDataReader2* m_reader;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Construit un gestionnaire de variable pour le cas \a pn
 * \warning Pour un cas donné, il faut créer un et un seul gestionnaire de
 * variable.
 */
VariableMng::
VariableMng(ISubDomain* sd)
: TraceAccessor(sd->traceMng())
, m_sub_domain(sd)
, m_parallel_mng(sd->parallelMng())
, m_vni_map(2000,true)
, m_write_observable(IObservable::createDefault())
, m_read_observable(IObservable::createDefault())
, m_utilities(new VariableUtilities(this))
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Détruit le gestionnaire
 *
 * Le gestionnaire effectue la libération mémoire des variables qu'il gère.
 */
VariableMng::
~VariableMng()
{
  delete m_utilities;

  m_write_observable->detachAllObservers();
  m_read_observable->detachAllObservers();

  delete m_write_observable;
  delete m_read_observable;

  m_variable_factories.each(Deleter());
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void VariableMng::
build()
{
  String s = platform::getEnvironmentVariable("ARCANE_TRACE_VARIABLE_CREATION");
  if (!s.null())
    VariableRef::setTraceCreation(true);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void VariableMng::
initialize()
{
  VariableFactoryRegisterer* vff = VariableFactoryRegisterer::firstVariableFactory();
  while (vff){
    IVariableFactory* vf = vff->createFactory();
    String full_name = vf->fullTypeName();
    // Vérifie qu'aucune fabrique avec le même nom n'existe.
    if (m_variable_factory_map.find(full_name)!=m_variable_factory_map.end()){
      ARCANE_FATAL("VariableFactoryMap already contains a factory for the same type '{0}'",
                   full_name);
    }
    m_variable_factory_map.insert(VariableFactoryPair(full_name,vf));
    m_variable_factories.add(vf);
    info(5) << "Add variable factory kind=" << vff->itemKind()
            << " data_type=" << vff->dataType()
            << " dim=" << vff->dimension()
            << " multi_tag=" << vff->multiTag()
            << " full_name=" << full_name
            << " addr=" << vf;
    vff = vff->nextVariableFactory();
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void VariableMng::
removeAllVariables()
{
  //ATTENTION: ceci entraine des appels à removeVariable()
  m_auto_create_variables.each(Deleter());

  OStringStream var_str;
  UniqueArray<IVariable*> remaining_vars;
  for( const auto& i : m_full_name_variable_map ){
    IVariable* v = i.second;
    if (v->nbReference()==0)
      delete v;
    else{
      remaining_vars.add(v);
      var_str() << "  " << v->fullName() << " (" << v->nbReference() << ")";
    }
  }
  bool is_check = arcaneIsCheck();
  const bool has_remaining_vars = !remaining_vars.empty();
  if (has_remaining_vars && is_check)
    pwarning() << "The following variables are still referenced: "
               << var_str.str()
               << " (set the environment variable ARCANE_TRACE_VARIABLE_CREATION"
               << " to get the stack trace)";
  bool has_trace = VariableRef::hasTraceCreation();
  if (has_trace){
    for( const auto& i : remaining_vars ){
      for( VarRefEnumerator ivar(i); ivar.hasNext(); ++ivar ){
        VariableRef* var = *ivar;
        info() << " variable name=" << var->name()
               << " stack=" << var->assignmentStackTrace();
      }
    }
  }

  // Appelle explicitement 'unregisterVariable()' sur les variables restantes.
  // Sans cela, si ensuite l'instance 'this' est détruite avant que les variables
  // restantent ne le soit cela va provoquer un plantage (Read after free). Cela
  // n'arrive normalement pas pour le C++ mais peut arriver pour le wrapping.
  if (has_remaining_vars){
    // Recopie les références dans un tableau temporaire
    // car les appels à unregisterVariable() modifient l'itérateur ivar
    // et aussi m_full_name_variable_map.
    UniqueArray<VariableRef*> remaining_refs;
    for( const auto& i : remaining_vars )
      for( VarRefEnumerator ivar(i); ivar.hasNext(); ++ivar )
        remaining_refs.add(*ivar);
    for( VariableRef* r : remaining_refs )
      r->unregisterVariable();
    if (is_check)
      info() << "Remaining variables after cleanup n=" << m_full_name_variable_map.size();
  }

  m_full_name_variable_map.clear();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void VariableMng::
detachMeshVariables(IMesh* mesh)
{
  for( const auto& i : m_full_name_variable_map ){
    IVariable* v = i.second;
    ItemGroup group = v->itemGroup();
    if (group.null())
      continue;
    if (group.mesh()==mesh){
      v->setUsed(false);
    }
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void VariableMng::
addVariableRef(VariableRef* ref)
{
  ARCANE_UNUSED(ref);
  ++m_nb_created_variable_reference;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void VariableMng::
removeVariableRef(VariableRef* ref)
{
  ARCANE_UNUSED(ref);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void VariableMng::
addVariable(IVariable* var)
{
  const String& full_name = var->fullName();
  subDomain()->checkId("VariableMng::checkVariable()",full_name);
  if (var->itemKind()!=IK_Unknown && var->itemFamilyName().null())
    ARCANE_FATAL("Bad Variable full-name={0} name={1}",var->fullName(),var->name());

  info(5) << "Add variable"
          << " name=" << var->name()
          << " full_name=" << full_name
          << " datatype=" << var->dataType()
          << " kind=" << var->itemKind();

  VariableNameInfo vni;
  vni.m_local_name = var->name();
  vni.m_family_name = var->itemFamilyName();
  vni.m_mesh_name = var->meshName();
  m_vni_map.add(vni,var);

  m_full_name_variable_map.insert(FullNameVariablePair(full_name,var));
  m_variables_changed = true;
  m_used_variables_changed = true;
  ++m_nb_created_variable;
  IEntryPoint* ep = subDomain()->timeLoopMng()->currentEntryPoint();
  IModule* module = nullptr;
  if (ep)
    module = ep->module();
  m_variable_creation_modules.insert(std::make_pair(var,module));
  VariableStatusChangedEventArgs eargs(var,VariableStatusChangedEventArgs::Status::Added);
  m_on_variable_added.notify(eargs);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void VariableMng::
removeVariable(IVariable* var)
{
  int p = var->property();
  if (p & IVariable::PTemporary){
    debug() << "** ** REMOVE " << var->name() << " " << var->nbReference()
           << " property=" << p
           << " nodump=" << (p & IVariable::PNoDump)
           << " tmp=" << (p & IVariable::PTemporary)
           << " norestore=" << (p & IVariable::PNoRestore);
  }
  VariableStatusChangedEventArgs eargs(var,VariableStatusChangedEventArgs::Status::Removed);
  m_on_variable_removed.notify(eargs);
  {
    ItemGroup var_group = var->itemGroup();
    // Retire cette variable de tous les groupes [bien défini mais sur ItemGroupImplNull]
    if (!var_group.null())
      var_group.internal()->detachObserver(var);
    // Retire cette variable des variables existantes, puis la détruit
    m_full_name_variable_map.erase(var->fullName());
    {
      VariableNameInfo vni;
      vni.m_local_name = var->name();
      vni.m_family_name = var->itemFamilyName();
      vni.m_mesh_name = var->meshName();
      m_vni_map.remove(vni);
    }
    m_variables_changed = true;
    m_used_variables_changed = true;
    m_variable_creation_modules.erase(var);
    delete var;
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

IVariable* VariableMng::
checkVariable(const VariableInfo& infos)
{
  VariableNameInfo vni;
  vni.m_local_name = infos.localName();
  vni.m_family_name = infos.itemFamilyName();
  vni.m_mesh_name = infos.meshName();

  if (infos.itemKind()!=IK_Unknown && infos.meshName().null())
    ARCANE_FATAL("Mesh variable without a mesh  full-name={0} name={1}",infos.fullName(),infos.localName());

  // Si variable du maillage, vérifie qu'aucune variable globale non lié à un maillage
  // ne porte le même nom.
  if (arcaneIsCheck()){
    if (!infos.meshName().null()){
      String check_name = infos.localName();
      if (findVariableFullyQualified(check_name)){
        ARCANE_FATAL("Mesh variable has the same name that a global variable (name={0})",check_name);
      }
    }
    else{
      // Si variable globale, vérifie qu'aucune variable du maillage ne porte le même nom.
      String check_name = String("Mesh0_")+infos.localName();
      if (findVariableFullyQualified(check_name)){
        ARCANE_FATAL("Global variable has the same name that a mesh variable (name={0})",check_name);
      }
    }
  }

  IVariable* var = nullptr;
  //VNIMap::Data* var_data = m_vni_map.lookup(vni);

  //cerr << "** CHECK " << name << ' ' << infos.dataType() << ' ' << infos.kind() << '\n';
  VNIMap::Data* var_data = m_vni_map.lookup(vni);
  if (var_data){
    // Une variable de même nom que \a var existe déjà.
    // Il faut dans ce cas vérifier que son genre et son type sont les même.
    // elle a le même genre et le même type.
    var = var_data->value();
    //cerr << "** FIND " << prv->name() << ' ' << prv->dataType() << ' ' << prv->kind() << '\n';
    if (infos.dataType()!=var->dataType() ||
        infos.itemKind()!=var->itemKind() ||
        infos.dimension()!=var->dimension()){
      throw BadVariableKindTypeException(A_FUNCINFO,var,infos.itemKind(),
                                         infos.dataType(),infos.dimension());
    }
  }
  return var;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

IVariable* VariableMng::
findVariable(const String& name)
{
  IVariable* v = findVariableFullyQualified(name);
  if (v)
    return v;
  v = findVariableFullyQualified(String("Node_")+name);
  if (v)
    return v;
  v = findVariableFullyQualified(String("Edge_")+name);
  if (v)
    return v;
  v = findVariableFullyQualified(String("Face_")+name);
  if (v)
    return v;
  v = findVariableFullyQualified(String("Cell_")+name);
  if (v)
    return v;
  return nullptr;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

IVariable* VariableMng::
findMeshVariable(IMesh* mesh,const String& name)
{
  String mesh_name = mesh->name();
  mesh_name = mesh_name + "_";
  IVariable* v = findVariableFullyQualified(mesh_name+name);
  if (v)
    return v;
  v = findVariableFullyQualified(mesh_name+"Node_"+name);
  if (v)
    return v;
  v = findVariableFullyQualified(mesh_name+"Edge_"+name);
  if (v)
    return v;
  v = findVariableFullyQualified(mesh_name+"Face_"+name);
  if (v)
    return v;
  v = findVariableFullyQualified(mesh_name+"Cell_"+name);
  if (v)
    return v;
  return nullptr;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

IVariable* VariableMng::
findVariableFullyQualified(const String& name)
{
  auto i = m_full_name_variable_map.find(name);
  if (i!=m_full_name_variable_map.end())
    return i->second;
  return nullptr;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

String VariableMng::
generateTemporaryVariableName()
{
  bool is_bad = true;
  String name;
  while (is_bad){
    name = String("ArcaneTemporary") + m_generate_name_id;
    // Vérifie que le nom généré ne correspond pas à une variable existante
    if (findVariable(name))
      ++m_generate_name_id;
    else
      is_bad = false;
  }
  info() << "** GENERATED NAME =" << name;
  return name;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void VariableMng::
dumpList(std::ostream& o,IModule* c)
{
  o << "  ** VariableMng::Variable list\n";
  for( const auto& i : m_full_name_variable_map ){
    for( VarRefEnumerator ivar(i.second); ivar.hasNext(); ++ivar ){
      if ((*ivar)->module()!=c)
        continue;
      _dumpVariable(*(*ivar),o);
    }
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void VariableMng::
dumpList(std::ostream& o)
{
  o << "  ** VariableMng::Variable list\n";
  for( const auto& i : m_full_name_variable_map ){
    for( VarRefEnumerator ivar(i.second); ivar.hasNext(); ++ivar ){
      _dumpVariable(*(*ivar),o);
    }
  }
  {
    Real mem_used = 0;
    for( const auto& i : m_full_name_variable_map )
      mem_used += i.second->allocatedMemory();
    o << "  ** VariableMng::Allocated memory : " << mem_used;
    o << '\n';
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void VariableMng::
_dumpVariable(const VariableRef& var,std::ostream& o)
{
  o << "  ** Variable: " << &var << " : ";
  o.width(15);
  o << var.name() << " = ";
  var.print(o);
  o << " (Type " << var.dataType() << ")\n";
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void VariableMng::
initializeVariables(bool is_continue)
{
  ARCANE_UNUSED(is_continue);

  info() << "Initialisation des variables";
  for( const auto& i : m_full_name_variable_map ){
    for( VarRefEnumerator ivar(i.second); ivar.hasNext(); ++ivar ){
      VariableRef* var_ref = *ivar;
      IModule* module = var_ref->module();
      if (module && !module->used())
        continue;
      IVariable* var = var_ref->variable();
      if (var->isUsed())
        continue;
      var_ref->setUsed(true);
    }
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void VariableMng::
variables(VariableRefCollection v,IModule* c)
{
  for( const auto& i : m_full_name_variable_map ){
    for( VarRefEnumerator ivar(i.second); ivar.hasNext(); ++ivar ){
      if ((*ivar)->module()==c)
        v.add(*ivar);
    }
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

VariableCollection VariableMng::
variables()
{
  if (m_variables_changed){
    m_variables_changed = false;
    m_variables.clear();
    for( const auto& i : m_full_name_variable_map ){
      m_variables.add(i.second);
    }
  }
  return m_variables;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

VariableCollection VariableMng::
usedVariables()
{
  if (m_used_variables_changed){
    m_used_variables_changed = false;
    m_used_variables.clear();
    for( const auto& i : m_full_name_variable_map ){
      IVariable* var = i.second;
      if (var->isUsed())
        m_used_variables.add(var);
    }
  }
  return m_used_variables;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

bool VariableMng::
isVariableToSave(IVariable& var)
{
  if (!var.isUsed())
    return false;
  bool no_dump = var.property() & (IVariable::PNoDump|IVariable::PTemporary);
  if (no_dump)
    return false;
  IMesh* mesh = var.meshHandle()._internalMeshOrNull();
  if (mesh && !mesh->properties()->getBool("dump"))
    return false;
  return true;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void VariableMng::
writeCheckpoint(ICheckpointWriter* service)
{
  if (!service)
    ARCANE_FATAL("No protection service specified");

  Trace::Setter mci(traceMng(),_msgClassName());

  Timer::Phase tp(subDomain(),TP_InputOutput);

  CheckpointSaveFilter save_filter;

  service->notifyBeginWrite();
  IDataWriter* data_writer = service->dataWriter();
  if (!data_writer)
    ARCANE_FATAL("no writer() nor dataWriter()");
  writeVariables2(data_writer,&save_filter,true);
  service->notifyEndWrite();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void VariableMng::
writePostProcessing(IPostProcessorWriter* post_processor)
{
  Trace::Setter mci(traceMng(),_msgClassName());

  if (!post_processor)
    ARCANE_FATAL("No post-processing service specified");

  Timer::Phase tp(subDomain(),TP_InputOutput);

  post_processor->notifyBeginWrite();
  VariableCollection variables(post_processor->variables());
  IDataWriter* data_writer = post_processor->dataWriter();
  if (!data_writer)
    ARCANE_FATAL("no writer() nor dataWriter()");
  writeVariables2(data_writer,variables,false);
  post_processor->notifyEndWrite();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void VariableMng::
writeVariables(IDataWriter* writer,const VariableCollection& vars)
{
  writeVariables2(writer,vars,false);
}

void VariableMng::
writeVariables(IDataWriter* writer,IVariableFilter* filter)
{
  writeVariables2(writer,filter,false);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void VariableMng::
writeVariables2(IDataWriter* writer,IVariableFilter* filter,bool use_hash)
{
  Trace::Setter mci(traceMng(),_msgClassName());

  if (!writer)
    ARCANE_FATAL("No writer available for protection");

  // Calcul la liste des variables à sauver
  VariableList vars;
  for( const auto& i : m_full_name_variable_map ){
    IVariable* var = i.second;
    bool apply_var = true;
    if (filter)
      apply_var = filter->applyFilter(*var);
    if (apply_var)
      vars.add(var);
  }
  _writeVariables(writer,vars,use_hash);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \todo prendre en compte le NoDump
 */
void VariableMng::
writeVariables2(IDataWriter* writer,const VariableCollection& vars,bool use_hash)
{
  if (!writer)
    return;
  if (vars.empty()){
    VariableList var_array;
    for( const auto& i : m_full_name_variable_map ){
      IVariable* var = i.second;
      var_array.add(var);
    }
    _writeVariables(writer,var_array,use_hash);
  }
  else
    _writeVariables(writer,vars,use_hash);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

String VariableMng::
_generateMetaData(const VariableCollection& vars,bool use_hash)
{
  IApplication* app = subDomain()->application();
  ScopedPtrT<IXmlDocumentHolder> doc(app->ressourceMng()->createXmlDocument());
  XmlNode doc_node = doc->documentNode();
  XmlElement root_element(doc_node,"arcane-checkpoint-metadata");
  XmlElement variables_node(root_element,"variables");
  _generateVariablesMetaData(variables_node,vars,use_hash);
  XmlElement meshes_node(root_element,"meshes");
  _generateMeshesMetaData(meshes_node);
  String s = doc->save();
  info(6) << "META_DATA=" << s;
  return s;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void VariableMng::
_generateVariablesMetaData(XmlNode variables_node,const VariableCollection& vars,bool use_hash)
{
  StringBuilder var_full_type_b;
  ByteUniqueArray hash_values;
  MD5HashAlgorithm hash_algo;

  for( VariableCollection::Enumerator i(vars); ++i; ){
    IVariable* var = *i;
    ScopedPtrT<VariableMetaData> vmd(var->createMetaData());
    String var_full_type = vmd->fullType();
    String var_family_name = var->itemFamilyName();
    String var_mesh_name = var->meshName();
    XmlNode var_node = XmlElement(variables_node,"variable");
    var_node.setAttrValue("base-name",var->name());
    if (!var_family_name.null())
      var_node.setAttrValue("item-family-name",var_family_name);
    if (var->isPartial())
      var_node.setAttrValue("item-group-name",var->itemGroupName());
    if (!var_mesh_name.null()){
      var_node.setAttrValue("mesh-name",var_mesh_name);
    }
    var_node.setAttrValue("full-type",var_full_type);
    var_node.setAttrValue("data-type",dataTypeName(var->dataType()));
    var_node.setAttrValue("dimension",String::fromNumber(var->dimension()));
    var_node.setAttrValue("multi-tag",String::fromNumber(var->multiTag()));
    var_node.setAttrValue("property",String::fromNumber(var->property()));
    if (use_hash){
      hash_values.clear();
      var->data()->computeHash(&hash_algo,hash_values);
      String hash_str = Convert::toHexaString(hash_values);
      var_node.setAttrValue("hash",hash_str);
    }
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * TEMPORAIRE.
 * TODO Cela doit normalement être fait via le 'IMeshMng'.
 */
void VariableMng::
_generateMeshesMetaData(XmlNode meshes_node)
{
  // Positionne un numéro de version pour compatibilité avec de futures versions
  meshes_node.setAttrValue("version","1");
  ISubDomain* sd = subDomain();
  IMesh* default_mesh = sd->defaultMesh();
  bool is_parallel = m_parallel_mng->isParallel();
  IParallelMng* seq_pm = m_parallel_mng->sequentialParallelMng();
  ConstArrayView<IMesh*> meshes = sd->meshes();
  for( Integer i=0, n=meshes.size(); i<n; ++i ){
    IMesh* mesh = meshes[i];
    bool do_dump = mesh->properties()->getBool("dump");
    // Sauve le maillage s'il est marqué dump ou s'il s'agit du maillage par défaut
    if (do_dump || mesh==default_mesh){
      XmlNode mesh_node = XmlElement(meshes_node,"mesh");
      mesh_node.setAttrValue("name",mesh->name());
      mesh_node.setAttrValue("factory-name",mesh->factoryName());
      // Indique si le maillage utilise le gestionnaire de parallélisme
      // séquentiel car dans ce cas en reprise il faut le créer avec le
      // même gestionnaire.
      // TODO: il faudrait traiter les cas où un maillage est créé
      // avec un IParallelMng qui n'est ni séquentiel, ni celui du
      // sous-domaine.
      if (is_parallel && mesh->parallelMng()==seq_pm)
        mesh_node.setAttrValue("sequential","true");
    }
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void VariableMng::
_writeVariables(IDataWriter* writer,const VariableCollection& vars,bool use_hash)
{
  if (!writer)
    return;

  m_write_observable->notifyAllObservers();
  writer->beginWrite(vars);

  // Appelle la notification de l'écriture des variables
  // Il faut le faire avant de positionner les méta-données
  // car cela autorise de changer la valeur de la variable
  // lors de cet appel.
  for( VariableCollection::Enumerator i(vars); ++i; ){
    IVariable* var = *i;
    if (var->isUsed())
      var->notifyBeginWrite();
  }

  String meta_data = _generateMetaData(vars,use_hash);
  writer->setMetaData(meta_data);

  for( VariableCollection::Enumerator i(vars); ++i; ){
    IVariable* var = *i;
    if (!var->isUsed())
      continue;
    try{
      writer->write(var,var->data());
    }
    catch(const Exception& ex)
    {
      error() << "Exception Arcane while VariableMng::writeVariables()"
              << " var=" << var->fullName()
              << " exception=" << ex;
      throw;
    }
    catch(const std::exception& ex){
      error() << "Exception while VariableMng::writeVariables()"
              << " var=" << var->fullName()
              << " exception=" << ex.what();
      throw;
    }
  }
  writer->endWrite();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

VariableRef* VariableMng::
_createVariableFromType(const String& full_type,const VariableBuildInfo& vbi)
{
  auto i = m_variable_factory_map.find(full_type);
  if (i==m_variable_factory_map.end())
    ARCANE_FATAL("No factory to create variable name={0} type={1}",vbi.name(),full_type);

  IVariableFactory* vf = i->second;
  info(5) << "Automatic creation of the variable"
          << " name=" << vbi.name()
          << " family=" << vbi.itemFamilyName()
          << " type=" << full_type
          << " vf=" << vf;
  VariableRef* var_ref = vf->createVariable(vbi);
  // Ajoute la variable à une liste pour être sur qu'elle sera bien
  // détruite.
  m_auto_create_variables.add(var_ref);
  return var_ref;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class VariableMetaDataList
{
 public:
  typedef std::map<String,VariableMetaData*> VMDMap;
 public:
  VariableMetaDataList() = default;
  VariableMetaDataList(const VariableMetaDataList& rhs) = delete;
  VariableMetaDataList& operator=(const VariableMetaDataList& rhs) = delete;
  ~VariableMetaDataList()
  {
    clear();
  }
 public:
  VariableMetaData* add(const String& base_name,const String& mesh_name,
                        const String& family_name,const String& group_name,
                        bool is_partial)
  {
    auto vmd = new VariableMetaData(base_name,mesh_name,family_name,group_name,is_partial);
    return add(vmd);
  }
  VariableMetaData* add(VariableMetaData* vmd)
  {
    m_vmd_map.insert(std::make_pair(vmd->fullName(),vmd));
    return vmd;
  }
  void clear()
  {
    for( const auto& x : m_vmd_map )
      delete x.second;
    m_vmd_map.clear();
  }
  VariableMetaData* findMetaData(const String& full_name)
  {
    auto x = m_vmd_map.find(full_name);
    if (x!=m_vmd_map.end())
      return x->second;
    return nullptr;
  }
  VMDMap::const_iterator begin() const { return m_vmd_map.begin(); }
  VMDMap::const_iterator end() const { return m_vmd_map.end(); }
 public:
  std::map<String,VariableMetaData*> m_vmd_map;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Gestion de la lecture des variables.
 */
class VariableReaderMng
: public TraceAccessor
{
  struct VarReadInfo
  {
   public:
    VarReadInfo(IVariable* var,IData* data,VariableMetaData* meta_data)
    : m_variable(var), m_data(data), m_meta_data(meta_data){}
   public:
    IVariable* m_variable;
    IData* m_data;
    VariableMetaData* m_meta_data;
  };
 public:
  VariableReaderMng(ITraceMng* tm)
  : TraceAccessor(tm){}
 public:
  void readVariablesData(IVariableMng* vm,VariableMng::IDataReaderWrapper* reader);
  VariableMetaDataList& variableMetaDataList() { return m_vmd_list; }
  const VariableList& variablesToRead() { return m_vars_to_read; }
 private:
  VariableMetaDataList m_vmd_list;
  VariableList m_vars_to_read;
  UniqueArray<VarReadInfo> m_var_read_info_list;
 private:
  void _buildVariablesToRead(IVariableMng* vm);
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Lecture des méta-données.
 *
 * En considérant que \a meta_data est un fichier XML valide, parcours
 * l'ensemble des variables le contenant et crée une référence sur
 * chacune si elles n'existent pas encore.
 *
 * Si nécessaire, pour chaque variable présente dans les méta-données,
 * la créée si elle n'existe pas encore. De plus, sauve le nom pour être
 * certain que la valeur de cette variable sera bien lue.
 */
void VariableMng::
_readMetaData(VariableMetaDataList& vmd_list,Span<const Byte> bytes)
{
  ISubDomain* sd = subDomain();
  IIOMng* io_mng = sd->ioMng();
  ScopedPtrT<IXmlDocumentHolder> doc(io_mng->parseXmlBuffer(bytes,"meta_data"));
  if (!doc.get())
    ARCANE_FATAL("The meta-data are invalid");

  XmlNode root_node = doc->documentNode().documentElement();
  XmlNode variables_node = root_node.child("variables");
  _readVariablesMetaData(vmd_list,variables_node);
  XmlNode meshes_node = root_node.child("meshes");
  _readMeshesMetaData(meshes_node);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace
{
// Extraction des infos de type à partir d'une chaîne de caractères
// NOTE L'extraction doit être cohérente avec la construction qui est
// dans Variable.cc
class VariableDataTypeInfo
{
 public:
  explicit VariableDataTypeInfo(const String& full_type)
  : m_is_partial(false)
  {
    std::vector<String> split_strs;
    full_type.split(split_strs,'.');
    size_t nb_split = split_strs.size();
    if (nb_split==5){
      if (split_strs[4]!="Partial")
        ARCANE_FATAL("Invalid value for partial full_type '{0}'",full_type);
      m_is_partial = true;
      --nb_split;
    }
    if (nb_split!=4)
      ARCANE_FATAL("Invalid value for full_type '{0}'",full_type);
    m_data_type_name = split_strs[0];
    m_item_kind_name = split_strs[1];
    m_dimension = split_strs[2];
    m_multi_tag = split_strs[3];
  }
 public:
  const String& dataTypeName() const { return m_data_type_name; }
  const String& itemKindName() const { return m_item_kind_name; }
  const String& dimension() const { return m_dimension; }
  const String& multiTag() const { return m_multi_tag; }
  bool isPartial() const { return m_is_partial; }
 private:
  String m_data_type_name;
  String m_item_kind_name;
  String m_dimension;
  String m_multi_tag;
  bool m_is_partial;
};
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void VariableMng::
_readVariablesMetaData(VariableMetaDataList& vmd_list,const XmlNode& variables_node)
{
  XmlNodeList vars = variables_node.children("variable");
  String ustr_base_name("base-name");
  String ustr_family_name("item-family-name");
  String ustr_group_name("item-group-name");
  String ustr_mesh_name("mesh-name");
  String ustr_full_type("full-type");
  String ustr_hash("hash");
  String ustr_property("property");
  String ustr_multitag("multi-tag");
  vmd_list.clear();

  for( const auto& var : vars ){
    String full_type = var.attrValue(ustr_full_type);
    VariableDataTypeInfo vdti(full_type);

    String base_name = var.attrValue(ustr_base_name);
    String mesh_name = var.attrValue(ustr_mesh_name);
    String family_name = var.attrValue(ustr_family_name);
    // Actuellement, si la variable n'est pas partielle alors son groupe
    // n'est pas sauvé dans les meta-données. Il faut donc le générer.
    String group_name = var.attrValue(ustr_group_name);
    bool is_partial = vdti.isPartial();
    if (!is_partial){
      // NOTE: Cette construction doit être cohérente avec celle de
      // DynamicMeshKindInfos. A terme il faudra toujours sauver le nom du groupe
      // dans les meta-données.
      group_name = "All" + family_name + "s";
    }
    auto vmd = vmd_list.add(base_name,mesh_name,family_name,group_name,is_partial);

    vmd->setFullType(var.attrValue(ustr_full_type));
    vmd->setHash(var.attrValue(ustr_hash));
    vmd->setMultiTag(var.attrValue(ustr_multitag));
    vmd->setProperty(var.attr(ustr_property).valueAsInteger());

    info(5) << "CHECK VAR: "
            << " base-name=" << vmd->baseName()
            << " mesh-name=" << vmd->meshName()
            << " family-name=" << vmd->itemFamilyName()
            << " full-type=" << vmd->fullType()
            << " name=" << vmd->fullName()
            << " multitag=" << vmd->multiTag()
            << " property=" << vmd->property()
            << " hash=" << vmd->hash();
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void VariableMng::
_createVariablesFromMetaData(const VariableMetaDataList& vmd_list)
{
  ISubDomain* sd = subDomain();
  // Récupère ou construit les variables qui n'existent pas encore.
  for( const auto& xvmd : vmd_list ){
    auto& vmd = *(xvmd.second);
    const String& full_name = vmd.fullName();
    IVariable* var = findVariableFullyQualified(full_name);
    if (var)
      continue;
    const String& base_name = vmd.baseName();
    Integer property = vmd.property();
    const String& mesh_name = vmd.meshName();
    const String& group_name = vmd.itemGroupName();
    const String& family_name = vmd.itemFamilyName();
    VariableBuildInfo vbi(sd,base_name,property);
    if (!mesh_name.null()){
      if (vmd.isPartial())
        vbi = VariableBuildInfo(sd,base_name,mesh_name,family_name,group_name,property);
      else
        vbi = VariableBuildInfo(sd,base_name,mesh_name,family_name,property);
    }
    info(5) << "Create variable TYPE=" << full_name;
    _createVariableFromType(vmd.fullType(),vbi);
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void VariableMng::
_readMeshesMetaData(const XmlNode& meshes_node)
{
  XmlNodeList meshes = meshes_node.children("mesh");
  ISubDomain* sd = subDomain();
  IMeshMng* mesh_mng = sd->meshMng();
  IMeshFactoryMng* mesh_factory_mng = mesh_mng->meshFactoryMng();
  for( XmlNode var : meshes ){
    String mesh_name = var.attrValue("name");
    String mesh_factory_name = var.attrValue("factory-name");
    MeshHandle* mesh_handle = mesh_mng->findMeshHandle(mesh_name,false);
    IMesh* mesh = (mesh_handle) ? mesh_handle->mesh() : nullptr;
    if (mesh)
      continue;
    bool is_sequential = var.attr("sequential",false).valueAsBoolean();
    info() << "Creating from checkpoint mesh='" << mesh_name
           << "' sequential?=" << is_sequential
           << " factory=" << mesh_factory_name;
    // Depuis avril 2020, l'attribut 'factory-name' est doit être présent
    // et sa valeur non nulle.
    if (mesh_factory_name.null())
      ARCANE_FATAL("No attribute 'factory-name' for mesh");

    {
      MeshBuildInfo mbi(mesh_name);
      mbi.addFactoryName(mesh_factory_name);
      IParallelMng* mesh_pm = m_parallel_mng;
      if (is_sequential)
        mesh_pm = mesh_pm->sequentialParallelMng();
      mbi.addParallelMng(Arccore::makeRef(mesh_pm));
      mesh_factory_mng->createMesh(mbi);
    }
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void VariableMng::
readCheckpoint(ICheckpointReader* service)
{
  Trace::Setter mci(traceMng(),_msgClassName());
  
  if (!service)
    ARCANE_FATAL("No protection service specified");

  service->notifyBeginRead();

  IDataReader* data_reader = service->dataReader();
  if (!data_reader)
    ARCANE_FATAL("no dataReader()");
  String meta_data = data_reader->metaData();

  if (meta_data.null())
    ARCANE_FATAL("No meta-data in checkpoint.");

  OldDataReaderWrapper wrapper(data_reader);

  info(6) << "METADATA (ICheckpointReader): FromCheckpoint: " << meta_data;

  VariableReaderMng var_read_mng(traceMng());
  VariableMetaDataList& vmd_list = var_read_mng.variableMetaDataList();
  {
    _readMetaData(vmd_list,meta_data.bytes());
    _createVariablesFromMetaData(vmd_list);
    _readVariablesData(var_read_mng,&wrapper);
  }

  service->notifyEndRead();
  _checkHashFunction(vmd_list);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void VariableMng::
readCheckpoint(const CheckpointReadInfo& infos)
{
  Trace::Setter mci(traceMng(),_msgClassName());

  ICheckpointReader2* service = infos.reader();
  ARCANE_CHECK_POINTER2(service,"No checkpoint service specified");

  IParallelMng* pm = infos.parallelMng();
  ARCANE_CHECK_POINTER2(pm,"no parallelMng()");

  service->notifyBeginRead(infos);

  IDataReader2* data_reader = service->dataReader();
  ARCANE_CHECK_POINTER2(data_reader,"No dataReader()");

  UniqueArray<Byte> meta_data_bytes;
  data_reader->fillMetaData(meta_data_bytes);
  if (meta_data_bytes.empty())
    ARCANE_FATAL("No meta-data in checkpoint.");

  DataReaderWrapper wrapper(data_reader);

  info(6) << "METADATA (ICheckpointReader2): FromCheckpoint: " << String(meta_data_bytes);

  VariableReaderMng var_read_mng(traceMng());
  VariableMetaDataList& vmd_list = var_read_mng.variableMetaDataList();
  {
    _readMetaData(vmd_list,meta_data_bytes);
    _createVariablesFromMetaData(vmd_list);
    _readVariablesData(var_read_mng,&wrapper);
  }

  service->notifyEndRead();
  _checkHashFunction(vmd_list);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Vérifie les valeurs des fonctions de hashage.
 *
 * Vérifie pour chaque variable que sa valeur est correcte en calculant
 * sa fonction de hashage et en la comparant à la valeur dans la
 * protection.
 * Si une variable à une valeur différente, elle est écrite dans
 * le répertoire de listing au même niveau que les logs.
 */
void VariableMng::
_checkHashFunction(const VariableMetaDataList& vmd_list)
{
  ByteUniqueArray hash_values;
  MD5HashAlgorithm hash_algo;
  Integer nb_error = 0;
  IParallelMng* pm = m_parallel_mng;
  Int32 sid = pm->commRank();
  Directory listing_dir = subDomain()->listingDirectory();
  
  for( const auto& i : vmd_list ){
    String reference_hash = i.second->hash();
    // Teste si la valeur de hashage est présente. C'est normalement
    // toujours le cas, sauf si la protection vient d'une ancienne
    // version de Arcane qui ne sauvait pas cette information.
    // Ce test pourra être supprimé plus tard.
    if (reference_hash.null())
      continue;
    const String& full_name = i.first;
    IVariable* var = findVariableFullyQualified(full_name);
    if (!var)
      // Ne devrait pas arriver
      continue;
    hash_values.clear();
    IData* data = var->data();
    data->computeHash(&hash_algo,hash_values);
    String hash_str = Convert::toHexaString(hash_values);
    if (hash_str!=reference_hash){
      ++nb_error;
      error() << "Hash values are different. Corrumpted values."
              << " name=" << var->fullName()
              << " ref=" << reference_hash
              << " current=" << hash_str;
      Ref<ISerializedData> sdata(data->createSerializedDataRef(false));
      Span<const Byte> buf(sdata->constBytes());
      String fname = listing_dir.file(String::format("dump-{0}-sid_{1}",var->fullName(),sid));
      std::ofstream ofile(fname.localstr());
      ofile.write(reinterpret_cast<const char*>(buf.data()),buf.size());
    }
  }
  Integer total_nb_error = pm->reduce(Parallel::ReduceSum,nb_error);
  if (total_nb_error!=0){
    bool allow_bad = !platform::getEnvironmentVariable("ARCANE_ALLOW_DIFFERENT_CHECKPOINT_HASH").null();
    if (!allow_bad)
      throw ParallelFatalErrorException(A_FUNCINFO,"hash functions differs");
  }
}

/*-----------------------------------------Q----------------------------------*/
/*---------------------------------------------------------------------------*/

void VariableMng::
_buildFilteredVariableList(VariableReaderMng& var_read_mng,IVariableFilter* filter)
{
  VariableMetaDataList& vmd_list = var_read_mng.variableMetaDataList();
  for( const auto& i : m_full_name_variable_map ){
    IVariable* var = i.second;
    bool apply_me = true;
    if (filter)
      apply_me = filter->applyFilter(*var);
    info(5) << "Read variable name=" << var->fullName() << " filter=" << apply_me;
    if (apply_me){
      VariableMetaData* vmd = var->createMetaData();
      vmd_list.add(vmd);
    }
  }
}

/*-----------------------------------------Q----------------------------------*/
/*---------------------------------------------------------------------------*/

void VariableMng::
readVariables(IDataReader* reader,IVariableFilter* filter)
{
  Trace::Setter mci(traceMng(),_msgClassName());
  VariableReaderMng var_read_mng(traceMng());
  _buildFilteredVariableList(var_read_mng,filter);
  OldDataReaderWrapper wrapper(reader);
  _readVariablesData(var_read_mng,&wrapper);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void VariableMng::
_readVariablesData(VariableReaderMng& var_read_mng,IDataReaderWrapper* reader)
{
  var_read_mng.readVariablesData(this,reader);
  _finalizeReadVariables(var_read_mng.variablesToRead());
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void VariableReaderMng::
_buildVariablesToRead(IVariableMng* vm)
{
  m_vars_to_read.clear();
  m_var_read_info_list.clear();
  for( const auto& x : m_vmd_list ){
    const String& full_name = x.first;
    IVariable* var = vm->findVariableFullyQualified(full_name);
    if (!var)
      ARCANE_FATAL("Var {0} not in VariableMng",full_name);
    m_vars_to_read.add(var);
    m_var_read_info_list.add(VarReadInfo(var,var->data(),x.second));
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void VariableReaderMng::
readVariablesData(IVariableMng* vm,VariableMng::IDataReaderWrapper* reader)
{
  _buildVariablesToRead(vm);
  reader->beginRead(m_vars_to_read);
  for( const auto& ivar : m_var_read_info_list ){
    // NOTE: var peut-être nul
    IVariable* var = ivar.m_variable;
    IData* data = ivar.m_data;
    VariableMetaData* vmd = ivar.m_meta_data;
    String exception_message;
    bool has_error = false;
    try{
      reader->read(vmd,var,data);
      if (var)
        var->notifyEndRead();
    }
    catch(const Exception& ex){
      OStringStream ostr;
      ostr() << ex;
      exception_message = ostr.str();
      has_error = true;
    }
    catch(const std::exception& ex){
      exception_message = ex.what();
      has_error = true;
    }
    if (has_error){
      OStringStream ostr;
      String var_full_name = vmd->fullName();
      ostr() << "Variable = " << var_full_name;
      if (var){
        for( VarRefEnumerator ivar(var); ivar.hasNext(); ++ivar ){
          VariableRef* ref = *ivar;
          String s = ref->assignmentStackTrace();
          if (!s.null())
            ostr() << "Stack assignement: " << s;
        }
      }
    
      ARCANE_FATAL("Can not read variable variable={0} exception={1} infos={2}",
                   var_full_name,exception_message,ostr.str());
    }
  }
  reader->endRead();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void VariableMng::
_finalizeReadVariables(const VariableList& vars_to_read)
{
  ARCANE_UNUSED(vars_to_read);

  info(4) << "VariableMng: _finalizeReadVariables()";

  // Resynchronise en lecture les valeurs de toutes les variables pour
  // être sur que les références sont toujours correctes (en cas de
  // réallocation mémoire).
  // NOTE: en théorie cela ne doit pas être utile car IVariable::notifyEndRead()
  // se charge de faire cela.
  // NOTE: de plus, il n'est nécessaire de le faire que sur les variables
  // de \a vars_to_read.
  for( const auto& i : m_full_name_variable_map  )
    i.second->syncReferences();

  // Notifie les observateurs qu'une lecture vient d'être faite.
  m_read_observable->notifyAllObservers();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \todo prendre en compte le NoDump
 */
Real VariableMng::
exportSize(const VariableCollection& vars)
{
  Real total_size = 0;
  if (vars.empty()){
    for( const auto& i : m_full_name_variable_map ){
      IVariable* var = i.second;
      if (var->isUsed()){
        Real n = (Real)(var->allocatedMemory());
        total_size += n;
      }
    }
  }
  else{
    for( VariableCollection::Enumerator i(vars); ++i; ){
      IVariable* var = *i;
      if (var->isUsed())
        total_size += (Real)(var->allocatedMemory());
    }
  }
  total_size /= 1.0e6;

  return total_size;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
//! Trieur de variable suivant leur taille mémoire utilisée
class VariableSizeSorter
{
 public:
  bool operator()(IVariable* v1,IVariable* v2)
  {
    return v1->allocatedMemory() > v2->allocatedMemory();
  }
};

void VariableMng::
dumpStats(std::ostream& ostr,bool is_verbose)
{
  ostr.precision(20);
  ostr << "\nMemory statistics for variables:\n";
  ostr << " Number of created variables:              " << m_nb_created_variable << '\n';
  ostr << " Number of created variables reference:    " << m_nb_created_variable_reference << '\n';
  ostr << " Number of currently allocated variables:  " << m_full_name_variable_map.size() << '\n';
  ostr << " Number of currently reference:            " << m_variables_ref.count() << '\n';

  // Statistiques sur la mémoire utilisée
  Integer total_nb_var = 0;
  Integer nb_var[NB_ITEM_KIND];
  Real mem_used[NB_ITEM_KIND];
  Real mem_used_array = 0.;
  Real mem_used_particle = 0.;
  Real mem_used_link = 0.;
  Real total_mem_used = 0.;
  Integer nb_var_array = 0;
  Integer nb_var_particle = 0;
  Integer nb_var_link = 0;
  for( Integer i=0; i<NB_ITEM_KIND; ++i ){
    mem_used[i] = 0;
    nb_var[i] = 0;
  }

  // Récupère le nombre de mailles pour faire des stats d'utilisation
  // mémoire moyenne par maille.
  Integer nb_cell = 1;
  if (subDomain()->defaultMesh()) nb_cell = subDomain()->defaultMesh()->allCells().size();
  if (nb_cell==0)
    nb_cell = 1;

  typedef std::map<IModule*,std::set<IVariable*> > ModuleVariableMap;
  std::set<IVariable*> variables_with_module;
  UniqueArray<IVariable*> memory_sorted_variables;

  ModuleVariableMap modules_variables;
  for( const auto& i : m_full_name_variable_map ){
    //for( VariableRefList::Enumerator ivar(m_variables_ref); ++ivar; ){
    IVariable* var = i.second;
    if (!var->isUsed())
      continue;
    // Pas de statistiques sur les variables scalaires
    if (var->dimension()==0)
      continue;
    for( VarRefEnumerator ivar(var); ivar.hasNext(); ++ivar ){
      VariableRef* vref = *ivar;
      IModule* var_module = vref->module();
      // Si la variable n'a pas de module, recherche le module éventuel
      // qui l'a créée
      if (!var_module)
        var_module = m_variable_creation_modules[var];
      if (var_module){
        variables_with_module.insert(var);
        modules_variables[var_module].insert(var);
      }
    }
  }

  for( const auto& i : m_full_name_variable_map ){
    IVariable* var = i.second;
    // Pas de statistiques sur les variables non utilisées
    if (!var->isUsed())
      continue;
    // Pas de statistiques sur les variables scalaires
    if (var->dimension()==0)
      continue;
    memory_sorted_variables.add(var);
    // Si la variable n'a pas de module associé, la place dans
    // la liste des variables sans module
    if (variables_with_module.find(var)==variables_with_module.end())
      modules_variables[0].insert(var);
    ++total_nb_var;
    eItemKind ik = var->itemKind();
    Real mem = var->allocatedMemory();
    total_mem_used += mem;
    if (is_verbose)
      ostr << "Var: <" << var->name() << "> Kind=" << itemKindName(ik) << " Mem=" << mem << '\n';
    switch(ik){
    case IK_Node:
    case IK_Edge:
    case IK_Face:
    case IK_Cell:
    case IK_DoF:
      mem_used[ik] += mem;
      ++nb_var[ik];
      break;
    case IK_Particle:
      mem_used_particle += mem;
      ++nb_var_particle;
      break;
    case IK_Unknown:
      mem_used_array += mem;
      ++nb_var_array;
      break;
    }
  }

  ostr << "Memory repartition by module:\n";
  ostr << Trace::Width(30) << ""
       << Trace::Width(7) << ""
       << Trace::Width(12) << ""
       << Trace::Width(14) << "Memory (Mo)"
       << Trace::Width(10) << ""
       << Trace::Width(7) << " "
       << Trace::Width(13) << "Memory (Ko)"
       << '\n';
  ostr << Trace::Width(30) << "Module"
       << Trace::Width(7) << "Nvar"
       << Trace::Width(12) << "Private"
       << Trace::Width(12) << "Shared"
       << Trace::Width(12) << "Total"
       << Trace::Width(7) << "%"
       << Trace::Width(13) << "per cell"
       << "\n\n";
  String pr_true("X ");
  String pr_false("  ");
  for( ModuleVariableMap::const_iterator imodvar = modules_variables.begin();
       imodvar!=modules_variables.end(); ++imodvar ){
    IModule* module = imodvar->first;
    Real private_mem_used = 0.0;
    Real shared_mem_used = 0.0;
    for( std::set<IVariable*>::const_iterator i = imodvar->second.begin();
         i!=imodvar->second.end(); ++i ){
      IVariable* var = *i;
      Real mem_used2 = var->allocatedMemory();
      bool is_private = var->nbReference()==1;
      if (is_private)
        private_mem_used += mem_used2;
      else
        shared_mem_used += mem_used2;
      if (is_verbose)
        ostr << "Var: <" << var->name() << "> Kind=" << itemKindName(var->itemKind())
             << " Mem=" << mem_used << " private?=" << is_private << '\n';
    }
    String module_name = "None";
    if (module)
      module_name = module->name();
    Real module_mem_used = private_mem_used + shared_mem_used;
    ostr << Trace::Width(30) << module_name
         << Trace::Width(7) << imodvar->second.size()
         << Trace::Width(12) << String::fromNumber(private_mem_used / 1e6,3)
         << Trace::Width(12) << String::fromNumber(shared_mem_used / 1e6,3)
         << Trace::Width(12) << String::fromNumber(module_mem_used / 1e6,3)
         << Trace::Width(7) << String::fromNumber(100.0 * module_mem_used / total_mem_used,1) << "%"
         << Trace::Width(12) << String::fromNumber(module_mem_used / ((Real)nb_cell * 1000.0) ,2)
         << '\n';
  }
  ostr << '\n';
  ostr << Trace::Width(30) << "TOTAL"
       << Trace::Width(7) << total_nb_var
       << Trace::Width(12) << ""
       << Trace::Width(12) << ""
       << Trace::Width(12) << String::fromNumber(total_mem_used/ 1e6,3)
       << Trace::Width(7) << " "
       << Trace::Width(13) << String::fromNumber(total_mem_used / ((Real)nb_cell * 1000.0) ,2)
       << '\n';

  if (is_verbose){
    for( Integer i=0; i<NB_ITEM_KIND; ++i ){
      ostr << "Variable " << itemKindName((eItemKind)i) << " N=" << nb_var[i]
           << " Mémoire=" << mem_used[i] << '\n';
    }
    ostr << "Variable Particle N=" << nb_var_particle
         << " Mémoire=" << mem_used_particle << '\n';
    ostr << "Variable Link N=" << nb_var_link
         << " Mémoire=" << mem_used_link << '\n';
    ostr << "Variable Array N=" << nb_var_array
         << " Mémoire=" << mem_used_array << '\n';
    ostr << "Variable Total N=" << total_nb_var
         << " Mémoire=" << total_mem_used << '\n';
  }

  std::sort(std::begin(memory_sorted_variables),std::end(memory_sorted_variables),
            VariableSizeSorter());


  Integer nb_var_to_display = memory_sorted_variables.size();
  if (!is_verbose)
    nb_var_to_display = math::min(nb_var_to_display,15);
  ostr << "\nBiggest variables (D=Dump, E=Exchange R=Restore):\n";
  ostr << Trace::Width(45) << "Variable"
       << Trace::Width(10) << "Kind"
       << Trace::Width(16) << "Memory (Ko)"
       << Trace::Width(14) << "per cell (o)"
       << Trace::Width(7) << "D E R"
       << "\n\n";
  for( Integer i=0; i<nb_var_to_display; ++i ){
    IVariable* var = memory_sorted_variables[i];
    Real mem_used2 = var->allocatedMemory();
    //ostr << "Var: <" << var->name() << "> Kind=" << itemKindName(var->itemKind())
    //<< " Mem=" <<var->allocatedMemory() << '\n';
  
    StringBuilder properties;
    int var_property = var->property();
    bool is_no_exchange = (var_property & IVariable::PNoExchange);
    if (var->itemKind()==IK_Unknown)
      // Seules les variables du maillage peuvent s'échanger
      is_no_exchange = true;
    properties += (var_property & IVariable::PNoDump) ? pr_false : pr_true;
    properties += (is_no_exchange) ? pr_false : pr_true;
    properties += (var_property & IVariable::PNoRestore) ? pr_false : pr_true;
    ostr << Trace::Width(45) << var->name()
         << Trace::Width(10) << itemKindName(var->itemKind())
         << Trace::Width(14) << String::fromNumber(mem_used2 / 1e3,3)
         << Trace::Width(12) << String::fromNumber(mem_used2 / ((Real)nb_cell),1)
         << Trace::Width(12) << properties.toString()
         << '\n';
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void VariableMng::
dumpStatsJSON(JSONWriter& writer)
{
  writer.writeKey("Variables");
  writer.beginArray();
  for( const auto& i : m_full_name_variable_map ){
    IVariable* var = i.second;
    if (var->dimension()==0)
      continue;
    {
      JSONWriter::Object o(writer);
      writer.write("Used",var->isUsed());
      Real mem = var->allocatedMemory();
      writer.write("Name",var->name());
      writer.write("DataType",dataTypeName(var->dataType()));
      writer.write("Dimension",(Int64)var->dimension());
      writer.write("NbElement",(Int64)var->nbElement());
      writer.write("ItemFamily",var->itemFamilyName());
      writer.write("Mesh",var->meshName());
      writer.write("Group",var->itemGroupName());
      writer.write("Property",(Int64)var->property());
      writer.write("AllocatedMemory",mem);
    }
  }
  writer.endArray();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
