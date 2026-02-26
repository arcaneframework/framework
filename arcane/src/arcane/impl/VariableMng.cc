// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* VariableMng.cc                                              (C) 2000-2025 */
/*                                                                           */
/* Classe gérant l'ensemble des variables.                                   */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/impl/internal/VariableMng.h"

#include "arcane/utils/Deleter.h"
#include "arcane/utils/StringBuilder.h"
#include "arcane/utils/FatalErrorException.h"
#include "arcane/utils/PlatformUtils.h"
#include "arcane/utils/JSONWriter.h"
#include "arcane/utils/OStringStream.h"
#include "arcane/utils/Math.h"

#include "arcane/core/ArcaneException.h"
#include "arcane/core/VarRefEnumerator.h"
#include "arcane/core/IModule.h"
#include "arcane/core/ISubDomain.h"
#include "arcane/core/IObservable.h"
#include "arcane/core/VariableInfo.h"
#include "arcane/core/VariableBuildInfo.h"
#include "arcane/core/VariableFactoryRegisterer.h"
#include "arcane/core/IVariableFactory.h"
#include "arcane/core/ItemGroup.h"
#include "arcane/core/IMesh.h"
#include "arcane/core/MeshHandle.h"
#include "arcane/core/IParallelMng.h"
#include "arcane/core/ITimeLoopMng.h"
#include "arcane/core/IEntryPoint.h"
#include "arcane/core/Properties.h"
#include "arcane/core/VariableStatusChangedEventArgs.h"

#include "arcane/impl/VariableUtilities.h"
#include "arcane/impl/internal/VariableSynchronizerMng.h"

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
, m_variable_io_writer_mng(new VariableIOWriterMng(this))
, m_variable_io_reader_mng(new VariableIOReaderMng(this))
, m_variable_synchronizer_mng(new VariableSynchronizerMng(this))
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
  delete m_variable_synchronizer_mng;

  delete m_variable_io_reader_mng;
  delete m_variable_io_writer_mng;
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
  m_time_stats = m_parallel_mng->timeStats();

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

  m_variable_synchronizer_mng->initialize();
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
    // For partial variable: if exists already must be defined on the same group, otherwise fatal
    if (infos.isPartial()) {
      if (infos.itemGroupName() != var->itemGroupName())
        throw BadPartialVariableItemGroupNameException(A_FUNCINFO, var, infos.itemGroupName());
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
  if (var.property() & (IVariable::PInShMem)) {
    bool is_all_no_dump = true;
    // Si certains processus ont NoDump et pas d'autres, on dump.
    if (var.itemKind() == IK_Unknown) {
      is_all_no_dump = var.variableMng()->parallelMng()->reduce(MessagePassing::ReduceMin, (var.property() & (IVariable::PNoDump | IVariable::PTemporary)));
    }
    else {
      is_all_no_dump = var.meshHandle().mesh()->parallelMng()->reduce(MessagePassing::ReduceMin, (var.property() & (IVariable::PNoDump | IVariable::PTemporary)));
    }
    if (is_all_no_dump) {
      return false;
    }
  }
  else {
    bool no_dump = var.property() & (IVariable::PNoDump | IVariable::PTemporary);
    if (no_dump)
      return false;
  }
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
  m_variable_io_writer_mng->writeCheckpoint(service);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void VariableMng::
writePostProcessing(IPostProcessorWriter* post_processor)
{
  m_variable_io_writer_mng->writePostProcessing(post_processor);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void VariableMng::
writeVariables(IDataWriter* writer,const VariableCollection& vars)
{
  m_variable_io_writer_mng->writeVariables(writer,vars,false);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void VariableMng::
writeVariables(IDataWriter* writer,IVariableFilter* filter)
{
  m_variable_io_writer_mng->writeVariables(writer,filter,false);
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

void VariableMng::
readCheckpoint(ICheckpointReader* service)
{
  m_variable_io_reader_mng->readCheckpoint(service);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void VariableMng::
readCheckpoint(const CheckpointReadInfo& infos)
{
  m_variable_io_reader_mng->readCheckpoint(infos);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void VariableMng::
readVariables(IDataReader* reader,IVariableFilter* filter)
{
  m_variable_io_reader_mng->readVariables(reader,filter);
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

IVariableSynchronizerMng* VariableMng::
synchronizerMng() const
{
  return m_variable_synchronizer_mng;
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
  if (subDomain()->defaultMesh())
    nb_cell = subDomain()->defaultMesh()->allCells().size();
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
