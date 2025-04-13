// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* Variable.cc                                                 (C) 2000-2025 */
/*                                                                           */
/* Classe gérant une variable.                                               */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_VARIABLE_CC
#define ARCANE_VARIABLE_CC
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/List.h"
#include "arcane/utils/FatalErrorException.h"
#include "arcane/utils/OStringStream.h"
#include "arcane/utils/Iterator.h"
#include "arcane/utils/Iostream.h"
#include "arcane/utils/String.h"
#include "arcane/utils/ITraceMng.h"
#include "arcane/utils/PlatformUtils.h"
#include "arcane/utils/IStackTraceService.h"
#include "arcane/utils/MemoryAccessInfo.h"
#include "arcane/utils/NotImplementedException.h"
#include "arcane/utils/ScopedPtr.h"
#include "arcane/utils/StringBuilder.h"
#include "arcane/utils/MemoryView.h"

#include "arcane/core/ItemGroupObserver.h"
#include "arcane/core/Variable.h"
#include "arcane/core/VarRefEnumerator.h"
#include "arcane/core/IVariableAccessor.h"
#include "arcane/core/ItemGroup.h"
#include "arcane/core/IMesh.h"
#include "arcane/core/IItemFamily.h"
#include "arcane/core/ISubDomain.h"
#include "arcane/core/VariableInfo.h"
#include "arcane/core/ISerializer.h"
#include "arcane/core/VariableBuildInfo.h"
#include "arcane/core/VariableComputeFunction.h"
#include "arcane/core/CommonVariables.h"
#include "arcane/core/Observable.h"
#include "arcane/core/IVariableMng.h"
#include "arcane/core/IDataReader.h"
#include "arcane/core/IDataWriter.h"
#include "arcane/core/IParallelMng.h"
#include "arcane/core/VariableDependInfo.h"
#include "arcane/core/IParallelReplication.h"
#include "arcane/core/VariableMetaData.h"
#include "arcane/core/IMeshMng.h"
#include "arcane/core/MeshHandle.h"
#include "arcane/core/VariableComparer.h"
#include "arcane/core/datatype/DataAllocationInfo.h"
#include "arcane/core/internal/IItemFamilyInternal.h"
#include "arcane/core/internal/IVariableMngInternal.h"
#include "arcane/core/internal/IVariableInternal.h"
#include "arcane/core/internal/IDataInternal.h"

#include <map>
#include <set>
#include <atomic>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*
 * \brief Partie privée d'une variable.
 */
class VariablePrivate
: public IVariableInternal
{
 public:

  VariablePrivate(const VariableBuildInfo& v, const VariableInfo& vi, Variable* var);

 public:

  static std::atomic<Int64> modified_time_global_value;

 public:

  ISubDomain* m_sub_domain = nullptr;
  IDataFactoryMng* m_data_factory_mng = nullptr;
  MeshHandle m_mesh_handle; //!< Maillage (peut être nul)
  Ref<IData> m_data; //!< Données de la variable
  ItemGroup m_item_group; //!< Groupe d'entité sur lequel est associé la variable
  IItemFamily* m_item_family = nullptr; //!< Familly d'entité (peut être nul)
  VariableInfo m_infos; //!< Infos caractéristiques de la variable
  int m_property = 0; //!< Propriétés de la variable
  bool m_is_partial = false; //!< Vrai si la variable est partielle
  bool m_need_property_update = false;
  bool m_is_used = false; //!< Etat d'utilisation de la variable
  bool m_has_valid_data = false; //!< Vrai si les données sont valide
  Real m_last_update_time = 0.0; //!< Temps physique de la dernière mise à jour
  VariableRef* m_first_reference = nullptr; //! Première référence sur la variable
  Integer m_nb_reference = 0;
  UniqueArray<VariableDependInfo> m_depends; //!< Liste des dépendances de cette variable
  Int64 m_modified_time = 0; //!< Tag de la dernière modification
  ScopedPtrT<IVariableComputeFunction> m_compute_function; //!< Fonction de calcul
  AutoDetachObservable m_write_observable; //!< Observable en écriture
  AutoDetachObservable m_read_observable; //!< Observable en lecture
  AutoDetachObservable m_on_size_changed_observable; //!< Observable en redimensionnement
  std::map<String,String> m_tags; //!< Liste des tags
  bool m_has_recursive_depend = true; //!< Vrai si les dépendances sont récursives
  bool m_want_shrink = false;
  Variable* m_variable = nullptr; //!< Variable associée

 public:

  /*!
   * \brief Sérialise le `hashid`.
   *
   * Lors de la désérialisation, vérifie que le `hashid` est correctement
   * et si ce n'est pas le cas renvoie une exception.
   */
  void serializeHashId(ISerializer* sbuf)
  {
    switch(sbuf->mode()){
    case ISerializer::ModeReserve:
      sbuf->reserveSpan(eBasicDataType::Byte,HASHID_SIZE);
      break;
    case ISerializer::ModePut:
      sbuf->putSpan(Span<const Byte>(m_hash_id,HASHID_SIZE));
      break;
    case ISerializer::ModeGet:
      {
        Byte read_hash_id_buf[HASHID_SIZE];
        Span<Byte> read_span(read_hash_id_buf,HASHID_SIZE);
        sbuf->getSpan(read_span);
        Span<const Byte> ref_span(m_hash_id,HASHID_SIZE);
        if (ref_span!=Span<const Byte>(read_span))
          ARCANE_FATAL("Bad hashid for variable name='{0}'\n"
                       "  expected_hash_id='{1}'\n"
                       "  hash_id         ='{2}'\n"
                       " This may be due to incoherence in variable list (order) between ranks"
                       " during serialization",
                       m_infos.fullName(),String(ref_span),String(read_span));
      }
      break;
    }
  }

 public:

  //!@{ \name Implémentation de IVariableInternal
  String computeComparisonHashCollective(IHashAlgorithm* hash_algo, IData* sorted_data) override;
  void changeAllocator(const MemoryAllocationOptions& alloc_info) override;
  void resize(const VariableResizeArgs& resize_args) override;
  //!@}

 private:

  static const int HASHID_SIZE = 64;
  /*!
   * \brief hash de la variable pour vérifier la cohérence de la sérialisation.
   *
   * Les 16 premiers octets sont le hash du nom au format hexadécimal (issu d'un Int64)
   * et les suivants sont le nom complet (fullName()), éventuellement tronqué, de la variable.
   * Les éventuels caractères restants sont des '~'.
   */
  Byte m_hash_id[HASHID_SIZE];

  void _setHashId()
  {
    constexpr Int64 hashid_hexa_length = 16;
    constexpr Int64 name_length = HASHID_SIZE - hashid_hexa_length;
    Span<Byte> hash_id(m_hash_id,HASHID_SIZE);
    hash_id.fill('~');
    const String& full_name = m_infos.fullName();
    Int64 hash_value = IntegerHashFunctionT<StringView>::hashfunc(full_name.view());
    Convert::toHexaString(hash_value,hash_id);
    Span<const Byte> bytes = full_name.bytes();
    if (bytes.size()>name_length)
      bytes = bytes.subspan(0,name_length);
    auto hash_id2 = hash_id.subspan(hashid_hexa_length,name_length);
    hash_id2.copy(bytes);
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

std::atomic<Int64> VariablePrivate::modified_time_global_value = 1;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Int64 IVariable::
incrementModifiedTime()
{
  Int64 v = VariablePrivate::modified_time_global_value;
  ++VariablePrivate::modified_time_global_value;
  return v;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

VariablePrivate::
VariablePrivate(const VariableBuildInfo& v, const VariableInfo& vi, Variable* var)
: m_sub_domain(v._subDomain())
, m_data_factory_mng(v.dataFactoryMng())
, m_mesh_handle(v.meshHandle())
, m_infos(vi)
, m_property(v.property())
, m_is_partial(vi.isPartial())
, m_variable(var)
{
  _setHashId();
  m_infos.setDefaultItemGroupName();

  // Pour test uniquement
  if (!platform::getEnvironmentVariable("ARCANE_NO_RECURSIVE_DEPEND").null())
    m_has_recursive_depend = false;

  // Pour teste de libération mémoire.
  {
    String str = platform::getEnvironmentVariable("ARCANE_VARIABLE_SHRINK_MEMORY");
    if (str=="1")
      m_want_shrink = true;
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

//! Observer des évènements sur ItemGroup sous-jacent.
class ItemGroupPartialVariableObserver
: public IItemGroupObserver
{
 public:

  explicit ItemGroupPartialVariableObserver(IVariable* var)
  : m_var(var)
  {
    ARCANE_ASSERT((m_var),("Variable pointer null")); 
    
    if(var->itemGroup().isAllItems()) 
      ARCANE_FATAL("No observer should be attached on all items group");
  }

  void executeExtend(const Int32ConstArrayView* info) override
  {
    const Int32ConstArrayView & new_ids = *info;
    if (new_ids.empty())
      return;
    ItemGroup group = m_var->itemGroup();
    SharedPtrT<GroupIndexTable> id_to_index = group.localIdToIndex();

    const Integer old_size = id_to_index->size();
    const Integer group_size = group.size();
    if (group_size != (old_size+new_ids.size()))
      ARCANE_FATAL("Inconsitent extended size");
    m_var->resizeFromGroup();
    //id_to_index->update();
  }

  void executeReduce(const Int32ConstArrayView* info) override
  {
    // contient la liste des localids des items supprimés dans l'ancien groupe
    const Int32ConstArrayView & removed_lids = *info; 
    if (removed_lids.empty())
      return;
    ItemGroup group = m_var->itemGroup();
    SharedPtrT<GroupIndexTable> id_to_index = group.localIdToIndex();
    
    const Integer old_size = id_to_index->size();
    const Integer group_size = group.size();

    if (group_size != (old_size-removed_lids.size()))
      ARCANE_FATAL("Inconsitent reduced size {0} vs {1}",group_size,old_size);
    [[maybe_unused]] ItemVectorView view = group.view();
    Int32UniqueArray source;
    Int32UniqueArray destination;
    source.reserve(group_size);
    destination.reserve(group_size);
    for(Integer i=0,index=0,removed_index=0; i<old_size ;++i) {
      if (removed_index < removed_lids.size() && 
          id_to_index->keyLocalId(i) == removed_lids[removed_index]) {
        ++removed_index;
      }
      else {
        ARCANE_ASSERT((id_to_index->keyLocalId(i) == view[index].localId()),
                      ("Inconsistent key (pos=%d,key=%d) vs (pos=%d,key=%d)",
                       i,id_to_index->keyLocalId(i),index,view[index].localId()));
        if (i != index) {
          destination.add(index);
          source.add(i);
        }
        ++index;
      }
    }
    m_var->copyItemsValues(source,destination);
    m_var->resizeFromGroup();
  }

  void executeCompact(const Int32ConstArrayView* info) override
  {
    const Int32ConstArrayView & ids = *info;
    if (ids.empty()) return;
    ItemGroup group = m_var->itemGroup();
    SharedPtrT<GroupIndexTable> id_to_index = group.localIdToIndex();
    m_var->compact(*info);
    //id_to_index->compact(info);
  }

  void executeInvalidate() override
  {
    ItemGroup group = m_var->itemGroup();
    SharedPtrT<GroupIndexTable> id_to_index = group.localIdToIndex();
    m_var->resizeFromGroup();
    //id_to_index->update();
  }

  bool needInfo() const override { return true; }

 private:

  IVariable* m_var = nullptr;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Variable::
Variable(const VariableBuildInfo& v,const VariableInfo& vi)
: TraceAccessor(v.traceMng())
, m_p(new VariablePrivate(v, vi, this))
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Variable::
~Variable()
{
  //NOTE: si la variable possède un groupe, c'est le IVariableMng
  // qui supprime la référence de cette variable sur le groupe
  delete m_p;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

bool Variable::
_hasReference() const
{
  return m_p->m_first_reference;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void Variable::
addVariableRef(VariableRef* ref)
{
  _checkSetProperty(ref);
  ++m_p->m_nb_reference;
  ref->setNextReference(m_p->m_first_reference);
  if (m_p->m_first_reference){
    VariableRef* _list = m_p->m_first_reference;
    if (_list->previousReference())
      _list->previousReference()->setNextReference(ref);
    _list->setPreviousReference(ref);
  }
  else{
    ref->setPreviousReference(0);
  }
  m_p->m_first_reference = ref;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void Variable::
removeVariableRef(VariableRef* ref)
{
  {
    VariableRef* tmp = ref;
    if (tmp->previousReference())
      tmp->previousReference()->setNextReference(tmp->nextReference());
    if (tmp->nextReference())
      tmp->nextReference()->setPreviousReference(tmp->previousReference());
    if (m_p->m_first_reference==tmp)
      m_p->m_first_reference = m_p->m_first_reference->nextReference();
  }
  // La référence peut être utilisée par la suite donc il ne faut pas oublier
  // de supprimer le précédent et le suivant.
  ref->setNextReference(0);
  ref->setPreviousReference(0);

  --m_p->m_nb_reference;
  _checkSetProperty(ref);

  // Lorsqu'il n'y a plus de références sur cette variable, le signale au
  // gestionnaire de variable, sauf s'il s'agit d'une variable persistante
  if (!_hasReference()){
    bool is_persistant = property() & IVariable::PPersistant;
    if (!is_persistant){
      //m_p->m_trace->info() << " REF PROPERTY name=" << name() << " " << ref->referenceProperty();
      _removeMeshReference();
      ISubDomain* sd = m_p->m_sub_domain;
      IVariableMng* vm = sd->variableMng();
      vm->_internalApi()->removeVariable(this);
    }
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

VariableRef* Variable::
firstReference() const
{
  return m_p->m_first_reference;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void Variable::
_checkSetProperty(VariableRef* ref)
{
  // Garantie que la propriété est correctement mise à jour avec la valeur
  // de la seule référence.
  if (!_hasReference()){
    m_p->m_property = ref->referenceProperty();
    m_p->m_need_property_update = false;
  }
  else
    m_p->m_need_property_update = true;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Integer Variable::
nbReference() const
{
  return m_p->m_nb_reference;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ISubDomain* Variable::
subDomain()
{
  return m_p->m_sub_domain;
}


/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

IVariableMng* Variable::
variableMng() const
{
  return m_p->m_sub_domain->variableMng();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

String Variable::
name() const
{
  return m_p->m_infos.localName();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

String Variable::
fullName() const
{
  return m_p->m_infos.fullName();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

String Variable::
itemFamilyName() const
{
  return m_p->m_infos.itemFamilyName();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

String Variable::
itemGroupName() const
{
  return m_p->m_infos.itemGroupName();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

String Variable::
meshName() const
{
  return m_p->m_infos.meshName();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

eDataType Variable::
dataType() const
{
  return m_p->m_infos.dataType();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \todo traiter le cas PSubDomainPrivate.
 */
int Variable::
property() const
{
  if (!m_p->m_need_property_update)
    return m_p->m_property;

  // Les propriétés de la variable dépendent de ce que chaque
  // référence souhaite et il faut les remettre à jour lorsque ces
  // dernières changent.
  // Par exemple, si toutes les références sont PNoDump et qu'une seule
  // ne l'est pas, la variable ne doit pas l'être.
  m_p->m_need_property_update = false;

  bool want_dump = false;
  bool want_sync = false;
  bool want_replica_sync = false;
  bool sub_domain_depend = false;
  bool execution_depend = false;
  bool want_private = false;
  bool want_restore = false;
  bool want_notemporary = false;
  bool want_exchange = false;
  bool want_persistant = false;

  int property = 0;
  for( VarRefEnumerator i(this); i.hasNext(); ++i ){
    VariableRef* vref = *i;
    int p = vref->referenceProperty();
    if ( ! (p & IVariable::PNoDump) )
      want_dump = true;
    if ( ! (p & IVariable::PNoNeedSync) )
      want_sync = true;
    if ( ! (p & IVariable::PNoReplicaSync) )
      want_replica_sync = true;
    if ( (p & IVariable::PSubDomainDepend) )
      sub_domain_depend = true;
    if ( (p & IVariable::PExecutionDepend) )
      execution_depend = true;
    if ( (p & IVariable::PPersistant) )
      want_persistant = true;
    if ( (p & IVariable::PPrivate) )
      want_private = true;
    if ( ! (p & IVariable::PNoRestore) )
      want_restore = true;
    if ( ! (p & IVariable::PNoExchange) )
      want_exchange = true;
    if ( ! (p & IVariable::PTemporary) )
      want_notemporary = true;
  }

  if (!want_dump)
    property |= IVariable::PNoDump;
  if (!want_sync)
    property |= IVariable::PNoNeedSync;
  if (!want_replica_sync)
    property |= IVariable::PNoReplicaSync;
  if (sub_domain_depend)
    property |= IVariable::PSubDomainDepend;
  if (execution_depend)
    property |= IVariable::PExecutionDepend;
  if (want_private)
    property |= IVariable::PPrivate;
  if (want_persistant)
    property |= IVariable::PPersistant;
  if (!want_restore)
    property |= IVariable::PNoRestore;
  if (!want_exchange)
    property |= IVariable::PNoExchange;
  if (!want_notemporary)
    property |= IVariable::PTemporary;

  m_p->m_property = property;
  return m_p->m_property;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void Variable::
notifyReferencePropertyChanged()
{
  m_p->m_need_property_update = true;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void Variable::
setUsed(bool is_used)
{
  if (m_p->m_is_used==is_used)
    return;

  m_p->m_is_used = is_used;

  eItemKind ik = itemKind();
  
  if (m_p->m_is_used){
    if (m_p->m_item_group.null() && ik!=IK_Unknown){
      _checkSetItemFamily();
      _checkSetItemGroup();
      // Attention à ne pas reinitialiser les valeurs lorsque ces dernières
      // sont valides, ce qui est le cas par exemple après une protection.
      if (!m_p->m_has_valid_data){
        resizeFromGroup();
        // Historiquement on remplissait dans tous les cas la variable avec le
        // constructeur par défaut
        // de la donnée en appelant systématiquement fillDefautt(). Cependant,
        // ce n'était pas le comportement souhaité qui doit être celui défini par
        // getGlobalDataInitialisationPolicy() (dans DataTypes.h).
        // On ne le fait maintenant que si le mode d'initialisation est égal
        // à DIP_Legacy. Ce mode doit à terme disparaître.
        if (getGlobalDataInitialisationPolicy()==DIP_Legacy)
          m_p->m_data->fillDefault();
        m_p->m_has_valid_data = true;
      }
    }
  }
  else{
    _removeMeshReference();
    if (ik==IK_Unknown)
      resize(0);
    else
      resizeFromGroup();
    // Indique que les valeurs ne sont plus valides
    m_p->m_has_valid_data = false;
  }

  for( VarRefEnumerator i(this); i.hasNext(); ++i ){
    VariableRef* ref = *i;
    ref->internalSetUsed(m_p->m_is_used);
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void Variable::
_removeMeshReference()
{
  IItemFamily* family = m_p->m_item_family;
  if (family)
    family->_internalApi()->removeVariable(this);
  
  if (isPartial())
    m_p->m_item_group.internal()->detachObserver(this);
  
  m_p->m_item_group = ItemGroup();
  m_p->m_item_family = 0;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

bool Variable::
isUsed() const
{
  return m_p->m_is_used;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace
{

String _buildVariableFullType(const IVariable* var)
{
  StringBuilder full_type_b;
  full_type_b = dataTypeName(var->dataType());
  full_type_b += ".";
  full_type_b += itemKindName(var->itemKind());
  full_type_b += ".";
  full_type_b += var->dimension();
  full_type_b += ".";
  full_type_b += var->multiTag();
  if (var->isPartial())
    full_type_b += ".Partial";
  return full_type_b.toString();
}

}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

VariableMetaData* Variable::
_createMetaData() const
{
  auto vmd = new VariableMetaData(name(),meshName(),itemFamilyName(),
                                  itemGroupName(),isPartial());
  vmd->setFullType(_buildVariableFullType(this));
  vmd->setMultiTag(String::fromNumber(multiTag()));
  vmd->setProperty(property());
  return vmd;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

VariableMetaData* Variable::
createMetaData() const
{
  return _createMetaData();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Ref<VariableMetaData> Variable::
createMetaDataRef() const
{
  return makeRef(_createMetaData());
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void Variable::
syncReferences()
{
  //cout << "** SYNC REFERENCE N=" << m_p->m_nb_reference << " F=" << m_p->m_first_reference << '\n';
  for( VarRefEnumerator i(this); i.hasNext(); ++i ){
    VariableRef* ref = *i;
    //cout << "** SYNC REFERENCE V=" << ref << '\n';
    ref->updateFromInternal();
  }
  // Il faut le faire après la mise à jour des références
  // car les observateurs peuvent lire les valeurs via une référence
  onSizeChangedObservable()->notifyAllObservers();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Int32 Variable::
checkIfSync(int max_print)
{
  VariableComparerArgs compare_args;
  compare_args.setCompareMode(VariableComparerArgs::eCompareMode::Sync);
  compare_args.setMaxPrint(max_print);
  compare_args.setCompareGhost(true);
  VariableComparerResults results = _compareVariable(compare_args);
  return results.nbDifference();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Integer Variable::
checkIfSameOnAllReplica(Integer max_print)
{
  //TODO: regarder si la variable est utilisée.
  IMesh* mesh = this->mesh();
  IParallelMng* pm = (mesh) ? mesh->parallelMng() : subDomain()->parallelMng();
  IParallelReplication* pr = pm->replication();
  if (!pr->hasReplication())
    return 0;
  return _checkIfSameOnAllReplica(pr->replicaParallelMng(),max_print);
}

Int32 Variable::
checkIfSame(IDataReader* reader, Integer max_print, bool compare_ghost)
{
  VariableComparerArgs compare_args;
  compare_args.setMaxPrint(max_print);
  compare_args.setCompareGhost(compare_ghost);
  compare_args.setDataReader(reader);
  VariableComparerResults r = _compareVariable(compare_args);
  return r.nbDifference();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

IMesh* Variable::
mesh() const
{
  if (m_p->m_mesh_handle.hasMesh())
    return m_p->m_mesh_handle.mesh();
  return nullptr;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

MeshHandle Variable::
meshHandle() const
{
  return m_p->m_mesh_handle;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ItemGroup Variable::
itemGroup() const
{
  return m_p->m_item_group;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

eItemKind Variable::
itemKind() const
{
  return m_p->m_infos.itemKind();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Integer Variable::
dimension() const
{
  return m_p->m_infos.dimension();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Integer Variable::
multiTag() const
{
  return m_p->m_infos.multiTag();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

bool Variable::
isPartial() const
{
  return m_p->m_is_partial;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

IItemFamily* Variable::
itemFamily() const
{
  return m_p->m_item_family;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void Variable::
_setData(const Ref<IData>& data)
{
  m_p->m_data = data;
  if (!data.get()){
    ARCANE_FATAL("Invalid data: name={0} datatype={1} dimension={2} multitag={3}",
                 m_p->m_infos.fullName(),m_p->m_infos.dataType(),
                 m_p->m_infos.dimension(),m_p->m_infos.multiTag());
  }
  data->setName(m_p->m_infos.fullName());
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void Variable::
_setValidData(bool valid_data)
{
  m_p->m_has_valid_data = valid_data;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

bool Variable::
_hasValidData() const
{
  return m_p->m_has_valid_data;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void Variable::
_setProperty(int property)
{
  m_p->m_property |= property;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

IDataFactoryMng* Variable::
dataFactoryMng() const
{
  return m_p->m_data_factory_mng;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void Variable::
serialize(ISerializer* sbuffer,Int32ConstArrayView ids,IDataOperation* operation)
{
  debug(Trace::High) << "Serialize (partial) variable name=" << fullName();
  m_p->serializeHashId(sbuffer);
  m_p->m_data->serialize(sbuffer,ids,operation);
  // En mode lecture, les données sont modifiées
  if (sbuffer->mode()==ISerializer::ModeGet)
    syncReferences();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void Variable::
serialize(ISerializer* sbuffer,IDataOperation* operation)
{
  debug(Trace::High) << "Serialize (full) variable name=" << fullName();

  m_p->serializeHashId(sbuffer);
  m_p->m_data->serialize(sbuffer,operation);
  // En mode lecture, les données sont modifiées
  if (sbuffer->mode()==ISerializer::ModeGet)
    syncReferences();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void Variable::
_resize(const VariableResizeArgs& resize_args)
{
  eItemKind ik = itemKind();
  if (ik!=IK_Unknown){
    ARCANE_FATAL("This call is invalid for item variable. Use resizeFromGroup() instead");
  }
  _internalResize(resize_args);
  syncReferences();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void Variable::
resize(Integer new_size)
{
  _resize(VariableResizeArgs(new_size));
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void Variable::
resizeFromGroup()
{
  eItemKind ik = itemKind();
  if (ik==IK_Unknown)
    return;
  Integer new_size = 0;
  IItemFamily* family = m_p->m_item_family;
  if (family){
    if (m_p->m_item_group.isAllItems())
      new_size = m_p->m_item_family->maxLocalId();
    else
      new_size = m_p->m_item_group.size();
  }
  else{
    ItemGroup group = m_p->m_item_group;
    if (!group.null()){
      ARCANE_FATAL("Variable '{0}' has group but no family",fullName());
    }
  }
  debug(Trace::High) << "Variable::resizeFromGroup() var='" << fullName()
                     << "' with " << new_size << " items "
                     << " this=" << this;
  _internalResize(VariableResizeArgs(new_size,new_size/20));
  syncReferences();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void Variable::
_checkSetItemFamily()
{
  if (m_p->m_item_family || !m_p->m_item_group.null())
    return;

  if (m_p->m_mesh_handle.isNull())
    m_p->m_mesh_handle = m_p->m_sub_domain->meshMng()->findMeshHandle(m_p->m_infos.meshName());

  IMesh* mesh = m_p->m_mesh_handle.mesh();
  if (!mesh)
    ARCANE_FATAL("No mesh named '{0}' exists for variable '{1}'",m_p->m_infos.meshName(),name());

  eItemKind ik = itemKind();

  IItemFamily* family = 0;
  const String& family_name = m_p->m_infos.itemFamilyName();
  if (ik==IK_Particle || ik==IK_DoF){
    if (family_name.null()){
      ARCANE_FATAL("family name not specified for variable {0}",name());
    }
    family = mesh->findItemFamily(ik,family_name,true);
  }
  else{
    family = mesh->itemFamily(ik);
  }

  if (family && family->itemKind()!=itemKind())
    ARCANE_FATAL("Bad family kind '{0}' '{1}'",family->itemKind(),itemKind());

  if (family && family->name()!=itemFamilyName())
    ARCANE_FATAL("Incoherent family name. var={0} from_type={1} given={2}",
                 name(),family->name(),itemFamilyName());
  
  if (!family)
    ARCANE_FATAL("Family not found");

  if (isPartial() && !family->hasUniqueIdMap())
    ARCANE_FATAL("Cannot have partial variable for a family without unique id map");

  m_p->m_item_family = family;
  debug(Trace::High) << "Variable::setItemFamily() name=" << name()
                     << " family=" << family
                     << " familyname='" << family_name << "'";
  family->_internalApi()->addVariable(this);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void Variable::
_checkSetItemGroup()
{
  if (!m_p->m_item_group.null())
    return;
  const String& group_name = m_p->m_infos.itemGroupName();
  //info() << " CHECK SET GROUP var=" << name() << " group=" << group_name;
  if (group_name.null()){
    m_p->m_item_group = m_p->m_item_family->allItems();
  }
  else
    m_p->m_item_group = m_p->m_item_family->findGroup(group_name,true);

  ItemGroupImpl * internal = m_p->m_item_group.internal();
  // (HP) TODO: faut il garder ce controle hérité de l'ancienne implémentation de addVariable
  if (internal->parent() && (mesh()->parallelMng()->isParallel() && internal->isOwn()))
    ARCANE_FATAL("Cannot add variable ({0}) on a own group (name={1})",
                 fullName(),internal->name());
  if (isPartial()) {
    if (group_name.empty())
      ARCANE_FATAL("Cannot create a partial variable with an empty item_group_name");
    debug(Trace::High) << "Attach ItemGroupPartialVariableObserver from " << fullName() 
                       << " to " << m_p->m_item_group.name();
    internal->attachObserver(this,new ItemGroupPartialVariableObserver(this));
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

IObservable* Variable::
writeObservable()
{
  return &(m_p->m_write_observable);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

IObservable* Variable::
readObservable()
{
  return &(m_p->m_read_observable);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

IObservable* Variable::
onSizeChangedObservable()
{
  return &(m_p->m_on_size_changed_observable);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void Variable::
update()
{
  update(DPT_PreviousTime);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void Variable::
update(Real wanted_time)
{
  if (m_p->m_last_update_time<wanted_time){
    for( Integer k=0,n=m_p->m_depends.size(); k<n; ++k ){
      VariableDependInfo& vdi = m_p->m_depends[k];
      if (vdi.dependType()==DPT_PreviousTime)
        vdi.variable()->update(wanted_time);
    }
  }
    
  if (m_p->m_has_recursive_depend){
    for( Integer k=0,n=m_p->m_depends.size(); k<n; ++k ){
      VariableDependInfo& vdi = m_p->m_depends[k];
      if (vdi.dependType()==DPT_CurrentTime)
        vdi.variable()->update(m_p->m_last_update_time);
    }
  }

  bool need_update = false;
  Int64 modified_time = m_p->m_modified_time;
  for( Integer k=0,n=m_p->m_depends.size(); k<n; ++k ){
    VariableDependInfo& vdi = m_p->m_depends[k];
    Int64 mt = vdi.variable()->modifiedTime();
    if (mt>modified_time){
      need_update = true;
      break;
    }
  }
  if (need_update){
    IVariableComputeFunction* cf = m_p->m_compute_function.get();
    //msg->info() << "Need Compute For Variable <" << name() << "> " << cf;
    if (cf){
      //msg->info() << "Compute For Variable <" << name() << ">";
      cf->execute();
    }
    else{
      ARCANE_FATAL("No compute function for variable '{0}'",fullName());
    }
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void Variable::
setUpToDate()
{
  m_p->m_last_update_time = subDomain()->commonVariables().globalTime();
  m_p->m_modified_time = IVariable::incrementModifiedTime();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Int64 Variable::
modifiedTime()
{
  return m_p->m_modified_time;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void Variable::
addDepend(IVariable* var,eDependType dt)
{
  m_p->m_depends.add(VariableDependInfo(var,dt,TraceInfo()));
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void Variable::
addDepend(IVariable* var,eDependType dt,const TraceInfo& tinfo)
{
  m_p->m_depends.add(VariableDependInfo(var,dt,tinfo));
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void Variable::
removeDepend(IVariable* var)
{
  ARCANE_UNUSED(var);
  throw NotImplementedException(A_FUNCINFO);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void Variable::
setComputeFunction(IVariableComputeFunction* v)
{
  m_p->m_compute_function = v;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

IVariableComputeFunction* Variable::
computeFunction()
{
  return m_p->m_compute_function.get();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void Variable::
dependInfos(Array<VariableDependInfo>& infos)
{
  for( Integer k=0,n=m_p->m_depends.size(); k<n; ++k ){
    VariableDependInfo& vdi = m_p->m_depends[k];
    infos.add(vdi);
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void Variable::
addTag(const String& tagname,const String& tagvalue)
{
  m_p->m_tags[tagname] = tagvalue;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void Variable::
removeTag(const String& tagname)
{
  m_p->m_tags.erase(tagname);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

bool Variable::
hasTag(const String& tagname)
{
  return m_p->m_tags.find(tagname)!=m_p->m_tags.end();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

String Variable::
tagValue(const String& tagname)
{
  std::map<String,String>::const_iterator i = m_p->m_tags.find(tagname);
  if (i==m_p->m_tags.end())
    return String();
  return i->second;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void Variable::
notifyEndRead()
{
  setUpToDate();
  syncReferences();
  readObservable()->notifyAllObservers();
  _setValidData(true);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void Variable::
notifyBeginWrite()
{
  writeObservable()->notifyAllObservers();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void Variable::
read(IDataReader* reader)
{
  reader->read(this,data());
  notifyEndRead();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void Variable::
write(IDataWriter* writer)
{
  notifyBeginWrite();
  writer->write(this,data());
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void Variable::
changeGroupIds(Int32ConstArrayView old_to_new_ids)
{
  ARCANE_UNUSED(old_to_new_ids);
 // pH: default implementation since this method is not yet official
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Vérifie qu'il est possible d'échanger les valeurs de l'instance
 * avec celle de \a rhs.
 *
 * Cette méthode étant appelée par une classe dérivée, on est sur que \a rhs
 * est du même type C++ que l'instance et donc il n'y a pas besoin de
 * vérifier par exemple que les dimensions ou le type des données sont les
 * mêmes. Pour que l'échange soit valide, il faut que le maillage, la famille
 * et le groupe soit les mêmes. Pour cela, il suffit de vérifier que le
 * groupe est le même.
 */
void Variable::
_checkSwapIsValid(Variable* rhs)
{
  if (!m_p->m_is_used)
    ARCANE_FATAL("Can not swap variable values for unused variable (instance)");
  if (!rhs->m_p->m_is_used)
    ARCANE_FATAL("Can not swap variable values for unused variable (argument)");
  if (isPartial() || rhs->isPartial())
    ARCANE_FATAL("Can not swap variable values for partial variables");
  if (itemGroup()!=rhs->itemGroup())
    ARCANE_FATAL("Can not swap variable values for variables from different groups");
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

bool Variable::
_wantShrink() const
{
  return m_p->m_want_shrink;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void Variable::
setAllocationInfo(const DataAllocationInfo& v)
{
  data()->setAllocationInfo(v);
  // Il est possible que le changement d'allocation modifie les données
  // allouées. Il faut donc synchroniser les références.
  syncReferences();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

DataAllocationInfo Variable::
allocationInfo() const
{
  return data()->allocationInfo();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

IVariableInternal* Variable::
_internalApi()
{
  return m_p;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

String VariablePrivate::
computeComparisonHashCollective(IHashAlgorithm* hash_algo,
                                IData* sorted_data)
{
  ARCANE_CHECK_POINTER(hash_algo);
  ARCANE_CHECK_POINTER(sorted_data);

  INumericDataInternal* num_data = sorted_data->_commonInternal()->numericData();
  if (!num_data)
    return {};
  if (!m_item_family)
    return {};

  IParallelMng* pm = m_item_family->parallelMng();
  Int32 my_rank = pm->commRank();
  Int32 master_rank = pm->masterIORank();
  ConstMemoryView memory_view = num_data->memoryView();

  UniqueArray<Byte> bytes;

  pm->gatherVariable(Arccore::asSpan<Byte>(memory_view.bytes()).smallView(), bytes, master_rank);

  String hash_string;
  if (my_rank == master_rank) {
    HashAlgorithmValue hash_value;
    hash_algo->computeHash(asBytes(bytes), hash_value);
    hash_string = Convert::toHexaString(asBytes(hash_value.bytes()));
  }
  return hash_string;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void VariablePrivate::
changeAllocator(const MemoryAllocationOptions& mem_options)
{
  INumericDataInternal* dx = m_data->_commonInternal()->numericData();
  if (dx) {
    dx->changeAllocator(mem_options);
    m_variable->syncReferences();
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void VariablePrivate::
resize(const VariableResizeArgs& resize_args)
{
  return m_variable->_resize(resize_args);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
