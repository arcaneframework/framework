// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2023 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ItemGroupImpl.cc                                            (C) 2000-2023 */
/*                                                                           */
/* Implémentation d'un groupe d'entités de maillage.                         */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/ItemGroupImpl.h"

#include "arcane/utils/Iostream.h"
#include "arcane/utils/String.h"
#include "arcane/utils/ITraceMng.h"
#include "arcane/utils/NotImplementedException.h"
#include "arcane/utils/NotSupportedException.h"
#include "arcane/utils/ArgumentException.h"
#include "arcane/utils/Array.h"
#include "arcane/utils/SharedPtr.h"
#include "arcane/utils/MemoryUtils.h"

#include "arcane/core/ItemGroupObserver.h"
#include "arcane/core/IItemFamily.h"
#include "arcane/core/ItemGroup.h"
#include "arcane/core/IMesh.h"
#include "arcane/core/ItemPrinter.h"
#include "arcane/core/VariableTypes.h"
#include "arcane/core/IItemOperationByBasicType.h"
#include "arcane/core/IParallelMng.h"
#include "arcane/core/ItemFunctor.h"
#include "arcane/core/ItemGroupComputeFunctor.h"
#include "arcane/core/IVariableSynchronizer.h"
#include "arcane/core/MeshPartInfo.h"
#include "arcane/core/ParallelMngUtils.h"
#include "arcane/core/datatype/DataAllocationInfo.h"
#include "arcane/core/internal/IDataInternal.h"

#include <algorithm>
#include <set>
#include <map>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \internal
 * \brief Classe d'un groupe nul.
 */
class ItemGroupImplNull
: public ItemGroupImpl
{
 public:

  ItemGroupImplNull() : ItemGroupImpl() {}
  virtual ~ItemGroupImplNull() {} //!< Libére les ressources

 public:

  //! Retourne le nom du groupe
  const String& name() const { return m_name; }
  const String& fullName() const { return m_name; }

 public:

 public:

  virtual void convert(NodeGroup& g) { g = NodeGroup(); }
  virtual void convert(EdgeGroup& g) { g = EdgeGroup(); }
  virtual void convert(FaceGroup& g) { g = FaceGroup(); }
  virtual void convert(CellGroup& g) { g = CellGroup(); }

 public:

 private:

  String m_name;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \internal
 * \brief Implémentation de la classe ItemGroupImpl.
 *
 Le container contenant la liste des entités du groupe est soit une
 variable dans le cas d'une groupe standard, soit un tableau simple
 dans le cas d'un groupe ayant un parent. En effet, les groupes
 ayant des parents sont des groupes générés dynamiquement (par
 exemple le groupe des entités propres) et ne sont donc pas
 toujours présents sur tous les sous-domaines (une variable doit toujours
 exister sur tous les sous-domaines). De plus, leur valeur n'a
 pas besoin d'être sauvée lors d'une protection.

 \todo ajouter notion de groupe générée, avec les propriétés suivantes:
 - ces groupes ne doivent pas être transférés d'un sous-domaine à l'autre
 - ils ne peuvent pas être modifiés directement.
 */
class ItemGroupImplPrivate
{
 public:

  ItemGroupImplPrivate();
  ItemGroupImplPrivate(IItemFamily* family,const String& name);
  ItemGroupImplPrivate(IItemFamily* family,ItemGroupImpl* parent,const String& name);
  virtual ~ItemGroupImplPrivate();

 public:

  const String& name() const { return m_name; }
  const String& fullName() const { return m_full_name; }
  bool null() const { return m_is_null; }
  IMesh* mesh() const { return m_mesh; }
  eItemKind kind() const { return m_kind; }
  Integer maxLocalId() const { return m_item_family->maxLocalId(); }
  ItemInternalList items() const
  {
    if (m_item_family)
      return m_item_family->itemsInternal();
    return m_mesh->itemsInternal(m_kind);
  }
  ItemInfoListView itemInfoListView() const
  {
    if (m_item_family)
      return m_item_family->itemInfoListView();
    return m_mesh->itemFamily(m_kind)->itemInfoListView();
  }

  Int32ArrayView itemsLocalId() { return *m_items_local_id; }
  Int32ConstArrayView itemsLocalId() const { return *m_items_local_id; }
  Int32Array& mutableItemsLocalId() { return *m_items_local_id; }
  VariableArrayInt32* variableItemsLocalid() { return m_variable_items_local_id; }

  Int64 timestamp() const { return m_timestamp; }
  bool isContigous() const { return m_is_contigous; }
  void checkIsContigous();

  void updateTimestamp()
  {
    ++m_timestamp;
    m_is_contigous = false;
  }

  void setNeedRecompute()
  {
    // NOTE: normalement il ne faudrait mettre cette valeur à 'true' que pour
    // les groupes recalculés (qui ont un parent ou pour lequel 'm_compute_functor' n'est
    // pas nul). Cependant, cette méthode est aussi appelé sur le groupe de toutes les entités
    // et peut-être d'autres groupes.
    // Changer ce comportement risque d'impacter pas mal de code donc il faudrait bien vérifier
    // que tout est OK avant de faire cette modification.
    m_need_recompute = true;
  }

 public:

  IMesh* m_mesh; //!< Gestionnare de groupe associé
  IItemFamily* m_item_family; //!< Famille associée
  ItemGroupImpl* m_parent; //! Groupe parent (groupe null si aucun)
  String m_variable_name; //!< Nom de la variable contenant les indices des éléments du groupe
  String m_full_name; //!< Nom complet du groupe.
  bool m_is_null; //!< \a true si le groupe est nul
  eItemKind m_kind; //!< Genre de entités du groupe
  String m_name; //!< Nom du groupe
  bool m_is_own = false; //!< \a true si groupe local.
 private:
  Int64 m_timestamp = -1; //!< Temps de la derniere modification
 public:
  Int64 m_simd_timestamp = -1; //!< Temps de la derniere modification pour le calcul des infos SIMD
  //@{ @name alias locaux à des sous-groupes de m_sub_groups
  ItemGroupImpl* m_own_group = nullptr;        //!< Items owned by the subdomain
  ItemGroupImpl* m_ghost_group = nullptr;      //!< Items not owned by the subdomain
  ItemGroupImpl* m_interface_group = nullptr;  //!< Items on the boundary of two subdomains
  ItemGroupImpl* m_node_group = nullptr;       //!< Groupe des noeuds
  ItemGroupImpl* m_edge_group = nullptr;       //!< Groupe des arêtes
  ItemGroupImpl* m_face_group = nullptr;       //!< Groupe des faces
  ItemGroupImpl* m_cell_group = nullptr;       //!< Groupe des mailles
  ItemGroupImpl* m_inner_face_group = nullptr; //!< Groupe des faces internes
  ItemGroupImpl* m_outer_face_group = nullptr; //!< Groupe des faces externes
  //! AMR
  // FIXME on peut éviter de stocker ces groupes en introduisant des predicats
  // sur les groupes parents
  ItemGroupImpl* m_active_cell_group = nullptr;       //!< Groupe des mailles actives
  ItemGroupImpl* m_own_active_cell_group = nullptr;   //!< Groupe des mailles propres actives
  ItemGroupImpl* m_active_face_group = nullptr;       //!< Groupe des faces actives
  ItemGroupImpl* m_own_active_face_group = nullptr;   //!< Groupe des faces actives propres
  ItemGroupImpl* m_inner_active_face_group = nullptr; //!< Groupe des faces internes actives
  ItemGroupImpl* m_outer_active_face_group = nullptr; //!< Groupe des faces externes actives
  std::map<Integer, ItemGroupImpl*>  m_level_cell_group;        //!< Groupe des mailles de niveau
  std::map<Integer, ItemGroupImpl*> m_own_level_cell_group;    //!< Groupe des mailles propres de niveau

  UniqueArray<ItemGroupImpl*> m_children_by_type; //!< Liste des fils de ce groupe par type d'entité
  //@}
  std::map<String,AutoRefT<ItemGroupImpl>> m_sub_groups; //!< Ensemble de tous les sous-groupes
  bool m_need_recompute = false; //!< Vrai si le groupe doit être recalculé
  bool m_need_invalidate_on_recompute = false; //!< Vrai si l'on doit activer les invalidate observers en cas de recalcul
  bool m_transaction_mode = false; //!< Vrai si le groupe est en mode de transaction directe
  bool m_is_local_to_sub_domain = false; //!< Vrai si le groupe est local au sous-domaine
  IFunctor* m_compute_functor = nullptr; //!< Fonction de calcul du groupe
  bool m_is_all_items = false; //!< Indique s'il s'agit du groupe de toutes les entités

  SharedPtrT<GroupIndexTable> m_group_index_table; //!< Table de hachage du local id des items vers leur position en enumeration
  Ref<IVariableSynchronizer> m_synchronizer; //!< Synchronizer du groupe

  // Anciennement dans DynamicMeshKindInfo
  Int32UniqueArray m_items_index_in_all_group; //! localids -> index (UNIQUEMENT ALLITEMS)
  
  std::map<const void*,IItemGroupObserver*> m_observers; //!< Observers du groupe
  bool m_observer_need_info = false; //!< Synthése de besoin de observers en informations de transition
  void notifyExtendObservers(const Int32ConstArrayView * info);
  void notifyReduceObservers(const Int32ConstArrayView * info);
  void notifyCompactObservers(const Int32ConstArrayView * info);
  void notifyInvalidateObservers();

  void resetSubGroups();

 private:
  UniqueArray<Int32> m_local_buffer{MemoryUtils::getAllocatorForMostlyReadOnlyData()};
  Array<Int32>* m_items_local_id = &m_local_buffer; //!< Liste des numéros locaux des entités de ce groupe
  VariableArrayInt32* m_variable_items_local_id = nullptr;
 private:
  bool m_is_contigous = false; //! Vrai si les localIds sont consécutifs.
 private:

  void _init();
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ItemGroupImplPrivate::
ItemGroupImplPrivate()
: m_mesh(nullptr)
, m_item_family(nullptr)
, m_parent(nullptr)
, m_variable_name()
, m_is_null(true)
, m_kind(IK_Unknown)
{
  _init();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ItemGroupImplPrivate::
ItemGroupImplPrivate(IItemFamily* family,const String& name)
: m_mesh(family->mesh())
, m_item_family(family)
, m_parent(nullptr)
, m_variable_name(String("GROUP_")+family->name()+name)
, m_is_null(false)
, m_kind(family->itemKind())
, m_name(name)
{
  _init();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ItemGroupImplPrivate::
ItemGroupImplPrivate(IItemFamily* family,ItemGroupImpl* parent,const String& name)
: m_mesh(parent->mesh())
, m_item_family(family)
, m_parent(parent)
, m_variable_name(String("GROUP_")+m_item_family->name()+name)
, m_is_null(false)
, m_kind(family->itemKind())
, m_name(name)
{
  _init();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ItemGroupImplPrivate::
~ItemGroupImplPrivate()
{
  // (HP) TODO: vérifier qu'il n'y a plus d'observer à cet instant
  // Ceux des sous-groupes n'ont pas été détruits
  for( const auto& i : m_observers ) {
    delete i.second;
  }
  delete m_variable_items_local_id;
  delete m_compute_functor;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ItemGroupImplPrivate::
_init()
{
  if (m_item_family)
    m_full_name = m_item_family->fullName() + "_" + m_name;
  // Si maillage associé et pas un groupe enfant alors les données du groupe
  // sont conservées dans une variable.
  if (m_mesh && !m_parent){
    int property = IVariable::PSubDomainDepend | IVariable::PPrivate;
    VariableBuildInfo vbi(m_mesh,m_variable_name,property);
    m_variable_items_local_id = new VariableArrayInt32(vbi);
    m_variable_items_local_id->variable()->setAllocationInfo(DataAllocationInfo(eMemoryLocationHint::HostAndDeviceMostlyRead));
    m_items_local_id = &m_variable_items_local_id->_internalTrueData()->_internalDeprecatedValue();
    updateTimestamp();
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ItemGroupImplPrivate::
resetSubGroups()
{
  if (!m_is_all_items)
    ARCANE_FATAL("Call to _resetSubGroups() is only valid for group of AllItems");

  m_own_group = nullptr;
  m_ghost_group = nullptr;
  m_interface_group = nullptr;
  m_node_group = nullptr;
  m_edge_group = nullptr;
  m_face_group = nullptr;
  m_cell_group = nullptr;
  m_inner_face_group = nullptr;
  m_outer_face_group = nullptr;
  m_active_cell_group = nullptr;
  m_own_active_cell_group = nullptr;
  m_active_face_group = nullptr;
  m_own_active_face_group = nullptr;
  m_inner_active_face_group = nullptr;
  m_outer_active_face_group = nullptr;
  m_level_cell_group.clear();
  m_own_level_cell_group.clear();
  m_children_by_type.clear();
  m_sub_groups.clear();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ItemGroupImplPrivate::
notifyExtendObservers(const Int32ConstArrayView * info)
{
  ARCANE_ASSERT((!m_need_recompute || m_is_all_items),("Operation on invalid group"));
  for( const auto& i : m_observers ) {
    IItemGroupObserver * obs = i.second;
    obs->executeExtend(info);
  }
  if (m_group_index_table.isUsed())
    m_group_index_table->update();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ItemGroupImplPrivate::
notifyReduceObservers(const Int32ConstArrayView * info)
{
  ARCANE_ASSERT((!m_need_recompute || m_is_all_items),("Operation on invalid group"));
  for( const auto& i : m_observers ) {
    IItemGroupObserver * obs = i.second;
    obs->executeReduce(info);
  }
  if (m_group_index_table.isUsed())
    m_group_index_table->update();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ItemGroupImplPrivate::
notifyCompactObservers(const Int32ConstArrayView * info)
{
  ARCANE_ASSERT((!m_need_recompute || m_is_all_items),("Operation on invalid group"));
  for( const auto& i : m_observers ) {
    IItemGroupObserver * obs = i.second;
    obs->executeCompact(info);
  }
  if (m_group_index_table.isUsed())
    m_group_index_table->compact(info);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ItemGroupImplPrivate::
notifyInvalidateObservers()
{
#ifndef NO_USER_WARNING
#warning "(HP) Assertion need fix"
#endif /* NO_USER_WARNING */
  // Cela peut se produire en cas d'invalidation en cascade
  // ARCANE_ASSERT((!m_need_recompute),("Operation on invalid group"));
  for( const auto& i : m_observers ) {
    IItemGroupObserver * obs = i.second;
    obs->executeInvalidate();
  }
  if (m_group_index_table.isUsed())
    m_group_index_table->update();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Vérifie que les localIds() sont contigüs.
 */
void ItemGroupImplPrivate::
checkIsContigous()
{
  m_is_contigous = false;
  Int32ConstArrayView lids = itemsLocalId();
  if (lids.empty()){
    m_is_contigous = false;
    return;
  }
  Int32 first_lid = lids[0];

  bool is_bad = false;
  for( Integer i=0, n=lids.size(); i<n; ++i ){
    if (lids[i]!=(first_lid+i)){
      is_bad = true;
      break;
    }
  }
  if (!is_bad)
    m_is_contigous = true;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ItemGroupImpl* ItemGroupImpl::shared_null= 0;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class ItemGroupImplItemGroupComputeFunctor
: public ItemGroupComputeFunctor
{
 private:
  typedef void (ItemGroupImpl::*FuncPtr)();

 public:
  ItemGroupImplItemGroupComputeFunctor(ItemGroupImpl * parent, FuncPtr funcPtr)
    : ItemGroupComputeFunctor()
    , m_parent(parent)
    , m_function(funcPtr) { }

 public:
  virtual void executeFunctor()
  {
    (m_parent->*m_function)();
  }

 private:
  ItemGroupImpl * m_parent;
  FuncPtr m_function;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ItemGroupImpl* ItemGroupImpl::
checkSharedNull()
{
  // Normalement ce test n'est vrai que si on a une instance globale
  // de 'ItemGroup' ce qui est déconseillé. Sinon, _buildSharedNull() a été
  // automatiquement appelé lors de l'initialisation (dans arcaneInitialize()).
  if (!shared_null)
    _buildSharedNull();
  return shared_null;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ItemGroupImpl::
ItemGroupImpl(IItemFamily* family,const String& name)
: m_p (new ItemGroupImplPrivate(family,name))
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ItemGroupImpl::
ItemGroupImpl(IItemFamily* family,ItemGroupImpl* parent,const String& name)
: m_p(new ItemGroupImplPrivate(family,parent,name))
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ItemGroupImpl::
ItemGroupImpl()
: m_p (new ItemGroupImplPrivate())
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ItemGroupImpl::
~ItemGroupImpl()
{
  delete m_p;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

const String& ItemGroupImpl::
name() const
{
  return m_p->name();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

const String& ItemGroupImpl::
fullName() const
{
  return m_p->fullName();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Integer ItemGroupImpl::
size() const
{
  return m_p->itemsLocalId().size();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

bool ItemGroupImpl::
empty() const
{
  return m_p->itemsLocalId().empty();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Int32ConstArrayView ItemGroupImpl::
itemsLocalId() const
{
  return m_p->itemsLocalId();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ItemGroupImpl::
beginTransaction()
{
  if (m_p->m_transaction_mode)
    ARCANE_FATAL("Transaction mode already started");
  m_p->m_transaction_mode = true;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ItemGroupImpl::
endTransaction()
{
  if (!m_p->m_transaction_mode)
    ARCANE_FATAL("Transaction mode not started");
  m_p->m_transaction_mode = false;
  if (m_p->m_need_recompute) {
    m_p->m_need_recompute = false;
    m_p->m_need_invalidate_on_recompute = false;
    m_p->notifyInvalidateObservers();
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Int32Array& ItemGroupImpl::
unguardedItemsLocalId(const bool self_invalidate)
{
  ITraceMng* trace = m_p->m_mesh->traceMng();
  trace->debug(Trace::Medium) << "ItemGroupImpl::unguardedItemsLocalId on group " << name()
                              << " with self_invalidate=" << self_invalidate;

  if (m_p->m_compute_functor && !m_p->m_transaction_mode)
    ARCANE_FATAL("Direct access for computed group in only available during a transaction");

  _forceInvalidate(self_invalidate);
  return m_p->mutableItemsLocalId();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ItemGroupImpl* ItemGroupImpl::
parent() const
{
  return m_p->m_parent;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

IMesh* ItemGroupImpl::
mesh() const
{
  return m_p->mesh();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

IItemFamily* ItemGroupImpl::
itemFamily() const
{
  return m_p->m_item_family;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

bool ItemGroupImpl::
null() const
{
  return m_p->null();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

bool ItemGroupImpl::
isOwn() const
{
  return m_p->m_is_own;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ItemGroupImpl::
setOwn(bool v)
{
  const bool is_own = m_p->m_is_own;
  if (is_own == v)
    return;
  if (!is_own) {
    if (m_p->m_own_group)
      ARCANE_THROW(NotSupportedException,"Setting Own with 'Own' sub-group already defined");
  }
  else {
    // On a le droit de remettre setOwn() à 'false' pour le groupe de toutes
    // les entités. Cela est nécessaire en reprise si le nombre de parties
    // du maillage est différent du IParallelMng associé à la famille
    if (!isAllItems())
      ARCANE_THROW(NotSupportedException,"Un-setting Own on a own group");
  }
  m_p->m_is_own = v;
  // (HP) TODO: Faut il notifier des observers ?
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

eItemKind ItemGroupImpl::
itemKind() const
{
  return m_p->kind();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ItemGroupImpl* ItemGroupImpl::
ownGroup()
{
	ItemGroupImpl* ii = m_p->m_own_group;
	// Le flag est déjà positionné dans le ItemGroupImplPrivate::_init ou ItemGroupImpl::setOwn
	if (!ii) {
		if (m_p->m_is_own){
			ii = this;
			m_p->m_own_group = ii;
		} else {
			ii = createSubGroup("Own",m_p->m_item_family,new OwnItemGroupComputeFunctor());
			m_p->m_own_group = ii;
			ii->setOwn(true);
		}
	}
	return ii;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ItemGroupImpl* ItemGroupImpl::
ghostGroup()
{
	ItemGroupImpl* ii = m_p->m_ghost_group;
	if (!ii) {
		ii = createSubGroup("Ghost",m_p->m_item_family,new GhostItemGroupComputeFunctor());
		m_p->m_ghost_group = ii;
	}
	return ii;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ItemGroupImpl* ItemGroupImpl::
interfaceGroup()
{
	if (itemKind()!=IK_Face)
		return checkSharedNull();
	ItemGroupImpl* ii = m_p->m_interface_group;
	if (!ii) {
		ii = createSubGroup("Interface",m_p->m_item_family,new InterfaceItemGroupComputeFunctor());
		m_p->m_interface_group = ii;
	}
	return ii;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ItemGroupImpl* ItemGroupImpl::
nodeGroup()
{
  if (itemKind()==IK_Node)
    return this;
  ItemGroupImpl* ii = m_p->m_node_group;
  if (!ii){
    ii = createSubGroup("Nodes",m_p->m_mesh->nodeFamily(),new ItemItemGroupComputeFunctor<Node>());
    m_p->m_node_group = ii;
  }
  return ii;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ItemGroupImpl* ItemGroupImpl::
edgeGroup()
{
  if (itemKind()==IK_Edge)
    return this;
  ItemGroupImpl* ii = m_p->m_edge_group;
  if (!ii){
    ii = createSubGroup("Edges",m_p->m_mesh->edgeFamily(),new ItemItemGroupComputeFunctor<Edge>());
    m_p->m_edge_group = ii;
  }
  return ii;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ItemGroupImpl* ItemGroupImpl::
faceGroup()
{
  if (itemKind()==IK_Face)
    return this;
  ItemGroupImpl* ii = m_p->m_face_group;
  if (!ii){
    ii = createSubGroup("Faces",m_p->m_mesh->faceFamily(),new ItemItemGroupComputeFunctor<Face>());
    m_p->m_face_group = ii;
  }
  return ii;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ItemGroupImpl* ItemGroupImpl::
cellGroup()
{
  if (itemKind()==IK_Cell)
    return this;
  ItemGroupImpl* ii = m_p->m_cell_group;
  if (!ii){
    ii = createSubGroup("Cells",m_p->m_mesh->cellFamily(),new ItemItemGroupComputeFunctor<Cell>());
    m_p->m_cell_group = ii;
  }
  return ii;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ItemGroupImpl* ItemGroupImpl::
innerFaceGroup()
{
  if (itemKind()!=IK_Cell)
    return checkSharedNull();
  ItemGroupImpl* ii = m_p->m_inner_face_group;
  if (!ii){
    ii = createSubGroup("InnerFaces",m_p->m_mesh->faceFamily(),
                        new InnerFaceItemGroupComputeFunctor());
    m_p->m_inner_face_group = ii;
  }
  return ii;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ItemGroupImpl* ItemGroupImpl::
outerFaceGroup()
{
  if (itemKind()!=IK_Cell)
    return checkSharedNull();
  ItemGroupImpl* ii = m_p->m_outer_face_group;
  if (!ii){
    ii = createSubGroup("OuterFaces",m_p->m_mesh->faceFamily(),
                        new OuterFaceItemGroupComputeFunctor());
    m_p->m_outer_face_group = ii;
  }
  return ii;
}
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
//! AMR

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ItemGroupImpl* ItemGroupImpl::
activeCellGroup()
{
  if (itemKind()!=IK_Cell)
    return checkSharedNull();
  ItemGroupImpl* ii = m_p->m_active_cell_group;

  if (!ii){
    ii = createSubGroup("ActiveCells",m_p->m_mesh->cellFamily(),
                        new ActiveCellGroupComputeFunctor());
    m_p->m_active_cell_group = ii;
  }
  return ii;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ItemGroupImpl* ItemGroupImpl::
ownActiveCellGroup()
{
  if (itemKind()!=IK_Cell)
    return checkSharedNull();
  ItemGroupImpl* ii = m_p->m_own_active_cell_group;
  // Le flag est déjà positionné dans le ItemGroupImplPrivate::_init ou ItemGroupImpl::setOwn
  if (!ii) {
    ii = createSubGroup("OwnActiveCells",m_p->m_mesh->cellFamily(),
                        new OwnActiveCellGroupComputeFunctor());
    m_p->m_own_active_cell_group = ii;
    ii->setOwn(true);
  }
  return ii;
}
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ItemGroupImpl* ItemGroupImpl::
levelCellGroup(const Integer& level)
{
  if (itemKind()!=IK_Cell)
    return checkSharedNull();
  ItemGroupImpl* ii = m_p->m_level_cell_group[level];
  if (!ii){
    ii = createSubGroup(String::format("LevelCells{0}",level),
                        m_p->m_mesh->cellFamily(),
                        new LevelCellGroupComputeFunctor(level));
    m_p->m_level_cell_group[level] = ii;
  }
  return ii;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ItemGroupImpl* ItemGroupImpl::
ownLevelCellGroup(const Integer& level)
{
  if (itemKind()!=IK_Cell)
    return checkSharedNull();
  ItemGroupImpl* ii = m_p->m_own_level_cell_group[level];
  // Le flag est déjà positionné dans le ItemGroupImplPrivate::_init ou ItemGroupImpl::setOwn
  if (!ii) {
    ii = createSubGroup(String::format("OwnLevelCells{0}",level),
                        m_p->m_mesh->cellFamily(),
                        new OwnLevelCellGroupComputeFunctor(level));
    m_p->m_own_level_cell_group[level] = ii;
    ii->setOwn(true);
  }
  return ii;
}
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ItemGroupImpl* ItemGroupImpl::
activeFaceGroup()
{
  if (itemKind()!=IK_Cell)
    return checkSharedNull();
  ItemGroupImpl* ii = m_p->m_active_face_group;
  if (!ii){
    ii = createSubGroup("ActiveFaces",m_p->m_mesh->faceFamily(),
                        new ActiveFaceItemGroupComputeFunctor());
    m_p->m_active_face_group = ii;
  }
  return ii;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ItemGroupImpl* ItemGroupImpl::
ownActiveFaceGroup()
{
  if (itemKind()!=IK_Cell)
    return checkSharedNull();
  ItemGroupImpl* ii = m_p->m_own_active_face_group;
  if (!ii){
    ii = createSubGroup("OwnActiveFaces",m_p->m_mesh->faceFamily(),
                        new OwnActiveFaceItemGroupComputeFunctor());
    m_p->m_own_active_face_group = ii;
  }
  return ii;
}
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ItemGroupImpl* ItemGroupImpl::
innerActiveFaceGroup()
{
  if (itemKind()!=IK_Cell)
    return checkSharedNull();
  ItemGroupImpl* ii = m_p->m_inner_active_face_group;
  if (!ii){
    ii = createSubGroup("InnerActiveFaces",m_p->m_mesh->faceFamily(),
                        new InnerActiveFaceItemGroupComputeFunctor());
    m_p->m_inner_active_face_group = ii;
  }
  return ii;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ItemGroupImpl* ItemGroupImpl::
outerActiveFaceGroup()
{
  if (itemKind()!=IK_Cell)
    return checkSharedNull();
  ItemGroupImpl* ii = m_p->m_outer_active_face_group;
  if (!ii){
    ii = createSubGroup("OuterActiveFaces",m_p->m_mesh->faceFamily(),
                        new OuterActiveFaceItemGroupComputeFunctor());
    m_p->m_outer_active_face_group = ii;
  }
  return ii;
}
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ItemGroupImpl* ItemGroupImpl::
createSubGroup(const String& suffix, IItemFamily* family, ItemGroupComputeFunctor * functor)
{
  String sub_name = name() + "_" + suffix;
  auto finder = m_p->m_sub_groups.find(sub_name);
  if (finder != m_p->m_sub_groups.end()) {
    ARCANE_FATAL("Cannot create already existing sub-group ({0}) in group ({1})",
                 suffix, name());
  }
  ItemGroup ig = family->createGroup(sub_name,ItemGroup(this));
#ifndef NO_USER_WARNING
#warning "(HP) Assertion need fix"
#endif /* NO_USER_WARNING */
  // ARCANE_ASSERT((m_p->variableItemsLocalid() == NULL),("localIds variable for subGroup not allowed"));
  ItemGroupImpl* ii = ig.internal();
  ii->setComputeFunctor(functor);
  functor->setGroup(ii);
  // Observer par défaut : le sous groupe n'est pas intéressé par les infos détaillées de transition
  attachObserver(ii,newItemGroupObserverT(ii,
                                          &ItemGroupImpl::_executeInvalidate));
  m_p->m_sub_groups[sub_name] = ii;
  ii->invalidate(false);
  return ii;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ItemGroupImpl* ItemGroupImpl::
findSubGroup(const String& suffix)
{
  String sub_name = name() + "_" + suffix;
  auto finder = m_p->m_sub_groups.find(sub_name);
  if (finder != m_p->m_sub_groups.end()) {
    return finder->second.get();
  }
  else {
    // ou bien erreur ?
    return checkSharedNull();
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ItemGroupImpl::
changeIds(Int32ConstArrayView old_to_new_ids)
{
  ITraceMng* trace = m_p->m_mesh->traceMng();
  if (m_p->m_compute_functor) {
    trace->debug(Trace::High) << "ItemGroupImpl::changeIds on "
                              << name() << " : skip computed group";
    return;
  }

  // ItemGroupImpl ne fait d'habitude pas le checkNeedUpdate lui meme, plutot le ItemGroup
  checkNeedUpdate();
  if (isAllItems()) {
    // Int32ArrayView items_lid = m_p->itemsLocalId();
    // for( Integer i=0, is=items_lid.size(); i<is; ++i ) {
    //   Integer lid = items_lid[i];
    //   ARCANE_ASSERT((lid = old_to_new_ids[lid]),("Non uniform order on allItems"));;
    // }
    // m_p->notifyCompactObservers(&old_to_new_ids);
    m_p->notifyInvalidateObservers();
    return;
  }

  Int32ArrayView items_lid = m_p->itemsLocalId();
  for( Integer i=0, is=items_lid.size(); i<is; ++i ){
    Integer old_id = items_lid[i];
    items_lid[i] = old_to_new_ids[old_id];
  }

  m_p->updateTimestamp();
  // Pour l'instant, il ne faut pas trier les entités des variables
  // partielles car les valeurs correspondantes des variables ne sont
  // pas mises à jour (il faut implémenter changeGroupIds pour cela)
  // NOTE: est-ce utile de le faire ?
  // NOTE: faut-il le faire pour les autre aussi ? A priori, cela n'est
  // utile que pour garantir le meme resultat parallele/sequentiel
  if (m_p->m_observer_need_info) {
    m_p->notifyCompactObservers(&old_to_new_ids);
  } else {
    // Pas besoin d'infos, on peut changer arbitrairement leur ordre
#ifndef NO_USER_WARNING
#warning "(HP) Connexion de ce triage d'item avec la famille ?"
#endif /* NO_USER_WARNING */
    std::sort(std::begin(items_lid),std::end(items_lid));
    m_p->notifyCompactObservers(nullptr);
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ItemGroupImpl::
invalidate(bool force_recompute)
{
#ifdef ARCANE_DEBUG
  //   if (force_recompute){
  ITraceMng* msg = m_p->m_mesh->traceMng();
  msg->debug(Trace::High) << "ItemGroupImpl::invalidate(force=" << force_recompute << ")"
                          << " name=" << name();
//   }
#endif
#ifndef NO_USER_WARNING
#warning "(GG) Vérifier si test suivant OK"
#endif /* NO_USER_WARNING */
#if 0
  if (force_recompute)
    throw NotImplementedException(A_FUNCINFO,"remove this protection if force_recomputed is used");
#endif

  m_p->updateTimestamp();
  m_p->setNeedRecompute();
  if (force_recompute)
    checkNeedUpdate();
  m_p->notifyInvalidateObservers();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ItemInternalList ItemGroupImpl::
itemsInternal() const
{
  return m_p->items();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ItemInfoListView ItemGroupImpl::
itemInfoListView() const
{
  return m_p->itemInfoListView();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

bool ItemGroupImpl::
isLocalToSubDomain() const
{
  return m_p->m_is_local_to_sub_domain;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ItemGroupImpl::
setLocalToSubDomain(bool v)
{
  m_p->m_is_local_to_sub_domain = v;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ItemGroupImpl::
addItems(Int32ConstArrayView items_local_id,bool check_if_present)
{
  ARCANE_ASSERT(( (!m_p->m_need_recompute && !isAllItems()) || (m_p->m_transaction_mode && isAllItems()) ),
		("Operation on invalid group"));
  if (m_p->m_compute_functor && !m_p->m_transaction_mode)
    ARCANE_FATAL("Cannot add items on computed group ({0})", name());
  IMesh* amesh = mesh();
  if (!amesh)
    throw ArgumentException(A_FUNCINFO,"null group");
  ITraceMng* trace = amesh->traceMng();

  Integer nb_item_to_add = items_local_id.size();
  if (nb_item_to_add==0)
    return;

  Int32Array& items_lid = m_p->mutableItemsLocalId();
  const Integer current_size = items_lid.size();
  Integer nb_added = 0;

  if(isAllItems()) {
    // Ajoute les nouveaux items à la fin
    Integer nb_items_id = current_size;
    m_p->m_items_index_in_all_group.resize(m_p->maxLocalId());
    for( Integer i=0, is=nb_item_to_add; i<is; ++i ){
      Int32 local_id = items_local_id[i];
      items_lid.add(local_id);
      m_p->m_items_index_in_all_group[local_id] = nb_items_id+i;
    } 
  }
  else if (check_if_present) {
    // Vérifie que les entités à ajouter ne sont pas déjà présentes
    UniqueArray<bool> presence_checks(m_p->maxLocalId());
    presence_checks.fill(false);
    for( Integer i=0, is=items_lid.size(); i<is; ++i ){
      presence_checks[items_lid[i]] = true;
    }

    for( Integer i=0; i<nb_item_to_add; ++i ) {
      const Integer lid = items_local_id[i];
      if (!presence_checks[lid]){
        items_lid.add(lid);
        // Met à \a true comme cela si l'entité est présente plusieurs fois
        // dans \a items_local_id cela fonctionne quand même.
        presence_checks[lid] = true;
        ++nb_added;
      }
    }
  }
  else {
    nb_added = nb_item_to_add;
    items_lid.resize(current_size+nb_added);
    Int32ArrayView ptr = items_lid;
    memcpy(&ptr[current_size],items_local_id.data(),sizeof(Int32)*nb_added);
  }

  if (arcaneIsCheck()){
    trace->debug(Trace::High) << "ItemGroupImpl::addItems() group <" << name() << "> "
                              << " checkpresent=" << check_if_present
                              << " nb_current=" << current_size
                              << " want_to_add=" << nb_item_to_add
                              << " effective_added=" << nb_added;
    checkValid();
  }

  if (nb_added!=0) {
    m_p->updateTimestamp();
    Int32ConstArrayView observation_info(nb_added, items_lid.unguardedBasePointer()+current_size);
    m_p->notifyExtendObservers(&observation_info);
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ItemGroupImpl::
removeItems(Int32ConstArrayView items_local_id,bool check_if_present)
{
  // TODO: check_if_present n'est pas pris en compte
  ARCANE_UNUSED(check_if_present);

  ARCANE_ASSERT(( (!m_p->m_need_recompute && !isAllItems()) || (m_p->m_transaction_mode && isAllItems()) ),
		("Operation on invalid group"));
  if (m_p->m_compute_functor && !m_p->m_transaction_mode)
    ARCANE_FATAL("Cannot remove items on computed group ({0})", name());
  IMesh* amesh = mesh();
  if (!amesh)
    throw ArgumentException(A_FUNCINFO,"null group");
  ITraceMng* trace = amesh->traceMng();
  if (isOwn() && amesh->meshPartInfo().nbPart()!=1){
    ARCANE_THROW(NotSupportedException,"Cannot remove items if isOwn() is true");
  }

  Integer nb_item_to_remove = items_local_id.size();
  Int32UniqueArray removed_ids, removed_lids;

  if (nb_item_to_remove!=0) { // on ne peut tout de même pas faire de retour anticipé à cause des observers

    Int32Array& items_lid = m_p->mutableItemsLocalId();
    const Integer old_size = items_lid.size();
    bool has_removed = false;
   
    if(isAllItems()) {
      // Algorithme anciennement dans DynamicMeshKindInfo
      // Supprime des items du groupe all_items par commutation avec les 
      // éléments de fin de ce groupe
      // ie memoire persistante O(size groupe), algo O(remove items)
      has_removed = true;
      Integer nb_item = old_size;
      for( Integer i=0, is=nb_item_to_remove; i<is; ++i ){
        Int32 removed_local_id = items_local_id[i];
        Int32 index = m_p->m_items_index_in_all_group[removed_local_id];
        --nb_item;
        Int32 moved_local_id = items_lid[nb_item];
        items_lid[index] = moved_local_id;
        m_p->m_items_index_in_all_group[moved_local_id] = index;
      }
      items_lid.resize(nb_item);
    }
    else {
      // Algorithme pour les autres groupes
      // Décalage de tableau
      // ie mémoire locale O(size groupe), algo O(size groupe)
      // Marque un tableau indiquant les entités à supprimer
      BoolUniqueArray remove_flags(m_p->maxLocalId(),false);
      for( Integer i=0; i<nb_item_to_remove; ++i )
        remove_flags[items_local_id[i]] = true;
    
      {
        Integer next_index = 0;
        for( Integer i=0; i<old_size; ++i ){
          Int32 lid = items_lid[i];
          if (remove_flags[lid]) {
            removed_ids.add(i);
            removed_lids.add(lid);
            continue;
          }
          items_lid[next_index] = lid;
          ++next_index;
        }
        if (next_index!=old_size) {
          has_removed = true;
          items_lid.resize(next_index);
        }
      }
    }
  
    m_p->updateTimestamp();
    if(arcaneIsCheck()){
      trace->debug(Trace::High) << "ItemGroupImpl::removeItems() group <" << name() << "> "
                                << " old_size=" << old_size
                                << " new_size=" << size()
                                << " removed?=" << has_removed;
      checkValid();
    }
  }

  Int32ConstArrayView observation_info(removed_lids.size(), removed_lids.unguardedBasePointer());
  m_p->notifyReduceObservers(&observation_info);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ItemGroupImpl::
removeAddItems(Int32ConstArrayView removed_items_lids,
               Int32ConstArrayView added_items_lids,
               bool check_if_present)
{
  ARCANE_ASSERT(( (!m_p->m_need_recompute && !isAllItems()) || (m_p->m_transaction_mode && isAllItems()) ),
                ("Operation on invalid group"));
  if (m_p->m_compute_functor && !m_p->m_transaction_mode)
    ARCANE_FATAL("Cannot remove items on computed group ({0})", name());
  IMesh* amesh = mesh();
  if (!amesh)
    throw ArgumentException(A_FUNCINFO,"null group");
  ITraceMng* trace = amesh->traceMng();
  if (isOwn() && amesh->meshPartInfo().nbPart()!=1){
    ARCANE_THROW(NotSupportedException,"Cannot remove items if isOwn() is true");
  }
 
  Int32Array & items_lid = m_p->mutableItemsLocalId();
  
  if(isAllItems()) {
    ItemInternalList internals = itemsInternal();
    const Integer internal_size = internals.size();
    const Integer new_size = m_p->m_item_family->nbItem();
    items_lid.resize(new_size);
    m_p->m_items_index_in_all_group.resize(m_p->maxLocalId());
    if (new_size==internal_size){
      // Il n'y a pas de trous dans la numérotation
      for( Integer i=0; i<internal_size; ++i ){
        Int32 local_id = internals[i]->localId();
        items_lid[i] = local_id;
        m_p->m_items_index_in_all_group[local_id] = i;
      }
    }
    else{
      Integer index = 0;
      for( Integer i=0; i<internal_size; ++i ){
        ItemInternal* item = internals[i];
        if (!item->isSuppressed()){
          Int32 local_id = item->localId();
          items_lid[index] = local_id;
          m_p->m_items_index_in_all_group[local_id] = index;
          ++index;
        }
      }
      if (index!=new_size){
        trace->fatal() << "Inconsistent number of elements in the generation "
                       << "of the group " << name()
                       << " (expected: " << new_size
                       << " present: " << index
                       << ")";
      }
    }
  }
  else {
    removeItems(removed_items_lids,check_if_present);
    addItems(added_items_lids,check_if_present);
  }
  
  if(arcaneIsCheck()){
    trace->debug(Trace::High) << "ItemGroupImpl::removeAddItems() group <" << name() << "> "
                              << " old_size=" << m_p->m_item_family->nbItem()
                              << " new_size=" << size()
                              << " nb_removed=" << removed_items_lids.size()
                              << " nb_added=" << added_items_lids.size();
    checkValid();
  }
}


/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ItemGroupImpl::
setItems(Int32ConstArrayView items_local_id)
{
  if (m_p->m_compute_functor && !m_p->m_transaction_mode)
    ARCANE_FATAL("Cannot set items on computed group ({0})", name());
  Int32Array& buf = m_p->mutableItemsLocalId();
  buf.resize(items_local_id.size());
  Int32ArrayView buf_view(buf);
  buf_view.copy(items_local_id);
  m_p->updateTimestamp();
  m_p->m_need_recompute = false;
  if (arcaneIsCheck()){
    ITraceMng* trace = m_p->mesh()->traceMng();
    trace->debug(Trace::High) << "ItemGroupImpl::setItems() group <" << name() << "> "
                              << " size=" << size();
    checkValid();
  }

  // On tolére encore le setItems initial en le convertissant en un addItems
  if (size() != 0)
    m_p->notifyInvalidateObservers();
  else
    m_p->notifyExtendObservers(&items_local_id);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class ItemGroupImpl::ItemSorter
{
 public:
  ItemSorter(ItemInfoListView items) : m_items(items){}
 public:
  void sort(Int32ArrayView local_ids) const
  {
    std::sort(std::begin(local_ids),std::end(local_ids),*this);
  }
  bool operator()(Int32 lid1,Int32 lid2) const
  {
    return m_items[lid1].uniqueId() < m_items[lid2].uniqueId();
  }
 private:
  ItemInfoListView m_items;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ItemGroupImpl::
setItems(Int32ConstArrayView items_local_id,bool do_sort)
{
  if (!do_sort){
    setItems(items_local_id);
    return;
  }
  UniqueArray<Int32> sorted_lid(items_local_id);
  ItemSorter sorter(itemFamily()->itemInfoListView());
  sorter.sort(sorted_lid);
  setItems(sorted_lid);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ItemGroupImpl::
setIsAllItems()
{
  m_p->m_is_all_items = true;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

bool ItemGroupImpl::
isAllItems() const
{
  return m_p->m_is_all_items;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class ItemCheckSuppressedFunctor
{
 public:
  ItemCheckSuppressedFunctor(ItemInternalList items)
  : m_items(items)
    {
    }
 public:
  bool operator()(Integer item_lid)
    {
      return m_items[item_lid]->isSuppressed();
    }
 private:
  ItemInternalList m_items;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#ifndef NO_USER_WARNING
#warning "(HP) Check function call in the code"
#endif /* NO_USER_WARNING */
void ItemGroupImpl::
removeSuppressedItems()
{
#ifndef NO_USER_WARNING
#warning "(HP) Assertion need fix"
#endif /* NO_USER_WARNING */
  // ARCANE_ASSERT((!m_p->m_need_recompute),("Operation on invalid group"));
  ITraceMng* trace = m_p->mesh()->traceMng();
  if (m_p->m_compute_functor) {
    trace->debug(Trace::High) << "ItemGroupImpl::removeSuppressedItems on " << name() << " : skip computed group";
    return;
  }

  ItemInternalList items = m_p->items();
  Integer nb_item = items.size();
  Int32Array& items_lid = m_p->mutableItemsLocalId();
  Integer current_size = items_lid.size();
  if (arcaneIsCheck()){
    for( Integer i=0; i<current_size; ++i ){
      if (items_lid[i]>=nb_item){
        trace->fatal() << "ItemGroupImpl::removeSuppressedItems(): bad range "
                       << " name=" << name()
                       << " i=" << i << " lid=" << items_lid[i]
                       << " max=" << nb_item;
      }
    }
  }

  Int32UniqueArray removed_ids, removed_lids;
  Int32ConstArrayView * observation_info  = NULL;
  Int32ConstArrayView * observation_info2 = NULL;
  Integer new_size;
  // Si le groupe posséde des observers ayant besoin d'info, il faut les calculer
  if (m_p->m_observer_need_info){
    removed_ids.reserve(current_size); // préparation à taille max
    Integer index = 0;
    for( Integer i=0; i<current_size; ++i ){
      if (!items[items_lid[i]]->isSuppressed()){
        items_lid[index] = items_lid[i];
        ++index;
      } else {
        // trace->debug(Trace::Highest) << "Remove from group " << name() << " item " << i << " " << items_lid[i] << " " << ItemPrinter(items[items_lid[i]]);
        removed_lids.add(items_lid[i]);
      }
    }
    new_size = index;
    if (new_size!=current_size){
      items_lid.resize(new_size);
      observation_info = new Int32ConstArrayView(removed_lids.size(), removed_lids.unguardedBasePointer());
    }
  }
  else{
    ItemCheckSuppressedFunctor f(m_p->items());
    auto ibegin = std::begin(items_lid);
    auto new_end = std::remove_if(ibegin,std::end(items_lid),f);
    new_size = (Integer)(new_end-ibegin);
    items_lid.resize(new_size);
  }

  if (arcaneIsCheck()) {
    trace->debug(Trace::High) << "ItemGroupImpl::removeSupressedItems() group <" << name()
                              << "> NEW SIZE=" << new_size << " OLD=" << current_size;
    checkValid();
  }

  // Ne met à jour que si le groupe a été modifié
  if (current_size != new_size) {
    m_p->updateTimestamp();
    m_p->notifyReduceObservers(observation_info);
    delete observation_info;
    delete observation_info2;
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ItemGroupImpl::
checkValid()
{
#if 0
  if (!arcaneIsCheck())
    return;
#endif
  ITraceMng* msg = m_p->mesh()->traceMng();
  if (m_p->m_need_recompute && m_p->m_compute_functor) {
    msg->debug(Trace::High) << "ItemGroupImpl::checkValid on " << name() << " : skip group to recompute";
    return;
  }

  // Les points suivants sont vérifiés:
  // - une entité n'est présente qu'une fois dans le groupe
  // - les entités du groupe ne sont pas détruites
  UniqueArray<bool> presence_checks(m_p->maxLocalId());
  presence_checks.fill(0);
  Integer nb_error = 0;

  ItemInternalList items(m_p->items());
  Integer items_size = items.size();
  Int32ConstArrayView items_lid(m_p->itemsLocalId());

  for( Integer i=0, is=items_lid.size(); i<is; ++i ){
    Integer lid = items_lid[i];
    if (lid>=items_size){
      if (nb_error<10){
        msg->error() << "Wrong local index lid=" << lid << " max=" << items_size
                     << " var_max_size=" << m_p->maxLocalId();
      }
      ++nb_error;
      continue;
    }
    ItemInternal* item = items[lid];
    if (item->isSuppressed()){
      if (nb_error<10){
        msg->error() << "Item " << ItemPrinter(item) << " in group "
                     << name() << " does not exist anymore";
      }
      ++nb_error;
    }
    if (presence_checks[lid]){
      if (nb_error<10){
        msg->error() << "Item " << ItemPrinter(item) << " in group "
                     << name() << " was found twice or more";
      }
      ++nb_error;
    }
    presence_checks[lid] = true;
  }
  if (isAllItems()) {
    for( Integer i=0, n=items_lid.size(); i<n; ++i ){
      Int32 local_id = items_lid[i];
      Int32 index_in_all_group = m_p->m_items_index_in_all_group[local_id];
      if (index_in_all_group!=i){
        if (nb_error<10){
          msg->error() << A_FUNCINFO
                       << ": " << itemKindName(m_p->m_kind)
                       << ": incoherence between 'local_id' and index in the group 'All' "
                       << " i=" << i
                       << " local_id=" << local_id
                       << " index=" << index_in_all_group;
        }
        ++nb_error;
      }
    }
  }
  if (nb_error!=0) {
    String parent_name = "none";
    if (parent())
      parent_name = parent()->name();
    ARCANE_FATAL("Error in group name='{0}' parent='{1}' nb_error={2}",
                 name(),parent_name,nb_error);
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

bool ItemGroupImpl::
checkNeedUpdate()
{
  // En cas de problème sur des recalculs très imbriqués, une proposition est
  // de désactiver les lignes #A pour activer les lignes #B
  bool has_recompute = false;
  if (m_p->m_need_recompute) {
    m_p->m_need_recompute = false;
    // bool need_invalidate_on_recompute = m_p->m_need_invalidate_on_recompute; // #B
    // m_p->m_need_invalidate_on_recompute = false;                             // #B
    if (m_p->m_compute_functor) {
      m_p->m_compute_functor->executeFunctor();
    }
    //     if (need_invalidate_on_recompute) { // #B
    //       m_p->notifyInvalidateObservers(); // #B
    //     }                                   // #B

    if (m_p->m_need_invalidate_on_recompute) {     // #A
      m_p->m_need_invalidate_on_recompute = false; // #A
      m_p->notifyInvalidateObservers();            // #A
    }                                              // #A
    has_recompute = true;
  }
  _checkUpdateSimdPadding();
  return has_recompute;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Remplit les derniers éléments du groupe pour avoir un vecteur
 * SIMD complet.
 *
 * Pour que la vectorisation fonctionne il faut que le nombre d'éléments
 * du groupe soit un multiple de la taille d'un vecteur SIMD. Si ce n'est
 * pas le cas, on remplit les dernières valeurs du tableau des localId()
 * avec le dernier élément.
 *
 * Par exemple, on supporse une taille d'un vecteur SIMD de 8 (ce qui est le maximum
 * actuellement avec l'AVX512) et un groupe \a grp de 13 éléments. Il faut donc
 * remplit le groupe comme suit:
 \code
 Int32 last_local_id = grp[12];
 grp[13] = grp[14] = grp[15] = last_local_id.
 \endcode
 *
 * A noter que la taille du groupe reste effectivement de 13 éléments. Le
 * padding supplémentaire n'est que pour les itérations via ENUMERATE_SIMD.
 * Comme le tableau des localId() est alloué avec l'allocateur d'alignement
 * il est garanti que la mémoire allouée est suffisante pour faire le padding.
 *
 * \todo Ne pas faire cela dans tous les checkNeedUpdate() mais mettre
 * en place une méthode qui retourne un énumérateur spécifique pour
 * la vectorisation.
 */
void ItemGroupImpl::
_checkUpdateSimdPadding()
{
  if (m_p->m_simd_timestamp >= m_p->timestamp())
    return;
  // Fait un padding des derniers éléments du tableau en recopiant la
  // dernière valeurs.
  m_p->m_simd_timestamp = m_p->timestamp();
  applySimdPadding(m_p->mutableItemsLocalId());
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ItemGroupImpl::
clear()
{
  Int32Array& items_lid = m_p->mutableItemsLocalId();
  if (!items_lid.empty())
    // Incrémente uniquement si le groupe n'est pas déjà vide
    m_p->updateTimestamp();
  items_lid.clear();
  m_p->m_need_recompute = false;
  for( const auto& i : m_p->m_sub_groups )
    i.second->clear();
  m_p->notifyInvalidateObservers();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ItemGroup ItemGroupImpl::
parentGroup()
{
  if (m_p->m_parent)
    return ItemGroup(m_p->m_parent);
  return ItemGroup();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ItemGroupImpl::
applyOperation(IItemOperationByBasicType* operation)
{
  ARCANE_ASSERT((!m_p->m_need_recompute),("Operation on invalid group"));
  if (m_p->m_children_by_type.empty()){
    _initChildrenByType();
  }

#define APPLY_OPERATION_ON_TYPE(ITEM_TYPE)                    \
  {                                                           \
    ItemGroup group(m_p->m_children_by_type[IT_##ITEM_TYPE]); \
    if (!group.empty())                                       \
      operation->apply##ITEM_TYPE(group.view());              \
  }

  APPLY_OPERATION_ON_TYPE(Vertex);
  APPLY_OPERATION_ON_TYPE(Line2);
  APPLY_OPERATION_ON_TYPE(Triangle3);
  APPLY_OPERATION_ON_TYPE(Quad4);
  APPLY_OPERATION_ON_TYPE(Pentagon5);
  APPLY_OPERATION_ON_TYPE(Hexagon6);
  APPLY_OPERATION_ON_TYPE(Tetraedron4);
  APPLY_OPERATION_ON_TYPE(Pyramid5);
  APPLY_OPERATION_ON_TYPE(Pentaedron6);
  APPLY_OPERATION_ON_TYPE(Hexaedron8);
  APPLY_OPERATION_ON_TYPE(Heptaedron10);
  APPLY_OPERATION_ON_TYPE(Octaedron12);
  APPLY_OPERATION_ON_TYPE(HemiHexa7);
  APPLY_OPERATION_ON_TYPE(HemiHexa6);
  APPLY_OPERATION_ON_TYPE(HemiHexa5);
  APPLY_OPERATION_ON_TYPE(HemiHexa7);
  APPLY_OPERATION_ON_TYPE(AntiWedgeLeft6);
  APPLY_OPERATION_ON_TYPE(AntiWedgeRight6);
  APPLY_OPERATION_ON_TYPE(DiTetra5);
  APPLY_OPERATION_ON_TYPE(DualNode);
  APPLY_OPERATION_ON_TYPE(DualEdge);
  APPLY_OPERATION_ON_TYPE(DualFace);
  APPLY_OPERATION_ON_TYPE(DualCell);
  APPLY_OPERATION_ON_TYPE(Link);
  
#undef APPLY_OPERATION_ON_TYPE
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

bool ItemGroupImpl::
needSynchronization() const
{
  return !(m_p->m_compute_functor != 0 ||
           m_p->m_is_local_to_sub_domain ||
           m_p->m_is_own);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Int64 ItemGroupImpl::
timestamp() const
{
  return m_p->timestamp();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ItemGroupImpl::
attachObserver(const void * ref, IItemGroupObserver * obs)
{
  auto finder = m_p->m_observers.find(ref);
  auto end = m_p->m_observers.end();
  if (finder != end) {
    delete finder->second;
    finder->second = obs;
    return;
  }
  else {
    m_p->m_observers[ref] = obs;
  }

  // Mise à jour du flag de demande d'info
  _updateNeedInfoFlag(m_p->m_observer_need_info | obs->needInfo());
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ItemGroupImpl::
detachObserver(const void * ref)
{
  auto finder = m_p->m_observers.find(ref);
  auto end = m_p->m_observers.end();

  if (finder == end)
    return;

  IItemGroupObserver * obs = finder->second;
  delete obs;
  m_p->m_observers.erase(finder);
  // Mise à jour du flag de demande d'info
  bool new_observer_need_info = false;
  auto i = m_p->m_observers.begin();
  for( ; i != end ; ++i ) {
    IItemGroupObserver * obs = i->second;
    new_observer_need_info |= obs->needInfo();
  }
  _updateNeedInfoFlag(new_observer_need_info);

  // On invalide la table de hachage éventuelle
  // des variables partielles si il n'y a plus
  // de référence. 
  if(m_p->m_group_index_table.isUsed() && m_p->m_group_index_table.isUnique()) {
    m_p->m_group_index_table.reset();
    m_p->m_synchronizer.reset();
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

bool ItemGroupImpl::
hasInfoObserver() const
{
  return m_p->m_observer_need_info;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ItemGroupImpl::
setComputeFunctor(IFunctor* functor)
{
  delete m_p->m_compute_functor;
  m_p->m_compute_functor = functor;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

bool ItemGroupImpl::
hasComputeFunctor() const
{
  return (m_p->m_compute_functor);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ItemGroupImpl::
_initChildrenByType()
{
  IItemFamily* family = m_p->m_item_family;
  ItemTypeMng* type_mng = family->mesh()->itemTypeMng();
  Integer nb_basic_item_type= ItemTypeMng::nbBasicItemType(); //NB_BASIC_ITEM_TYPE
  m_p->m_children_by_type.resize(nb_basic_item_type);
  for( Integer i=0; i<nb_basic_item_type; ++i ){
    // String child_name(name()+i);
    String child_name(type_mng->typeName(i));
    ItemGroupImpl* igi = createSubGroup(child_name,family,
                                        new ItemGroupImplItemGroupComputeFunctor(this,&ItemGroupImpl::_computeChildrenByType));
    m_p->m_children_by_type[i] = igi;
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ItemGroupImpl::
_computeChildrenByType()
{
  ItemGroup that_group(this);
  ITraceMng * trace = that_group.mesh()->traceMng();
  trace->debug(Trace::High) << "ItemGroupImpl::_computeChildrenByType for " << name();

  Integer nb_basic_item_type = ItemTypeMng::nbBasicItemType();

  UniqueArray< SharedArray<Int32> > items_by_type(nb_basic_item_type);
  for( Integer i=0; i<nb_basic_item_type; ++i ){
    ItemGroupImpl * impl = m_p->m_children_by_type[i];
    impl->beginTransaction();
  }

  ENUMERATE_ITEM(iitem,that_group){
    const Item& item = *iitem;
    Integer item_type = item.type();
    if (item_type<nb_basic_item_type)
      items_by_type[item_type].add(iitem.itemLocalId());
  }

  for( Integer i=0; i<nb_basic_item_type; ++i ){
    ItemGroupImpl * impl = m_p->m_children_by_type[i];
    impl->setItems(items_by_type[i]);
    impl->endTransaction();
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ItemGroupImpl::
_executeExtend(const Int32ConstArrayView* info)
{
  ARCANE_UNUSED(info);
  // On ne sait encore calculer appliquer des transformations à des groupes calculés
  // On choisit l'invalidation systématique
  m_p->notifyInvalidateObservers();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ItemGroupImpl::
_executeReduce(const Int32ConstArrayView* info)
{
  ARCANE_UNUSED(info);
  // On ne sait encore calculer appliquer des transformations à des groupes calculés
  // On choisit l'invalidation systématique
  m_p->notifyInvalidateObservers();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ItemGroupImpl::
_executeCompact(const Int32ConstArrayView* info)
{
  ARCANE_UNUSED(info);
  // On ne sait encore calculer appliquer des transformations à des groupes calculés
  // On choisit l'invalidation systématique en différé (évaluée lors du prochain checkNeedUpdate)
  m_p->notifyInvalidateObservers();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ItemGroupImpl::
_executeInvalidate()
{
  m_p->setNeedRecompute();
  m_p->notifyInvalidateObservers();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ItemGroupImpl::
_updateNeedInfoFlag(const bool flag)
{
  if (m_p->m_observer_need_info == flag)
    return;
  m_p->m_observer_need_info = flag;
  // Si change, change aussi l'observer du parent pour qu'il adapte les besoins en info de transition
  ItemGroupImpl * parent = m_p->m_parent;
  if (parent) {
    parent->detachObserver(this);
    if (m_p->m_observer_need_info) {
      parent->attachObserver(this,newItemGroupObserverT(this,
                                                        &ItemGroupImpl::_executeExtend,
                                                        &ItemGroupImpl::_executeReduce,
                                                        &ItemGroupImpl::_executeCompact,
                                                        &ItemGroupImpl::_executeInvalidate));
    } else {
      parent->attachObserver(this,newItemGroupObserverT(this,
                                                        &ItemGroupImpl::_executeInvalidate));
    }
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ItemGroupImpl::
_forceInvalidate(const bool self_invalidate)
{
  // (HP) TODO: Mettre un observer forceInvalidate pour prévenir tout le monde ?
  // avec forceInvalidate on doit invalider mais ne rien calculer
  if (self_invalidate) {
    m_p->setNeedRecompute();
    m_p->m_need_invalidate_on_recompute = true;
  }

  for( const auto& i :  m_p->m_sub_groups )
    i.second->_forceInvalidate(true);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ItemGroupImpl::
destroy()
{
  // Détache les observateurs. Cela modifie \a m_observers donc il faut
  // en faire une copie
  {
    std::vector<const void*> ptrs;
    for( const auto& i : m_p->m_observers )
      ptrs.push_back(i.first);
    for( const void* i : ptrs )
      detachObserver(i);
  }

  // Le groupe de toutes les entités est spécial. Il ne faut jamais le détruire
  // vraiement.
  if (m_p->m_is_all_items)
    m_p->resetSubGroups();
  else{
    delete m_p;
    m_p = new ItemGroupImplPrivate();
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

SharedPtrT<GroupIndexTable> ItemGroupImpl::
localIdToIndex()
{ 
  if(!m_p->m_group_index_table.isUsed()) {
    m_p->m_group_index_table = SharedPtrT<GroupIndexTable>(new GroupIndexTable(this));
    ITraceMng* trace =  m_p->m_mesh->traceMng();
    trace->debug(Trace::High) << "** CREATION OF LOCAL ID TO INDEX TABLE OF GROUP : " << m_p->m_name;
    m_p->m_group_index_table->update();
  }
  return m_p->m_group_index_table;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

IVariableSynchronizer * ItemGroupImpl::
synchronizer()
{ 
  if(!m_p->m_synchronizer.get()) {
    IParallelMng* pm = m_p->m_mesh->parallelMng();
    ItemGroup this_group(this);
    m_p->m_synchronizer = ParallelMngUtils::createSynchronizerRef(pm,this_group);
    ITraceMng* trace =  m_p->m_mesh->traceMng();
    trace->debug(Trace::High) << "** CREATION OF SYNCHRONIZER OF GROUP : " << m_p->m_name;
    m_p->m_synchronizer->compute();
  }
  return m_p->m_synchronizer.get();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

bool ItemGroupImpl::
hasSynchronizer()
{
  return m_p->m_synchronizer.get();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

bool ItemGroupImpl::
checkIsSorted() const
{
  // TODO: stocker l'info dans un flag et ne refaire le test que si
  // la liste des entités a changer (utiliser timestamp()).
  ItemInternalList items(m_p->items());
  Int32ConstArrayView items_lid(m_p->itemsLocalId());
  Integer nb_item = items_lid.size();
  // On est toujours trié si une seule entité ou moins.
  if (nb_item<=1)
    return true;
  // Compare chaque uniqueId() avec le précédent et vérifie qu'il est
  // supérieur.
  ItemUniqueId last_uid = items[items_lid[0]]->uniqueId();
  for( Integer i=1; i<nb_item; ++i ){
    ItemUniqueId uid = items[items_lid[i]]->uniqueId();
    if (uid<last_uid)
      return false;
    last_uid = uid;
  }
  return true;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ItemGroupImpl::
deleteMe()
{
  // S'il s'agit de 'shared_null'. Il faut le positionner à nullptr pour qu'il
  // soit réalloué éventuellement par _buildSharedNull().
  if (this==shared_null){
    shared_null = nullptr;
  }
  delete this;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ItemGroupImpl::
_buildSharedNull()
{
  if (!shared_null){
    shared_null = new ItemGroupImplNull();
    shared_null->addRef();
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ItemGroupImpl::
_destroySharedNull()
{
  // Supprime une référence à 'shared_null'. Si le nombre de référence
  // devient égal à 0, alors l'instance 'shared_null' est détruite.
  if (shared_null)
    shared_null->removeRef();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

bool ItemGroupImpl::
isContigousLocalIds() const
{
  return m_p->isContigous();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ItemGroupImpl::
checkLocalIdsAreContigous() const
{
  m_p->checkIsContigous();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Int64 ItemGroupImpl::
capacity() const
{
  Int32Array& items_lid = m_p->mutableItemsLocalId();
  return items_lid.capacity();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ItemGroupImpl::
shrinkMemory()
{
  if (hasComputeFunctor()){
    // Groupe calculé. On l'invalide et on supprime ses éléments
    invalidate(false);
    m_p->mutableItemsLocalId().clear();
  }

  if (m_p->variableItemsLocalid())
    m_p->variableItemsLocalid()->variable()->shrinkMemory();
  else
    m_p->mutableItemsLocalId().shrink();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
