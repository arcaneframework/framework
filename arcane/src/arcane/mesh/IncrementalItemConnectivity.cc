// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2023 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* IncrementalItemConnectivity.cc                              (C) 2000-2023 */
/*                                                                           */
/* Connectivité incrémentale des entités.                                    */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/mesh/IncrementalItemConnectivity.h"

#include "arcane/utils/StringBuilder.h"
#include "arcane/utils/ArgumentException.h"

#include "arcane/IMesh.h"
#include "arcane/IItemFamily.h"
#include "arcane/ItemPrinter.h"
#include "arcane/ConnectivityItemVector.h"
#include "arcane/MeshUtils.h"
#include "arcane/ObserverPool.h"
#include "arcane/Properties.h"
#include "arcane/IndexedItemConnectivityView.h"
#include "arcane/mesh/IndexedItemConnectivityAccessor.h"

#include "arcane/core/internal/IDataInternal.h"
#include "arcane/core/internal/IItemFamilyInternal.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::mesh
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

IndexedItemConnectivityAccessor::
IndexedItemConnectivityAccessor(IndexedItemConnectivityViewBase view, IItemFamily* target_item_family)
: IndexedItemConnectivityViewBase(view)
, m_item_shared_info(target_item_family->_internalApi()->commonItemSharedInfo())
{}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

IndexedItemConnectivityAccessor::
IndexedItemConnectivityAccessor(IIncrementalItemConnectivity* connectivity)
: m_item_shared_info(connectivity->targetFamily()->_internalApi()->commonItemSharedInfo())
{
  auto* ptr = dynamic_cast<mesh::IncrementalItemConnectivityBase*>(connectivity);
  if (ptr)
    IndexedItemConnectivityViewBase::set(ptr->connectivityView()) ;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

AbstractIncrementalItemConnectivity::
AbstractIncrementalItemConnectivity(IItemFamily* source_family,
                                    IItemFamily* target_family,
                                    const String& connectivity_name)
: TraceAccessor(source_family->traceMng())
, m_source_family(source_family)
, m_target_family(target_family)
, m_name(connectivity_name)
{
  m_families.add(m_source_family);
  m_families.add(m_target_family);

  //TODO: il faudra supprimer ces références lors de la destruction.
  source_family->addSourceConnectivity(this);
  target_family->addTargetConnectivity(this);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Ref<IIncrementalItemConnectivity> AbstractIncrementalItemConnectivity::
toReference()
{
  return Arccore::makeRef<IIncrementalItemConnectivity>(this);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class IncrementalItemConnectivityContainer
{
 public:
  struct Idx
  {
    //! Nombre d'entités connecté
    Int32 nb;
    //! Indice de la première entité dans la liste des entités connectées
    Int32 index;

    Idx(Int32 n,Int32 i) : nb(n), index(i){}
  };
 public:
  IncrementalItemConnectivityContainer(IMesh* mesh,const String& var_name)
  : m_var_name(var_name),
    m_connectivity_nb_item_variable(VariableBuildInfo(mesh,var_name+"Nb",IVariable::PPrivate)),
    m_connectivity_index_variable(VariableBuildInfo(mesh,var_name+"Index",IVariable::PPrivate)),
    m_connectivity_list_variable(VariableBuildInfo(mesh,var_name+"List",IVariable::PPrivate)),
    m_connectivity_nb_item_array(m_connectivity_nb_item_variable._internalTrueData()->_internalDeprecatedValue()),
    m_connectivity_index_array(m_connectivity_index_variable._internalTrueData()->_internalDeprecatedValue()),
    m_connectivity_list_array(m_connectivity_list_variable._internalTrueData()->_internalDeprecatedValue())
  {
    // Ajoute un tag pour indiquer que ce sont des variables associées à la connectivité.
    // Pour l'instant cela n'est utilisé que pour les statistiques d'affichage.

    String tag_name = "ArcaneConnectivity";
    m_connectivity_nb_item_variable.addTag(tag_name,"1");
    m_connectivity_index_variable.addTag(tag_name,"1");
    m_connectivity_list_variable.addTag(tag_name,"1");
  }

  String m_var_name;

  VariableArrayInt32 m_connectivity_nb_item_variable;
  VariableArrayInt32 m_connectivity_index_variable;
  VariableArrayInt32 m_connectivity_list_variable;

  VariableArrayInt32::ContainerType& m_connectivity_nb_item_array;
  VariableArrayInt32::ContainerType& m_connectivity_index_array;
  VariableArrayInt32::ContainerType& m_connectivity_list_array;

  ObserverPool m_observers;

  /*!
   * \brief Nombre maximum d'entités connectées.
   *
   * Il s'agit d'un majorant du nombre maximum d'entité connectées.
   * Pour des raisons de performance, cette valeur n'est pas mise à jour
   * si des entités sont retirées.
   */
  Int32 m_max_nb_item = 0;

 public:

  Integer size() const { return m_connectivity_nb_item_array.size(); }

  bool isAllocated() const { return size()>0; }

  void _checkResize(Int32 lid)
  {
    //TODO: réutiliser le code de ItemFamily::_setUniqueId().
    Integer size = m_connectivity_nb_item_array.size();
    Integer wanted_size = lid + 1;
    if (wanted_size<size)
      return;
    Integer capacity = m_connectivity_nb_item_array.capacity();
    if (wanted_size<capacity){
      // Pas besoin d'augmenter la capacité.
    }
    else{
      Integer reserve_size = 1000;
      while (lid>reserve_size){
        reserve_size *= 2;
      }
      m_connectivity_nb_item_array.reserve(reserve_size);
      m_connectivity_index_array.reserve(reserve_size);
    }
    m_connectivity_nb_item_array.resize(wanted_size);
    m_connectivity_index_array.resize(wanted_size);
  }

  void reserveForItems(Int32 capacity)
  {
    m_connectivity_nb_item_array.reserve(capacity);
    m_connectivity_index_array.reserve(capacity);
  }

 public:

};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

IncrementalItemConnectivityBase::
IncrementalItemConnectivityBase(IItemFamily* source_family,IItemFamily* target_family,
                                const String& aname)
: AbstractIncrementalItemConnectivity(source_family,target_family,aname)
, m_item_connectivity_list(nullptr)
, m_item_connectivity_index(-1)
{
  StringBuilder var_name("Connectivity");
  var_name += aname;
  var_name += source_family->name();
  var_name += target_family->name();

  IMesh* mesh = source_family->mesh();
  m_p = new IncrementalItemConnectivityContainer(mesh,var_name);

  typedef IncrementalItemConnectivityBase ThatClass;
  // Récupère les évènements de lecture pour indiquer qu'il faut mettre
  // à jour les vues.
  m_p->m_observers.addObserver(this,&ThatClass::_notifyConnectivityNbItemChangedFromObservable,
                               m_p->m_connectivity_nb_item_variable.variable()->readObservable());

  m_p->m_observers.addObserver(this,&ThatClass::_notifyConnectivityIndexChanged,
                               m_p->m_connectivity_index_variable.variable()->readObservable());

  m_p->m_observers.addObserver(this,&ThatClass::_notifyConnectivityListChanged,
                               m_p->m_connectivity_list_variable.variable()->readObservable());

  // Met à jour les vues à partir des tableaux associées.
  // Il faut le faire dès que la taille d'un tableau change car alors
  // il peut être réalloué et donc la vue associée devenir invalide.
  _notifyConnectivityListChanged();
  _notifyConnectivityIndexChanged();
  _notifyConnectivityNbItemChangedFromObservable();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

IncrementalItemConnectivityBase::
~IncrementalItemConnectivityBase()
{
  delete m_p;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void IncrementalItemConnectivityBase::
reserveMemoryForNbSourceItems(Int32 n, bool pre_alloc_connectivity)
{
  if (n<=0)
    return;

  m_p->reserveForItems(n);
  _notifyConnectivityIndexChanged();
  _notifyConnectivityNbItemChanged();

  if (pre_alloc_connectivity){
    Int32 pre_alloc_size = preAllocatedSize();
    if (pre_alloc_size>0){
      m_p->m_connectivity_list_array.reserve(n * pre_alloc_size);
      _notifyConnectivityListChanged();
    }
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void IncrementalItemConnectivityBase::
_notifyConnectivityListChanged()
{
  m_connectivity_list = m_p->m_connectivity_list_array.view();
  if (m_item_connectivity_list)
    m_item_connectivity_list->setConnectivityList(m_item_connectivity_index,m_connectivity_list);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void IncrementalItemConnectivityBase::
_notifyConnectivityIndexChanged()
{
  m_connectivity_index = m_p->m_connectivity_index_array.view();
  if (m_item_connectivity_list)
    m_item_connectivity_list->setConnectivityIndex(m_item_connectivity_index,m_connectivity_index);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void IncrementalItemConnectivityBase::
_notifyConnectivityNbItemChanged()
{
  m_connectivity_nb_item = m_p->m_connectivity_nb_item_array.view();
  if (m_item_connectivity_list)
    m_item_connectivity_list->setConnectivityNbItem(m_item_connectivity_index,m_connectivity_nb_item);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void IncrementalItemConnectivityBase::
_setMaxNbConnectedItemsInConnectivityList()
{
  if (m_item_connectivity_list)
    m_item_connectivity_list->setMaxNbConnectedItem(m_item_connectivity_index,m_p->m_max_nb_item);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * Méthode appelée lorsque la le nombre d'entité est modifié de manière externe,
 * par exemple en reprise ou après un retour-arrière.
 */
void IncrementalItemConnectivityBase::
_notifyConnectivityNbItemChangedFromObservable()
{
  _notifyConnectivityNbItemChanged();
  _computeMaxNbConnectedItem();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void IncrementalItemConnectivityBase::
_setNewMaxNbConnectedItems(Int32 new_max)
{
  if (new_max > m_p->m_max_nb_item){
    m_p->m_max_nb_item = new_max;
    _setMaxNbConnectedItemsInConnectivityList();
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void IncrementalItemConnectivityBase::
_computeMaxNbConnectedItem()
{
  // Force la remise à zéro pour être sur qu'il sera mis à jour
  m_p->m_max_nb_item = -1;
  Int32 max_nb_item = 0;
  for( Int32 x : m_connectivity_nb_item )
    if (x>max_nb_item)
      max_nb_item = x;
  _setNewMaxNbConnectedItems(max_nb_item);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Int32 IncrementalItemConnectivityBase::
maxNbConnectedItem() const
{
  return m_p->m_max_nb_item;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Indique que cette connectivité est associée à une des connectivités
 * de ItemInternal.
 *
 * Cela permet de mettre à jour directement la structure \a ilist dès que
 * cette connectivité est modifiée.
 */
void IncrementalItemConnectivityBase::
setItemConnectivityList(ItemInternalConnectivityList* ilist,Int32 index)
{
  info(4) << "setItemConnectivityList name=" << name() << " ilist=" << ilist << " index=" << index;
  m_item_connectivity_list = ilist;
  m_item_connectivity_index = index;
  _notifyConnectivityListChanged();
  _notifyConnectivityIndexChanged();
  _notifyConnectivityNbItemChanged();
  _setMaxNbConnectedItemsInConnectivityList();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void IncrementalItemConnectivityBase::
notifySourceFamilyLocalIdChanged(Int32ConstArrayView new_to_old_ids)
{
  if(m_p->isAllocated()){
    m_p->m_connectivity_nb_item_variable.variable()->compact(new_to_old_ids);
    m_p->m_connectivity_index_variable.variable()->compact(new_to_old_ids);
    _notifyConnectivityNbItemChanged();
    _notifyConnectivityIndexChanged();
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void IncrementalItemConnectivityBase::
notifyTargetFamilyLocalIdChanged(Int32ConstArrayView old_to_new_ids)
{
  Int32ArrayView ids = m_connectivity_list;
  const Integer n = ids.size();
  for( Integer i=0; i<n; ++i )
    if (ids[i]!=NULL_ITEM_LOCAL_ID)
      ids[i] = old_to_new_ids[ ids[i] ];
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ItemVectorView IncrementalItemConnectivityBase::
_connectedItems(ItemLocalId item,ConnectivityItemVector& con_items) const
{
  return con_items.resizeAndCopy(_connectedItemsLocalId(item));
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ItemConnectivityContainerView IncrementalItemConnectivityBase::
connectivityContainerView() const
{
  return { ItemLocalId::fromSpanInt32(m_connectivity_list), m_connectivity_index, m_connectivity_nb_item };
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

IndexedItemConnectivityViewBase IncrementalItemConnectivityBase::
connectivityView() const
{
  return { connectivityContainerView(), _sourceFamily()->itemKind(), _targetFamily()->itemKind()};
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

IndexedItemConnectivityAccessor IncrementalItemConnectivityBase::
connectivityAccessor() const
{
  return IndexedItemConnectivityAccessor(connectivityView(),_targetFamily());
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void IncrementalItemConnectivityBase::
dumpInfos()
{
  info() << "Infos index=" << m_connectivity_index;
  info() << "Infos nb_item=" << m_connectivity_nb_item;
  info() << "Infos list=" << m_connectivity_list;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

IncrementalItemConnectivity::
IncrementalItemConnectivity(IItemFamily* source_family,IItemFamily* target_family,
                            const String& aname)
: IncrementalItemConnectivityBase(source_family,target_family,aname)
, m_nb_add(0)
, m_nb_remove(0)
, m_nb_memcopy(0)
, m_pre_allocated_size(0)
{
  m_pre_allocated_size = _sourceFamily()->properties()->getIntegerWithDefault(name()+"PreallocSize",0);
  info(4) << "PreallocSize1 var=" << m_p->m_var_name << " v=" << m_pre_allocated_size;

  // Vérifie s'il faut ajouter l'entité nulle en début de liste.
  _checkAddNullItem();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

IncrementalItemConnectivity::
~IncrementalItemConnectivity()
{
  info(4) << " connectivity name=" << name()
          << " prealloc_size=" << m_pre_allocated_size
          << " nb_add=" << m_nb_add
          << " nb_remove=" << m_nb_remove
          << " nb_memcopy=" << m_nb_memcopy;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

inline Integer IncrementalItemConnectivity::
_increaseConnectivityList(Int32 new_lid)
{
  Integer pos_in_list = m_connectivity_list.size();
  m_p->m_connectivity_list_array.add(new_lid);
  _notifyConnectivityListChanged();
  return pos_in_list;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

inline Integer IncrementalItemConnectivity::
_increaseConnectivityList(Int32 new_lid,Integer nb_value)
{
  Integer pos_in_list = m_connectivity_list.size();
  m_p->m_connectivity_list_array.addRange(new_lid,nb_value);
  _notifyConnectivityListChanged();
  return pos_in_list;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void IncrementalItemConnectivity::
_resetConnectivityList()
{
  m_p->m_connectivity_list_array.clear();
  _notifyConnectivityListChanged();
  _checkAddNullItem();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void IncrementalItemConnectivity::
_increaseIndexList(Int32 lid,Integer size,Int32 target_lid)
{
  Integer added_range = (m_pre_allocated_size>0) ? m_pre_allocated_size : 1;
  ++m_nb_memcopy;
  Integer pos_in_index = m_connectivity_index[lid];
  Integer new_pos_in_list = _increaseConnectivityList(NULL_ITEM_LOCAL_ID,size+added_range);
  ArrayView<Int32> current_list(size,&(m_connectivity_list[pos_in_index]));
  ArrayView<Int32> new_list(size+1,&(m_connectivity_list[new_pos_in_list]));
  new_list.copy(current_list);
  // Ajoute la nouvelle entité à la fin de la liste des connectivités
  // TODO: regarder pour le tri dans l'ordre des uid() croissant.
  new_list[size] = target_lid;
  m_connectivity_index[lid] = new_pos_in_list;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void IncrementalItemConnectivity::
addConnectedItem(ItemLocalId source_item,ItemLocalId target_item)
{
  ++m_nb_add;
  const Int32 lid = source_item.localId();
  const Int32 target_lid = target_item.localId();
  Integer size = m_connectivity_nb_item[lid];
  // Ajoute une entité connectée.
  // Pour l'instant, le fonctionnement est basique.
  // On ajoute toujours les entités à la fin de m_p->m_connectivity_list.
  // S'il n'y en a pas, il suffit d'ajouter à la fin.
  // S'il y en a déjà, il faut allouer à la fin de
  // m_p->m_connectivity_list assez \a size+1 éléments et
  // on recopie la précédente connectivité dans le nouvel emplacement.
  // Forcément, avec le temps la liste va toujours grossir
  // car les trous ne sont pas réutilisés.
  if (m_pre_allocated_size!=0){
    // En cas de préallocation, on alloue par bloc de taille 'm_pre_allocated_size'.
    // Il faut donc réallouer si la taille est un multiple de m_pre_allocated_size
    if (size==0){
      Integer new_pos_in_list = _increaseConnectivityList(NULL_ITEM_LOCAL_ID,m_pre_allocated_size);
      m_connectivity_index[lid] = new_pos_in_list;
      m_connectivity_list[new_pos_in_list] = target_lid;
    }
    else{
      if (size<m_pre_allocated_size || (size%m_pre_allocated_size)!=0){
        Integer index = m_connectivity_index[lid];
        m_connectivity_list[index+size] = target_lid;
      }
      else{
        _increaseIndexList(lid,size,target_lid);
      }
    }
  }
  else{
    if (size==0){
      Integer new_pos_in_list = _increaseConnectivityList(target_lid);
      m_connectivity_index[lid] = new_pos_in_list;
    }
    else{
      _increaseIndexList(lid,size,target_lid);
    }
  }
  ++(m_connectivity_nb_item[lid]);
  _setNewMaxNbConnectedItems(m_connectivity_nb_item[lid]);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Integer IncrementalItemConnectivity::
_computeAllocSize(Integer nb_item)
{
  if (m_pre_allocated_size!=0){
    // Alloue un multiple de \a m_pre_allocated_size
    Integer alloc_size = nb_item / m_pre_allocated_size;
    if (alloc_size==0)
      return m_pre_allocated_size;
    if ((nb_item%m_pre_allocated_size)==0)
      return nb_item;
    return m_pre_allocated_size * (alloc_size + 1);
  }
  return nb_item;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void IncrementalItemConnectivity::
addConnectedItems(ItemLocalId source_item,Integer nb_item)
{
  const Int32 lid = source_item.localId();
  Integer size = m_connectivity_nb_item[lid];
  if (size!=0)
    ARCANE_FATAL("source_item already have connected items");
  Integer alloc_size = _computeAllocSize(nb_item);
  Integer new_pos_in_list = _increaseConnectivityList(NULL_ITEM_LOCAL_ID,alloc_size);
  m_connectivity_index[lid] = new_pos_in_list;
  m_connectivity_nb_item[lid] += nb_item;
  _setNewMaxNbConnectedItems(m_connectivity_nb_item[lid]);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void IncrementalItemConnectivity::
removeConnectedItems(ItemLocalId source_item)
{
  Int32 lid = source_item.localId();
  m_connectivity_nb_item[lid] = 0;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void IncrementalItemConnectivity::
removeConnectedItem(ItemLocalId source_item,ItemLocalId target_item)
{
  ++m_nb_remove;
  Int32 lid = source_item.localId();
  Int32 target_lid = target_item.localId();
  Integer size = m_connectivity_nb_item[lid];
  Int32* items = &(m_connectivity_list[ m_connectivity_index[lid] ]);
  mesh_utils::removeItemAndKeepOrder(Int32ArrayView(size,items),target_lid);
  --(m_connectivity_nb_item[lid]);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void IncrementalItemConnectivity::
replaceConnectedItem(ItemLocalId source_item,Integer index,ItemLocalId target_item)
{
  Int32 lid = source_item.localId();
  Int32 target_lid = target_item.localId();
  ARCANE_CHECK_AT(index,m_connectivity_nb_item[lid]);
  m_connectivity_list[ m_connectivity_index[lid] + index ] = target_lid;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void IncrementalItemConnectivity::
replaceConnectedItems(ItemLocalId source_item,Int32ConstArrayView target_local_ids)
{
  Int32 lid = source_item.localId();
  Integer n = target_local_ids.size();
  ARCANE_CHECK_AT(n,m_connectivity_nb_item[lid]);
  for( Integer i=0; i<n; ++i )
    m_connectivity_list[ m_connectivity_index[lid] + i ] = target_local_ids[i];
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

bool IncrementalItemConnectivity::
hasConnectedItem(Arcane::ItemLocalId source_item,
                 Arcane::ItemLocalId target_local_id) const
{
  bool has_connection = false;
  auto connected_items = _connectedItemsLocalId(source_item);
  if (std::find(connected_items.begin(),connected_items.end(),target_local_id) != connected_items.end()) has_connection = true;
  return has_connection;
}


/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void IncrementalItemConnectivity::
notifySourceItemAdded(ItemLocalId item)
{
  Int32 lid = item.localId();
  m_p->_checkResize(lid);
  _notifyConnectivityIndexChanged();
  _notifyConnectivityNbItemChanged();

  m_connectivity_nb_item[lid] = 0;
  m_connectivity_index[lid] = 0;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void IncrementalItemConnectivity::
notifyReadFromDump()
{
  m_pre_allocated_size = _sourceFamily()->properties()->getIntegerWithDefault(name()+"PreallocSize",0);
  info(4) << "PreallocSize2 var=" << m_p->m_var_name << " v=" << m_pre_allocated_size;

  // Il n'y a priori rien à faire pour les variables car via les observables sur les
  // variables les vues sont correctement mises à jour.
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void IncrementalItemConnectivity::
setPreAllocatedSize(Integer prealloc_size)
{
  if (m_pre_allocated_size<0)
    throw ArgumentException(A_FUNCINFO,
                            String::format("Invalid prealloc_size v={0}",
                                           prealloc_size));

  // Ne fait rien rien si on a déjà alloué des entités sinon cela rendrait
  // incohérent les allocations.
  // NOTE: on pourrait autoriser cela mais cela nécessiterait de reconstruire
  // les indices des connectivités. A priori un appel à compactConnectivityList()
  // suffirait.
  if (m_connectivity_nb_item.size()!=0)
    return;

  m_pre_allocated_size = prealloc_size;
  _sourceFamily()->properties()->setInteger(name()+"PreallocSize",prealloc_size);

  // Même s'il n'y a pas d'entités, m_p->m_connectivity_list_array n'est pas
  // vide car on appelé _checkkAddNulItem() dans le constructeur. Il faut
  // maintenant le réallouer avec la nouvelle valeur de pré-allocation.
  _resetConnectivityList();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void IncrementalItemConnectivity::
dumpStats(std::ostream& out) const
{
  size_t allocated_size = m_p->m_connectivity_list_array.capacity()
  + m_p->m_connectivity_index_array.capacity()
  + m_p->m_connectivity_nb_item_array.capacity();
  allocated_size *= sizeof(Int32);

  out << " connectiviy name=" << name()
      << " prealloc_size=" << m_pre_allocated_size
      << " nb_add=" << m_nb_add
      << " nb_remove=" << m_nb_remove
      << " nb_memcopy=" << m_nb_memcopy
      << " list_size=" << m_connectivity_list.size()
      << " index_size=" << m_connectivity_index.size()
      << " nb_item_size=" << m_connectivity_nb_item.size()
      << " allocated_size=" << allocated_size;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void IncrementalItemConnectivity::
_checkAddNullItem()
{
  // Si la liste des connectivités est vide, créé un élément
  // (ou plusieurs si m_pre_allocated_size>0) pour contenir l'entité nulle.
  // Cela permet de récupérer pour une entité la liste des connectivités même si
  // elle est vide.
  if (m_connectivity_list.size()==0){
    if (m_pre_allocated_size>0){
      _increaseConnectivityList(NULL_ITEM_LOCAL_ID,m_pre_allocated_size);
    }
    else{
      _increaseConnectivityList(NULL_ITEM_LOCAL_ID);
    }
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Compacte la liste des connectivités.
 *
 * L'implémentation actuelle est assez simple:
 * - Copie la liste actuelle dans un tableau temporaire.
 * - Vide la liste actuelle.
 * - Recopie dans la liste les valeurs utiles du tableau temporaire.
 *
 * \note L'appel à cette méthode suppose que les entités de la famille
 * source soient compactées.
 */
void IncrementalItemConnectivity::
compactConnectivityList()
{
  info(4) << "Begin Compacting IncrementalItemConnectivity name=" << name()
          << " new_size=" << m_connectivity_list.size()
          << " prealloc_size=" << m_pre_allocated_size;
  // TODO: essayer de trouver un moyen de ne faire le compactage que si
  // cela est nécessaire. Une facon serait de compter le nombre d'appel à
  // _increaseIndexList() depuis le dernier compactage.
  UniqueArray<Int32> old_connectivity_list(m_connectivity_list);
  Integer old_size = old_connectivity_list.size();
  Integer nb_item = m_connectivity_nb_item.size();
  m_p->m_connectivity_list_array.clear();
  _notifyConnectivityListChanged();
  _checkAddNullItem();
  Integer new_pos_in_list = m_p->m_connectivity_list_array.size();
  Int32 pre_allocated_size = m_pre_allocated_size;
  for( Integer i=0; i<nb_item; ++i ){
    Int32 lid = i;
    Int32 nb = m_connectivity_nb_item[lid];
    if (nb==0){
      m_connectivity_index[lid] = 0;
      continue;
    }
    Int32 index = m_connectivity_index[lid];
    Int32ConstArrayView con_list(nb,old_connectivity_list.data()+index);
    Integer alloc_size = _computeAllocSize(nb);
    m_connectivity_index[lid] = new_pos_in_list;
    new_pos_in_list += alloc_size;
    //info() << "NEW_POS_IN_LIST=" << new_pos_in_list << " nb=" << nb << " alloc_size=" << alloc_size;
    // Vérifie que la position est bien un multiple de pre_allocated_size.
    if (pre_allocated_size!=0){
      Int32 pos_modulo = new_pos_in_list % pre_allocated_size;
      if (pos_modulo!=0)
        ARCANE_FATAL("Bad position i={0} pos={1} pre_alloc_size={2} modulo={3}",
                     i,new_pos_in_list,pre_allocated_size,pos_modulo);
    }
    m_p->m_connectivity_list_array.addRange(con_list);
    // Si préallocation, complète le reste des éléments avec l'entité nulle..
    if (alloc_size!=nb)
      m_p->m_connectivity_list_array.addRange(NULL_ITEM_LOCAL_ID,alloc_size-nb);
    if (m_pre_allocated_size==0 && nb==0)
      m_connectivity_index[lid] = 0;
  }
  _notifyConnectivityListChanged();
  _computeMaxNbConnectedItem();
  info(4) << "Compacting IncrementalItemConnectivity name=" << name()
          << " nb_item=" << nb_item << " old_size=" << old_size
          << " new_size=" << m_connectivity_list.size()
          << " prealloc_size=" << m_pre_allocated_size;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

OneItemIncrementalItemConnectivity::
OneItemIncrementalItemConnectivity(IItemFamily* source_family,IItemFamily* target_family,
                                   const String& aname)
: IncrementalItemConnectivityBase(source_family,target_family,aname)
{
  info(4) << "Using fixed OneItem connectivity for name=" << name();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

OneItemIncrementalItemConnectivity::
~OneItemIncrementalItemConnectivity()
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void OneItemIncrementalItemConnectivity::
addConnectedItem(ItemLocalId source_item,ItemLocalId target_item)
{
  Int32 lid = source_item.localId();
  Integer size = m_connectivity_nb_item[lid];
  if (size!=0)
    ARCANE_FATAL("source_item already have connected items");
  Int32 target_lid = target_item.localId();
  m_connectivity_list[lid] = target_lid;
  m_connectivity_nb_item[lid] = 1;
  _setNewMaxNbConnectedItems(1);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void OneItemIncrementalItemConnectivity::
removeConnectedItems(ItemLocalId source_item)
{
  Int32 lid = source_item.localId();
  m_connectivity_nb_item[lid] = 0;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void OneItemIncrementalItemConnectivity::
removeConnectedItem(ItemLocalId source_item,ItemLocalId target_item)
{
  Int32 lid = source_item.localId();
  Int32 target_local_id = target_item.localId();
  Integer size = m_connectivity_nb_item[lid];
  if (size!=1)
    ARCANE_FATAL("source_item has no connected item");
  Int32 target_lid = m_connectivity_list[lid];
  if (target_lid!=target_local_id)
    ARCANE_FATAL("source_item is not connected to item with wanted_lid={0} current_lid={1}",
                 target_local_id,target_lid);
  m_connectivity_nb_item[lid] = 0;
  m_connectivity_list[lid] = NULL_ITEM_LOCAL_ID;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void OneItemIncrementalItemConnectivity::
replaceConnectedItem(ItemLocalId source_item,Integer index,ItemLocalId target_item)
{
  if (index!=0)
    ARCANE_FATAL("index has to be '0'");
  Int32 lid = source_item.localId();
  Int32 target_lid = target_item.localId();
  m_connectivity_list[lid] = target_lid;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void OneItemIncrementalItemConnectivity::
replaceConnectedItems(ItemLocalId source_item,Int32ConstArrayView target_local_ids)
{
  Int32 lid = source_item.localId();
  Integer n = target_local_ids.size();
  if (n==0)
    return;
  if (n!=1)
    ARCANE_FATAL("Invalid size for target_list. value={0} (expected 1)",n);
  m_connectivity_list[lid] = target_local_ids[0];
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

bool OneItemIncrementalItemConnectivity::
hasConnectedItem(ItemLocalId source_item,
                 ItemLocalId target_local_id) const
{
  if (m_connectivity_list[source_item.localId()] == target_local_id.localId())
    return true;
  else
    return false;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void OneItemIncrementalItemConnectivity::
_checkResizeConnectivityList()
{
  // Redimensionne la liste des connectivités avec le même nombre d'éléments
  // que le nombre d'entités.
  Integer wanted_size = m_connectivity_nb_item.size();
  Integer list_size = m_connectivity_list.size();
  if (list_size==wanted_size)
    return;
  Integer capacity = m_p->m_connectivity_list_array.capacity();
  if (wanted_size>=capacity){
    m_p->m_connectivity_list_array.reserve(m_p->m_connectivity_nb_item_array.capacity());
  }
  m_p->m_connectivity_list_array.resize(wanted_size);
  _notifyConnectivityListChanged();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void OneItemIncrementalItemConnectivity::
notifySourceItemAdded(ItemLocalId item)
{
  Int32 lid = item.localId();
  m_p->_checkResize(lid);
  _notifyConnectivityIndexChanged();
  _notifyConnectivityNbItemChanged();
  _checkResizeConnectivityList();

  m_connectivity_nb_item[lid] = 0;
  m_connectivity_index[lid] = lid;
  m_connectivity_list[lid] = NULL_ITEM_LOCAL_ID;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void OneItemIncrementalItemConnectivity::
notifySourceFamilyLocalIdChanged(Int32ConstArrayView new_to_old_ids)
{
  // Pour cette implémentation, il ne faut pas mettre à jour
  // les index car sinon on n'aura plus m_connectivity_index[lid] = lid.
  // TODO: comme à priori m_connectivity_nb_item vaut 1 partout, cela
  // n'est pas utile non plus de le faire sur cette variable mais
  // comme il peut y avoir des entités pour lesquelles nb_item vaut 0 si
  // on n'a pas ajouté d'entité connecté, il vaut mieux faire le compactage.

  m_p->m_connectivity_nb_item_variable.variable()->compact(new_to_old_ids);
  _notifyConnectivityNbItemChanged();

  // Comme avec cette implémentation la liste des connectivités est indexée
  // par le localId() de l'entité source, il faut la compacter
  // m_p->m_connectivity_list_variable.
  m_p->m_connectivity_list_variable.variable()->compact(new_to_old_ids);
  _notifyConnectivityListChanged();

  // Ne compacte pas les index mais mets tout de même à jour la taille
  // du tableau.
  m_p->m_connectivity_index_array.resize(m_connectivity_nb_item.size());
  _notifyConnectivityIndexChanged();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void OneItemIncrementalItemConnectivity::
notifyReadFromDump()
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void OneItemIncrementalItemConnectivity::
dumpStats(std::ostream& out) const
{
  size_t allocated_size = m_p->m_connectivity_list_array.capacity()
  + m_p->m_connectivity_index_array.capacity()
  + m_p->m_connectivity_nb_item_array.capacity();
  allocated_size *= sizeof(Int32);

  out << " connectiviy name=" << name()
      << " list_size=" << m_connectivity_list.size()
      << " index_size=" << m_connectivity_index.size()
      << " nb_item_size=" << m_connectivity_nb_item.size()
      << " allocated_size=" << allocated_size;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void OneItemIncrementalItemConnectivity::
compactConnectivityList()
{
  _computeMaxNbConnectedItem();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane::mesh

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
