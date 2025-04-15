// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ItemGroup.cc                                                (C) 2000-2025 */
/*                                                                           */
/* Groupes d'entités du maillage.                                            */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/core/ItemGroup.h"

#include "arcane/utils/String.h"
#include "arcane/utils/ArgumentException.h"

#include "arcane/core/IItemFamily.h"
#include "arcane/core/IMesh.h"
#include "arcane/core/ICaseMng.h"
#include "arcane/core/CaseOptionBase.h"
#include "arcane/core/ICaseOptionList.h"
#include "arcane/core/MeshHandle.h"
#include "arcane/core/internal/ItemGroupInternal.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \class ItemGroup
 
 Un groupe d'entité du maillage est une liste d'entités
 d'une même famille (IItemFamily).

 Une entité ne peut être présente qu'une seule fois dans un groupe.

 Une instance de cette classe possède une référence sur un groupe
 quelconque d'entités du maillage. Il est possible à partir de cette référence
 de connaître le genre (itemKind(), le nom (name()) et le nombre d'éléments
 (size()) du groupe et d'itérer de manière générique sur les éléments qui le compose.
 Pour itérer sur les éléments dérivés (mailles, noeuds, ...), il faut d'abord le
 convertir en une référence sur un groupe spécifique (NodeGroup, FaceGroup,
 EdgeGroup ou CellGroup). Par exemple:
 \code
 ItemGroup group = subDomain()->defaultMesh()->findGroup("Surface");
 FaceGroup surface(surface);
 if (surface.null())
   // Pas une surface.
 if (surface.empty())
   // Surface existe mais est vide
 \endcode

 Il est possible de trier un groupe pour que ses éléments soient
 toujours être classés par ordre
 croissant des uniqueId() des éléments, afin de garantir que les codes
 séquentiels et parallèles peuvent donner le même résultat.

 Il existe un groupe spécial, dit groupe nul, permettant de représenter
 un groupe non référencé, c'est à dire qui n'existe pas. Ce groupe est
 le seul pour lequel null() retourne \c true. Le groupe nul possède
 les propriétés suivantes:
 \arg null() == \c true;
 \arg size() == \c 0;
 \arg name().null() == \c true;

 Cette classe utilise un compteur de référence et s'utilise donc par
 référence. Par exemple:
 \code
 ItemGroup a = subDomain()->defaultMesh()->findGroup("Toto");
 ItemGroup b = a; // b et a font référence au même groupe.
 if (a.null())
   // Groupe pas trouvé...
   ;
 \endcode

 Pour parcourir les entités d'un groupe, il faut utiliser un énumérateur,
 par l'intermédiaire des macros ENUMERATE_*, par exemple ENUMERATE_CELL
 pour un groupe de mailles:
 \code
 * CellGroup g;
 * ENUMERATE_CELL(icell,g){
 *   m_mass[icell] = m_volume[icell] * m_density[icell];
 * }
 \endcode

 Il est possible d'ajouter (addItems()) ou supprimer des entités
 d'un groupe (removeItems()).

 Les groupes qui n'ont pas de parents sont persistants et peuvent
 être récupérés lors d'une reprise. Les éléments de ces groupes sont
 automatiquement mis à jour lors de la modification de la famille
 associée. Par exemple, si un élément d'une famille est supprimé
 et qu'il appartenait à un groupe, il est automatiquement supprimé
 de ce groupe. De même les groupes sont mis à jour lors d'un
 repartitionnement du maillage. Il existe cependant une petite restriction
 avec l'implémentation actuelle sur cette utilisation. Pour éviter de remettre
 à jour le groupe à chaque changement de la famille, le groupe est marqué
 comme devant être remis à jour (via invalidate()) à chaque changement
 mais n'est réellement recalculé que lorsqu'il sera utilisé. Il est
 donc théoriquement possible que des ajouts et suppressions multiples
 entre deux utilisations du groupe rendent ses éléments incohérents
 (TODO: lien sur explication detaillée). Pour éviter ce problème, il
 est possible de forcer le recalcul du groupe en appelant invalidate()
 avec comme argument \a true.

 Les groupes dits dérivés (qui ont un parent) comme les own() ou
 les cellGroup() sont invalidés et vidés de leurs éléments lors d'une
 modification de la famille associée.

 Si un groupe est utilisé comme support pour des variables partielles, alors
 les entités appartenant au groupe doivent être cohérentes entre les
 sous-domaines. C'est à dire que si une entité \a x est présente dans
 plusieurs sous-domaines (que soit en tant qu'entité propre ou
 fantôme), il faut qu'elle soit dans ce groupe pour tous les
 sous-domaines ou dans aucun des groupes. Par exemple, si la maille de
 uniqueId() 238 est présente les sous-domaines 1, 4 et 8 et que
 pour le sous-domaine 4 elle est dans le groupe de mailles 'TOTO',
 alors il faut aussi qu'elle soit dans ce groupe de mailles 'TOTO'
 pour les sous-domaines 1 et 8.
*/
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ItemGroup::
ItemGroup(ItemGroupImpl* grp)
: m_impl(grp)
{
  // Si \a grp est nul, le remplace par le groupe nul.
  // Cela est fait (version 2.3) pour des raisons de compatibilité.
  // A terme, ce constructeur sera explicite et dans ce cas il
  // faudra faire:
  //   ARCANE_CHECK_POINTER(grp);
  if (!grp){
    std::cerr << "Creating group with null pointer is not allowed\n";
    m_impl = ItemGroupImpl::checkSharedNull();
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ItemGroup::
ItemGroup()
: m_impl(ItemGroupImpl::checkSharedNull())
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

bool ItemGroup::
isOwn() const
{
  if (null())
    return true;

  return m_impl->isOwn();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ItemGroup::
setOwn(bool v)
{
  if (!null())
    m_impl->setOwn(v);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Groupe équivalent à celui-ci mais contenant
 * uniquement les éléments propres au sous-domaine.
 *
 * Si ce groupe est déjà un groupe ne contenant que des éléments propres
 * au sous-domaine, c'est lui même qui est retourné:
 * \code
 * group.own()==group; // Pour un groupe local
 * group.own().own()==group.own(); // Invariant
 * \endcode
 */
ItemGroup ItemGroup::
own() const
{
  if (null() || isOwn())
    return (*this);
  m_impl->checkNeedUpdate();
  return ItemGroup(m_impl->ownGroup());
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

// group of items owned by the subdomain
ItemGroup ItemGroup::
ghost() const
{
  if (null())
  	return ItemGroup();
  m_impl->checkNeedUpdate();
  return ItemGroup(m_impl->ghostGroup());
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

// Items in the group lying on the boundary between two subdomains
// Implemented for faces only
ItemGroup ItemGroup::
interface() const
{
  if (null())
  	return ItemGroup();
  m_impl->checkNeedUpdate();
  return ItemGroup(m_impl->interfaceGroup());
}

// GERER PLANTAGE SORTED
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

NodeGroup ItemGroup::
nodeGroup() const
{
  if (null())
    return ItemGroup();
  m_impl->checkNeedUpdate();
  return NodeGroup(m_impl->nodeGroup());
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

EdgeGroup ItemGroup::
edgeGroup() const
{
  if (null())
    return ItemGroup();
  m_impl->checkNeedUpdate();
  return EdgeGroup(m_impl->edgeGroup());
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

FaceGroup ItemGroup::
faceGroup() const
{
  if (null())
    return ItemGroup();
  m_impl->checkNeedUpdate();
  return FaceGroup(m_impl->faceGroup());
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

CellGroup ItemGroup::
cellGroup() const
{
  if (null())
    return ItemGroup();
  m_impl->checkNeedUpdate();
  return CellGroup(m_impl->cellGroup());
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

FaceGroup ItemGroup::
innerFaceGroup() const
{
  if (null())
    return ItemGroup();
  m_impl->checkNeedUpdate();
  return FaceGroup(m_impl->innerFaceGroup());
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

FaceGroup ItemGroup::
outerFaceGroup() const
{
  if (null())
    return ItemGroup();
  m_impl->checkNeedUpdate();
  return FaceGroup(m_impl->outerFaceGroup());
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
//! AMR
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

CellGroup ItemGroup::
activeCellGroup() const
{
  if (null())
    return ItemGroup();
  m_impl->checkNeedUpdate();
  return CellGroup(m_impl->activeCellGroup());
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

CellGroup ItemGroup::
ownActiveCellGroup() const
{
  if (null())
    return ItemGroup();
  m_impl->checkNeedUpdate();
  return CellGroup(m_impl->ownActiveCellGroup());
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

CellGroup ItemGroup::
levelCellGroup(const Integer& level) const
{
  if (null())
    return ItemGroup();
  m_impl->checkNeedUpdate();
  return CellGroup(m_impl->levelCellGroup(level));
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

CellGroup ItemGroup::
ownLevelCellGroup(const Integer& level) const
{
  if (null())
    return ItemGroup();
  m_impl->checkNeedUpdate();
  return CellGroup(m_impl->ownLevelCellGroup(level));
}
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

FaceGroup ItemGroup::
activeFaceGroup() const
{
  if (null())
    return ItemGroup();
  m_impl->checkNeedUpdate();
  return FaceGroup(m_impl->activeFaceGroup());
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

FaceGroup ItemGroup::
ownActiveFaceGroup() const
{
  if (null())
    return ItemGroup();
  m_impl->checkNeedUpdate();
  return FaceGroup(m_impl->ownActiveFaceGroup());
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

FaceGroup ItemGroup::
innerActiveFaceGroup() const
{
  if (null())
    return ItemGroup();
  m_impl->checkNeedUpdate();
  return FaceGroup(m_impl->innerActiveFaceGroup());
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

FaceGroup ItemGroup::
outerActiveFaceGroup() const
{
  if (null())
    return ItemGroup();
  m_impl->checkNeedUpdate();
  return FaceGroup(m_impl->outerActiveFaceGroup());
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ItemGroup ItemGroup::
createSubGroup(const String & suffix, IItemFamily * family, ItemGroupComputeFunctor * functor) const
{
  if (null())
    return ItemGroup();
  m_impl->checkNeedUpdate();
  return ItemGroup(m_impl->createSubGroup(suffix,family,functor));
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ItemGroup ItemGroup::
findSubGroup(const String & suffix) const
{
  if (null())
    return ItemGroup();
  return ItemGroup(m_impl->findSubGroup(suffix));
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ItemGroup::
clear()
{
  m_impl->clear();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * Ajoute les entités de numéros locaux \a items_local_id.
 *
 * La paramètre \a check_if_present indique s'il vaut vérifier si les entités
 * à ajouter sont déjà présentes dans le groupe, auquel cas elles ne sont
 * pas ajouter. Si l'appelant est certain que les entités à ajouter
 * ne sont pas actuellement dans le groupe, il peut positionner le
 * paramètre \a check_if_present à \a false ce qui accélère l'ajout.
 */
void ItemGroup::
addItems(Int32ConstArrayView items_local_id,bool check_if_present)
{
  if (null())
    throw ArgumentException(A_FUNCINFO,"Can not addItems() to null group");
  if (isAllItems())
    throw ArgumentException(A_FUNCINFO,"Can not addItems() to all-items group");
  m_impl->_checkNeedUpdateNoPadding();
  m_impl->addItems(items_local_id,check_if_present);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * Supprime les entités de numéros locaux \a items_local_id.
 *
 * La paramètre \a check_if_present indique s'il vaut vérifier si les entités
 * à supprimer ne sont déjà présentes dans le groupe, auquel cas elles ne sont
 * pas supprimées. Si l'appelant est certain que les entités à supprimer
 * sont dans le groupe, il peut positionner le
 * paramètre \a check_if_present à \a false ce qui accélère la suppression.
 */
void ItemGroup::
removeItems(Int32ConstArrayView items_local_id,bool check_if_present)
{ 
  if (null())
    throw ArgumentException(A_FUNCINFO,"Can not removeItems() to null group");
  if (isAllItems())
    throw ArgumentException(A_FUNCINFO,"Can not removeItems() to all-items group");
  m_impl->_checkNeedUpdateNoPadding();
  m_impl->removeItems(items_local_id,check_if_present);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Positionne les entités du groupe.
 *
 * Positionne les entités dont les numéros locaux sont donnés par
 * \a items_local_id.
 * L'appelant garanti que chaque entité n'est présente qu'une fois dans
 * ce tableau
 */
void ItemGroup::
setItems(Int32ConstArrayView items_local_id)
{
  if (null())
    throw ArgumentException(A_FUNCINFO,"Can not setItems() to null group");
  m_impl->setItems(items_local_id);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Positionne les entités du groupe.
 *
 * Positionne les entités dont les numéros locaux sont donnés par
 * \a items_local_id.
 * L'appelant garanti que chaque entité n'est présente qu'une fois dans
 * ce tableau
 * Si \a do_sort est vrai, les entités sont triées par uniqueId croissant
 * avant d'être ajoutées au groupe.
 */
void ItemGroup::
setItems(Int32ConstArrayView items_local_id,bool do_sort)
{
  if (null())
    throw ArgumentException(A_FUNCINFO,"Can not setItems() to null group");
  m_impl->setItems(items_local_id,do_sort);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Vérification interne de la validité du groupe.
 */
void ItemGroup::
checkValid()
{
  m_impl->checkValid();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ItemGroup::
applyOperation(IItemOperationByBasicType* operation) const
{
  if (null())
    return;
  m_impl->checkNeedUpdate();
  m_impl->applyOperation(operation);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ItemEnumerator ItemGroup::
enumerator() const
{
  if (null())
    return ItemEnumerator();
  m_impl->_checkNeedUpdateNoPadding();
  return ItemEnumerator(m_impl->itemInfoListView(),m_impl->itemsLocalId(),m_impl.get());
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ItemEnumerator ItemGroup::
_simdEnumerator() const
{
  if (null())
    return ItemEnumerator();
  m_impl->_checkNeedUpdateWithPadding();
  return ItemEnumerator(m_impl->itemInfoListView(),m_impl->itemsLocalId(),m_impl.get());
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ItemVectorView ItemGroup::
_view(bool do_padding) const
{
  if (null())
    return ItemVectorView();
  m_impl->_checkNeedUpdate(do_padding);
  Int32 flags = 0;
  if (m_impl->isContigousLocalIds())
    flags |= ItemIndexArrayView::F_Contigous;
  // TODO: gérer l'offset
  return ItemVectorView(m_impl->itemFamily(),ItemIndexArrayView(m_impl->itemsLocalId(),0,flags));
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ItemVectorView ItemGroup::
view() const
{
  return _view(true);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ItemVectorView ItemGroup::
_paddedView() const
{
  return _view(true);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ItemVectorView ItemGroup::
_unpaddedView() const
{
  return _view(false);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

bool ItemGroup::
isAllItems() const
{
  return m_impl->isAllItems();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

IVariableSynchronizer* ItemGroup::
synchronizer() const
{
  return m_impl->synchronizer();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

bool ItemGroup::
isAutoComputed() const
{
  return m_impl->hasComputeFunctor();
}
  
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

bool ItemGroup::
hasSynchronizer() const 
{
  return m_impl->hasSynchronizer();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ItemGroupImplInternal* ItemGroup::
_internalApi() const
{
  return m_impl->_internalApi();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

bool ItemGroup::
checkIsSorted() const
{
  m_impl->_checkNeedUpdate(false);
  return m_impl->checkIsSorted();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ItemGroup::
incrementTimestamp() const
{
  m_impl->m_p->updateTimestamp();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

extern ARCANE_CORE_EXPORT bool
_caseOptionConvert(const CaseOptionBase& co,const String& name,ItemGroup& obj)
{
  IMesh* mesh = co.parentOptionList()->meshHandle().mesh();
  obj = mesh->findGroup(name);
  return obj.null();
}

extern ARCANE_CORE_EXPORT bool
_caseOptionConvert(const CaseOptionBase& co,const String& name,NodeGroup& obj)
{
  IMesh* mesh = co.parentOptionList()->meshHandle().mesh();
  obj = mesh->nodeFamily()->findGroup(name);
  return obj.null();
}

extern ARCANE_CORE_EXPORT bool
_caseOptionConvert(const CaseOptionBase& co,const String& name,EdgeGroup& obj)
{
  IMesh* mesh = co.parentOptionList()->meshHandle().mesh();
  obj = mesh->edgeFamily()->findGroup(name);
  return obj.null();
}

extern ARCANE_CORE_EXPORT bool
_caseOptionConvert(const CaseOptionBase& co,const String& name,FaceGroup& obj)
{
  IMesh* mesh = co.parentOptionList()->meshHandle().mesh();
  obj = mesh->faceFamily()->findGroup(name);
  return obj.null();
}

extern ARCANE_CORE_EXPORT bool
_caseOptionConvert(const CaseOptionBase& co,const String& name,CellGroup& obj)
{
  IMesh* mesh = co.parentOptionList()->meshHandle().mesh();
  obj = mesh->cellFamily()->findGroup(name);
  return obj.null();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
