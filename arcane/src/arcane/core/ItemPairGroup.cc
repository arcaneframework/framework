// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ItemPairGroup.cc                                            (C) 2000-2025 */
/*                                                                           */
/* Tableau de listes d'entités.                                              */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/IFunctor.h"
#include "arcane/utils/NotImplementedException.h"

#include "arcane/core/ItemPairGroup.h"
#include "arcane/core/IItemFamily.h"
#include "arcane/core/ItemGroup.h"
#include "arcane/core/ItemPairEnumerator.h"
#include "arcane/core/ItemPairGroupBuilder.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \class ItemPairGroup
 * \ingroup Mesh
 * \brief Tableau de listes d'entités.
 *
 * Cette classe permet de gérer une liste d'entités associée à chaque entité
 * d'un groupe d'entité (ItemGroup). Par exemple pour chaque noeud d'un groupe l'ensemble
 * des mailles connectées à ce noeud par les faces.
 *
 * Cette classe a une sémantique par référence de la même manière que la
 * classe ItemGroup.
 *
 * %Arcane fournit un ensemble prédéfini de méthodes pour calculer les connectivités
 * des entités connectées à d'autres entités par un genre spécifique
 * d'entité. Pour utiliser ces méthodes il faut utiliser le
 * constructeur suivant:
 * ItemPairGroup(const ItemGroup& group,const ItemGroup& sub_item_group,
 * #eItemKind link_kind). \a link_kind indique alors le genre d'entité
 * qui le lien. Par exemple:
 *
 \code
 * CellGroup cells1;
 * CellGroup cells2;
 * // g1 contient pour chaque maille de \a cells1 les mailles qui lui
 * // sont connectés par les noeuds et qui appartiennent au groupe \a cells2
 * CellCellGroup g1(cells1,cells2,IK_Node);
 * ENUMERATE_ITEMPAIR(Cell,Cell,iitem,ad_list){
 *   Cell cell = *iitem;
 *   // Itère sur les mailles connectées à 'cell'
 *   ENUMERATE_SUB_ITEM(Cell,isubitem,iitem){
 *     Cell sub_cell = *iitem;
 *     ...
 *   }
 * }
 \endcode
 *
 * Il est possible pour l'utilisateur de spécifier une manière particulière
 * de calcul des connectivités en spécifiant un fonctor de type
 * ItemPairGroup::CustomFunctor comme argument du constructeur.
 *
 * \warning Le fonctor passé en argument doit être alloué par
 * l'opérateur new et sera détruit en même temps que le ItemPairGroup associé.
 *
 * Voici un exemple complet qui calcule les mailles
 * connectées aux mailles via les faces:
 *
 \code
 * auto f = [](ItemPairGroupBuilder& builder)
 *   {
 *     const ItemPairGroup& pair_group = builder.group();
 *     const ItemGroup& items = pair_group.itemGroup();
 *     const ItemGroup& sub_items = pair_group.subItemGroup();

 *     // Marque toutes les entités qui n'ont pas le droit d'appartenir à
 *     // la liste des connectivités car elles ne sont pas dans \a sub_items;
 *     std::set<Int32> allowed_ids;
 *     ENUMERATE_CELL(iitem,sub_items) {
 *       allowed_ids.insert(iitem.itemLocalId());
 *     }

 *     Int32Array local_ids;
 *     local_ids.reserve(8);

 *     // Liste des entités déjà traitées pour la maille courante
 *     std::set<Int32> already_in_list;
 *     ENUMERATE_CELL(icell,items){
 *       Cell cell = *icell;
 *       local_ids.clear();
 *       Int32 current_local_id = icell.itemLocalId();
 *       already_in_list.clear();

 *       // Pour ne pas s'ajouter à sa propre liste de connectivité
 *       already_in_list.insert(current_local_id);

 *       for( FaceEnumerator iface(cell.faces()); iface.hasNext(); ++iface ){
 *         Face face = *iface;
 *         for( CellEnumerator isubcell(face.cells()); isubcell.hasNext(); ++isubcell ){
 *           const Int32 sub_local_id = isubcell.itemLocalId();
 *          // Vérifie qu'on est dans la liste des mailles autorisées et qu'on
 *           // n'a pas encore été traité.
 *           if (allowed_ids.find(sub_local_id)==allowed_ids.end())
 *             continue;
 *           if (already_in_list.find(sub_local_id)!=already_in_list.end())
 *             continue;
 *           // Cette maille doit être ajoutée. On la marque pour ne pas
 *           // la parcourir et on l'ajoute à la liste.
 *           already_in_list.insert(sub_local_id);
 *           local_ids.add(sub_local_id);
 *         }
 *       }
 *       builder.addNextItem(local_ids);
 *     }
 *   };
 *
 * // Créé un groupe qui calcule les connectivités sur toutes les mailles.
 * ItemPairGroupT<Cell,Cell> ad_list(allCells(),allCells(),functor::makePointer(f));
 \endcode
 */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \internal
 * \brief Wrapper sur un fonctor ItemPairGroup::CustomFunctor.
 */
class ItemPairGroup::CustomFunctorWrapper
: public IFunctor
{
 public:
  CustomFunctorWrapper(ItemPairGroupImpl* g,ItemPairGroup::CustomFunctor* f)
  : m_group(g), m_functor(f){}
  ~CustomFunctorWrapper()
  {
    delete m_functor;
  }
 public:
  void executeFunctor() override
  {
    ItemPairGroup pair_group(m_group);
    ItemPairGroupBuilder builder(pair_group);
    m_functor->executeFunctor(builder);
  }
 public:

  ItemPairGroupImpl* m_group = nullptr;
  ItemPairGroup::CustomFunctor* m_functor = nullptr;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ItemPairGroup::
ItemPairGroup(const ItemGroup& group,const ItemGroup& sub_item_group,
              eItemKind link_kind)
: m_impl(nullptr)
{
  IItemFamily* item_family = group.itemFamily();
  ItemPairGroup v = item_family->findAdjacencyItems(group, sub_item_group, link_kind, 1);
  m_impl = v.internal();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ItemPairGroup::
ItemPairGroup(const ItemGroup& group,const ItemGroup& sub_item_group,
              CustomFunctor* functor)
: m_impl(nullptr)
{
  ARCANE_CHECK_POINTER(functor);
  m_impl = new ItemPairGroupImpl(group,sub_item_group);
  IFunctor* f = new CustomFunctorWrapper(m_impl.get(),functor);
  m_impl->setComputeFunctor(f);
  m_impl->invalidate(true);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ItemPairGroup::
ItemPairGroup(ItemPairGroupImpl* p)
: m_impl(p)
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ItemPairGroup::
ItemPairGroup()
: m_impl(ItemPairGroupImpl::checkSharedNull())
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ItemPairEnumerator ItemPairGroup::
enumerator() const
{
  return ItemPairEnumerator(*this);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
