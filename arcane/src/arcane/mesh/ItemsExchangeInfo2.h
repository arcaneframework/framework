// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2024 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ItemsExchangeInfo2.h                                        (C) 2000-2024 */
/*                                                                           */
/* Informations pour échanger des entités et leur caractéristiques.          */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_MESH_ITEMSEXCHANGEINFO2_H
#define ARCANE_MESH_ITEMSEXCHANGEINFO2_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/List.h"
#include "arcane/utils/TraceAccessor.h"

#include "arcane/VariableCollection.h"
#include "arcane/IItemFamilyExchanger.h"
#include "arcane/IItemFamilySerializeStep.h"
#include "arcane/ParallelExchangerOptions.h"

#include "arcane/mesh/MeshGlobal.h"

#include <set>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{
class ItemInternal;
class IParallelExchanger;
class IItemFamilySerializer;
class IItemFamilySerializeStep;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::mesh
{
class ItemGroupsSerializer2;
class TiedInterfaceExchanger;
class ItemFamilyVariableSerializer;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Informations pour échanger des entités d'une famille donnée
 * et leur caractéristiques.
 
 Une instance de cette classe contient toutes les informations pour
 échanger les entités du maillage \a m_mesh liées à la famille \a item_family.
 
 L'échange des entités se comporte différemment suivant le genre (eItemKind)
 de l'entité. Pour les mailles, la description complète de la connectivité
 est envoyé au sous-domaine récepteur. Pour les noeuds (Node), arêtes (Edge)
 et faces (Face), la connectivité n'est pas envoyée car elle est donnée
 par les mailles. Il n'est donc pas possible de sérialiser ces trois
 types d'entités indépendamment des mailles (ce qui serait de toutes
 facons pas cohérents). Pour les particules, est envoyé en plus
 le numéro de la maille à laquelle chaque particule appartient.

 Lorsque des mailles ou des particules sont envoyées, il faut
 appeler la méthode readAndAllocItems() pour les créér, avant
 d'appeler readGroups() puis readVariables().

 En plus des entités elles même, cette classe échange les valeurs des
 variables ainsi que les appartenances aux groupes.
*/
class ARCANE_MESH_EXPORT ItemsExchangeInfo2
: public TraceAccessor
, public IItemFamilyExchanger
{
 public:

  ItemsExchangeInfo2(IItemFamily* item_family);
  ~ItemsExchangeInfo2();

 public:

  void computeExchangeItems() override;
  
  void setExchangeItems(ConstArrayView< std::set<Int32> > items_to_send) override;

  /*!
   * \brief Détermine les informations nécessaires pour les échanges.
   * \retval true s'il n'y a rien à échanger
   * \retval false sinon.
   */
  bool computeExchangeInfos() override;

  //! Prépare les structures d'envoie
  void prepareToSend() override;
  void releaseBuffer() override;

  /*!
   * \brief Après réception des messages, lit et créé les entités transférées.
   *
   * Cette méthode ne fait rien pour les entités autre
   * que pour les mailles et les particules.
   *
   * \warning Avant d'appeler cette méthode, il faut être certain
   * que les entités n'appartenant plus à ce sous-domaine ont été
   * détruites
   */
  void readAndAllocItems() override;
  void readAndAllocSubMeshItems() override;
  void readAndAllocItemRelations() override;

  //! Après réception des messages, lit les groupes
  void readGroups() override;

  //! Après réception des messages, lit les valeurs des variables
  void readVariables() override;

  /*!
   * \brief Supprime les entités envoyées.
   *
   * Cette opération ne doit se faire que pour les entités qui
   * ne dépendent pas d'une autre entité. Par exemple, il est impossible
   * de supprimer directement les noeuds, car certaines mailles qui
   * ne sont pas envoyées peuvent reposer dessus.
   *
   * En pratique, cette opération n'est utile que pour les particules.
   */
  void removeSentItems() override;

  //! Envoie les messages d'échange
  void processExchange() override;

  /*!
   * \brief Termine l'échange.
   *
   * Effectue les dernières mises à jour suite à un échange. Cette
   * méthode est appelée lorsque toutes les entités et les variables
   * ont été échangées.
   */
  void finalizeExchange() override;

  IItemFamily* itemFamily() override { return m_item_family; }

  void setParallelExchangerOption(const ParallelExchangerOptions& options) override;

 public:

  void addSerializeStep(IItemFamilySerializeStep* step);

 private:

  IItemFamily* m_item_family;

  //! Liste des entités à envoyer à chaque processeur
  UniqueArray< SharedArray<Int32> > m_send_local_ids;

  //! Sérialiseur des groupes
  UniqueArray<ItemGroupsSerializer2*> m_groups_serializers;

  /*!
   * \brief Liste des familles intégrées à l'échange.
   *
   * Il s'agit de \a m_item_family et des ces familles filles
   * (à un seul niveau).
   */
  UniqueArray<IItemFamily*> m_families_to_exchange;

  Ref<IParallelExchanger> m_exchanger;

  /*!
   * \brief Liste des numéros locaux des entités reçues.
   */
  UniqueArray< SharedArray<Int32> > m_receive_local_ids;

  IItemFamilySerializer* m_family_serializer;

  UniqueArray<IItemFamilySerializeStep*> m_serialize_steps;

  ParallelExchangerOptions m_exchanger_option;

 private:

  inline void _addItemToSend(Int32 sub_domain_id,Item item);
  bool _computeExchangeInfos();
  void _applySerializeStep(IItemFamilySerializeStep::ePhase phase,
                           const ItemFamilySerializeArgs& args);
  void _applyDeserializePhase(IItemFamilySerializeStep::ePhase phase);
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  

