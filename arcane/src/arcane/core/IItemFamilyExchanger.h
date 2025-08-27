// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2024 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* IItemFamilyExchanger.h                                      (C) 2000-2024 */
/*                                                                           */
/* Echange entre sous-domaine les entités d'une famille.                     */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_MESH_IITEMFAMILYEXCHANGER_H
#define ARCANE_MESH_IITEMFAMILYEXCHANGER_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/List.h"
#include "arcane/utils/TraceAccessor.h"

#include "arcane/core/VariableCollection.h"

#include "arcane/mesh/MeshGlobal.h"

#include <set>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class ParallelExchangerOptions;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \internal
 * \brief Échange des entités et leurs caractéristiques pour une famille donnée
 
 Cette classe gère l'échange d'entités entre les sous-domaines. Elle est
 utilisée par exemple lors d'un repartitionnement. En général cette classe
 n'est pas utilisée directement (sauf pour spécifier les entités à échanger)
 mais via l'interface IMeshExchanger.
 
 L'utilisateur de cette classe doit commencer par spécifier la liste des
 entités à envoyer à chaque sous-domaine via la méthode setExchangeItems().

 L'échange d'entités se fait en plusieurs étapes comme indiquée dans
 IMeshExchanger.

 La sérialisation proprement dite des entités se fait en trois phases
 successives: les entités, les groupes et les variables. La désérialisation
 se fait dans le même ordre. En effet, il est nécessaire pour désérialiser
 les variables de connaitre les groupes et pour désérialiser les groupes
 de connaître les entités.

 Lorsque des mailles ou des particules sont envoyées, il faut
 appeler la méthode readAndAllocItems() pour les créér, avant
 d'appeler readGroups() puis readVariables().
*/
class ARCANE_CORE_EXPORT IItemFamilyExchanger
{
 public:

  virtual ~IItemFamilyExchanger(){}

 public:

  /*!
   * \internal
   * \brief Détermine la liste des entités à échanger.

   * \warning Cette méthode ne doit être utilisée que pour les familles
   * de particules.

   Cette opération se sert de la variable itemsOwner() et du champ
   owner() de chaque entité pour déterminer à qui chaque entité doit
   être envoyée. Par conséquent, il faut appeler cette opération
   avant que DynamicMesh::_setOwnerFromVariable() ne soit appelé.
   *
   * \todo A supprimer
   */
  virtual void computeExchangeItems() =0;
  
  //! Positionne la liste des entités à échanger.
  virtual void setExchangeItems(ConstArrayView< std::set<Int32> > items_to_send) =0;

  /*!
   * \brief Détermine les informations nécessaires pour les échanges.
   * \retval true s'il n'y a rien à échanger
   * \retval false sinon.
   */
  virtual bool computeExchangeInfos() =0;

  //! Prépare les structures d'envoie
  virtual void prepareToSend() =0;
  virtual void releaseBuffer() =0;

  /*!
   * \brief Après réception des messages, lit et créé les entités transférées.
   *
   * Cette méthode ne fait rien pour les entités autre
   * que pour les mailles et les particules, pour la gestion legacy.
   * Avec le graphe des familles ItemFamilyNetwork, cette méthode crée les
   * items et leur dépendances (ie connectivités descendantes).
   * Cela implique de séparer le traitement des sous-items (sous-maillages)
   * et des relations (connectivités ascendantes ou dof), qui ne peuvent
   * être traités tant que tous les items ne sont pas créés.
   *
   * \warning Avant d'appeler cette méthode, il faut être certain
   * que les entités n'appartenant plus à ce sous-domaine ont été
   * détruites
   */
  virtual void readAndAllocItems() =0;
  virtual void readAndAllocSubMeshItems() =0;
  virtual void readAndAllocItemRelations() =0;

  //! Après réception des messages, lit les groupes
  virtual void readGroups() =0;

  //! Après réception des messages, lit les valeurs des variables
  virtual void readVariables() =0;

  /*!
   * \internal
   * \brief Supprime les entités envoyées.
   *
   * Cette opération ne doit se faire que pour les entités qui
   * ne dépendent pas d'une autre entité. Par exemple, il est impossible
   * de supprimer directement les noeuds, car certaines mailles qui
   * ne sont pas envoyées peuvent reposer dessus.
   *
   * \warning Cette opération n'est valide que pour les particules sans
   * notion de particule fantôme.
   * \todo A supprimer
   */
  virtual void removeSentItems() =0;

  //! Envoie les messages d'échange
  virtual void processExchange() =0;

  /*!
   * \brief Termine l'échange.
   *
   * Effectue les dernières mises à jour suite à un échange. Cette
   * méthode est appelée lorsque toutes les entités et les variables
   * ont été échangées.
   */
  virtual void finalizeExchange() =0;

  //! Famille associée
  virtual IItemFamily* itemFamily() =0;

  //! Positionne les options utilisées lors de l'échange des entités
  virtual void setParallelExchangerOption(const ParallelExchangerOptions& options) =0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  

