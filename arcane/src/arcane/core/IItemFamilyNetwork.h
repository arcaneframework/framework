// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2024 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* IItemFamilyNetwork.h                                        (C) 2000-2025 */
/*                                                                           */
/* Interface to handle ItemFamily relations through their connectivities.    */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_IITEMFAMILYNETWORK_H_ 
#define ARCANE_IITEMFAMILYNETWORK_H_ 
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/core/IItemFamily.h"
#include "arcane/core/IIncrementalItemConnectivity.h"
#include "arcane/core/IGraph2.h"

#include <functional>
#include <set>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class IItemFamilyNetwork
{
 public:

  virtual ~IItemFamilyNetwork() = default;

 public:

  typedef std::function<void(IItemFamily*)> IItemFamilyNetworkTask;

 public:

  enum eSchedulingOrder
  {
    TopologicalOrder,
    InverseTopologicalOrder,
    Unknown
  };

  // TMP for debug
 public:

  static constexpr bool plug_serializer = true;

 public:

  virtual bool isActivated() const = 0;

  /*!
   * \brief Ajoute une dépendance entre deux familles ; un élément de \a master_family est constitué d'éléments de \a slave_family.
   *  La responsabilité de la mémoire de \a master_to_slave_connectivity est prise en charge par ItemFamilyNetwork
   */
  virtual void addDependency(IItemFamily* master_family, IItemFamily* slave_family,
                             IIncrementalItemConnectivity* slave_to_master_connectivity,
                             bool is_deep_connectivity = true) = 0;

  /*!
   * \brief Ajoute une relation entre deux familles ; un élément de \a source_family est connecté à un ou plusieurs éléments de \a target_family
   *  La responsabilité de la mémoire de \a source_to_target_connectivity est prise en charge par ItemFamilyNetwork
   */
  virtual void addRelation(IItemFamily* source_family,
                           IItemFamily* target_family,
                           IIncrementalItemConnectivity* source_to_target_connectivity) = 0;

  //! Retourne la connectivité de dépendance entre la famille \a source_family et \a target_family
  virtual IIncrementalItemConnectivity* getDependency(IItemFamily* source_family, IItemFamily* target_family) = 0;
  virtual IIncrementalItemConnectivity* getRelation(IItemFamily* source_family, IItemFamily* target_family) = 0;

  //! Retourne la connectivité entre les familles \a source_family et \a target_family de nom \a name, qu'elle soit une relation ou une dépendance
  virtual IIncrementalItemConnectivity* getConnectivity(IItemFamily* source_family,
                                                        IItemFamily* target_family,
                                                        const String& name) = 0;
  virtual IIncrementalItemConnectivity* getConnectivity(IItemFamily* source_family,
                                                        IItemFamily* target_family,
                                                        const String& name,
                                                        bool& is_dependency) = 0;

  /*!
   * \brief Retourne, si elle est associée à un stockage, la connectivité entre les
   * familles \a source_family et \a target_family de nom \a name,
   * qu'elle soit une relation ou une dépendance.
   */
  virtual IIncrementalItemConnectivity* getStoredConnectivity(IItemFamily* source_family,
                                                              IItemFamily* target_family,
                                                              const String& name) = 0;
  virtual IIncrementalItemConnectivity* getStoredConnectivity(IItemFamily* source_family,
                                                              IItemFamily* target_family,
                                                              const String& name,
                                                              bool& is_dependency) = 0;

  //! Obtenir la liste de toutes les connectivités, qu'elles soient relation ou dépendance
  virtual List<IIncrementalItemConnectivity*> getConnectivities() = 0;

  //! Obtenir la liste de toutes les connectivités (dépendances ou relations), filles d'une famille \a source_family ou parentes d'une famille \a target_family
  virtual SharedArray<IIncrementalItemConnectivity*> getChildConnectivities(IItemFamily* source_family) = 0;
  virtual SharedArray<IIncrementalItemConnectivity*> getParentConnectivities(IItemFamily* target_family) = 0;

  //! Obtenir la liste de toutes les dépendances, filles d'une famille \a source_family ou parentes d'une famille \a target_family
  virtual SharedArray<IIncrementalItemConnectivity*> getChildDependencies(IItemFamily* source_family) = 0;
  virtual SharedArray<IIncrementalItemConnectivity*> getParentDependencies(IItemFamily* target_family) = 0;

  //! Obtenir la liste de toutes les relations, filles d'une famille \a source_family ou parentes d'une famille \a target_family
  virtual SharedArray<IIncrementalItemConnectivity*> getChildRelations(IItemFamily* source_family) = 0;
  virtual SharedArray<IIncrementalItemConnectivity*> getParentRelations(IItemFamily* source_family) = 0;

  //! Obtenir la liste de toutes les familles
  virtual const std::set<IItemFamily*>& getFamilies() const = 0;

  virtual SharedArray<IItemFamily*> getFamilies(eSchedulingOrder order) const = 0;

  //! Ordonnance l'exécution d'une tâche, dans l'ordre topologique ou topologique inverse du graphe de dépendance des familles
  virtual void schedule(IItemFamilyNetworkTask task, eSchedulingOrder order = TopologicalOrder) = 0;

  //! Positionne une connectivité comme étant stockée.
  virtual void setIsStored(IIncrementalItemConnectivity* connectivity) = 0;

  //! Récupère l'information relative au stockage de la connectivité
  virtual bool isStored(IIncrementalItemConnectivity* connectivity) = 0;

  //! Récupère l'information relative au stockage de la connectivité
  virtual bool isDeep(IIncrementalItemConnectivity* connectivity) = 0;

  //! enregistre un graphe gérant des DoFs connectés au maillage
  virtual Integer registerConnectedGraph(IGraph2* graph) = 0;

  //! dé enregistre un graphe gérant des DoFs connectés au maillage
  virtual void releaseConnectedGraph(Integer graph_id) = 0;

  //! supprime les DoFs et les liens entre DoFs connectés aux mailles supprimées
  virtual void removeConnectedDoFsFromCells(Int32ConstArrayView local_ids) = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif /* IITEMFAMILYNETWORK_H_ */
