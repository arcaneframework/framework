// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* IMeshUtilities.h                                            (C) 2000-2025 */
/*                                                                           */
/* Interface d'une classe proposant des fonctions utilitaires sur maillage.  */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_IMESHUTILITIES_H
#define ARCANE_CORE_IMESHUTILITIES_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/Real3.h"

#include "arcane/core/ArcaneTypes.h"
#include "arcane/core/ItemTypes.h"
#include "arcane/core/VariableTypedef.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Interface d'une classe proposant des fonctions utilitaires sur maillage.
 */
class ARCANE_CORE_EXPORT IMeshUtilities
{
 public:

  virtual ~IMeshUtilities() {} //!< Libère les ressources.

 public:

  /*!
   * \brief Recherche les identifiants locaux des entités à partir
   * de leur connectivité.
   *
   * Prend en entrée une liste d'entités décrite par les identifiants uniques
   * (Item::uniqueId()) de leurs noeuds et recherche les identifiants locaux (Item::localId())
   * de ces entités.
   *
   * \param item_kind Genre de l'entité (IK_Cell ou IK_Face)
   * \param items_nb_node tableau du nombre de noeuds de l'entité
   * \param items_connectivity tableau contenant les indices uniques des noeuds des entités.
   * \param local_ids en retour, contient les identificants locaux des
   * entités. Le nombre d'éléments de \a local_ids doit être égal à
   * celui de \a items_nb_node.
   *
   * Le tableau \a items_connectivity contient les identifiants des noeuds des entités,
   * rangés consécutivement. Par exemple, si \c items_nb_node[0]==3 et
   * \c items_node[1]==4, alors \a items_connectivity[0..2] contiendra les
   * noeuds de l'entité 0, et items_connectivity[3..6] ceux de l'entité 1.
   *
   * Si \a allow_null est faux, une erreur fatale est générée si
   * une entité n'est pas trouvée, sinon NULL_ITEM_LOCAL_ID est
   * retourné pour l'entité correspondante
   */
  virtual void localIdsFromConnectivity(eItemKind item_kind,
                                        IntegerConstArrayView items_nb_node,
                                        Int64ConstArrayView items_connectivity,
                                        Int32ArrayView local_ids,
                                        bool allow_null=false) =0;

  /*!
   * \brief Calcule la normale d'un groupe de face.
   *
   * Cette méthode calcule la normale à un groupe de face en considérant que
   * cette surface est un plan. Pour le calcul, l'algorithme essaie de
   * déterminer les noeuds aux extrémités de cette surface, et calcule une
   * normale à partir de ces noeuds. L'orientation de la normale (rentrante
   * ou sortante) est indéfinie.
   *
   * Si la surface n'est pas plane, le résultat est indéfini.
   *
   * L'algorithme actuel ne fonctionne pas toujours sur une surface composée
   * uniquement de triangles.
   *
   * Cette méthode est collective. L'algorithme utilisé garantit les
   * mêmes résultats en séquentiel et en parallèle.
   *
   * La variable \a nodes_coord est utilisée comme coordonnées pour les noeuds.
   * En général, il s'agit de IMesh::nodesCoordinates().
   */
  virtual Real3 computeNormal(const FaceGroup& face_group,
                              const VariableNodeReal3& nodes_coord) =0;

  /*!
   * \brief Calcule le vecteur directeur d'une ligne.
   *
   * Cette méthode calcule le vecteur directeur d'un groupe de noeuds
   * en considérant qu'il forme une ligne. Pour le calcul, l'algorithme essaie de
   * déterminer les noeuds aux extrémités de cette ligne, et calcule un
   * vecteur à partir de ces noeuds. Le sens du vecteur est indéfini.
   *
   * Si le groupe ne forme pas une ligne, le résultat est indéfini.
   *
   * Cette méthode est collective. L'algorithme utilisé garantit les
   * mêmes résultats en séquentiel et en parallèle.
   *
   * Si \a n1 et \a n2 ne sont pas nuls, ils contiendront en sortie
   * les coordonnées extrèmes à partir desquelles la direction est calculée.
   *
   * La variable \a nodes_coord est utilisée comme coordonnées pour les noeuds.
   * En général, il s'agit de IMesh::nodesCoordinates().
   */
  virtual Real3 computeDirection(const NodeGroup& node_group,
                                 const VariableNodeReal3& nodes_coord,
                                 Real3* n1,Real3* n2) =0;

  //! Calcul des adjacences, rangées dans \a adjacency_array
  ARCANE_DEPRECATED_REASON("Y2020: Use computeAdjacency() instead")
  virtual void computeAdjency(ItemPairGroup adjacency_array, eItemKind link_kind,
                              Integer nb_layer) =0;

  //! Calcul des adjacences, rangées dans \a adjacency_array
  virtual void computeAdjacency(const ItemPairGroup& adjacency_array, eItemKind link_kind,
                                Integer nb_layer);

  /*!
   * \brief Positionne les nouveaux propriétaires des noeuds, arêtes
   * et faces à partir des mailles.
   *
   * En considérant que les nouveaux propriétaires des mailles sont
   * connus (et synchronisés), détermine les nouveaux propriétaires des autres
   * entités et les synchronise.
   *
   * Cette méthode est collective.
   *
   * \note Cette méthode nécessite que les informations de synchronisations soient
   * valides. Si on souhaite déterminer les propriétaires des entités sans
   * information préalable, il faut utiliser computeAndSetOwnersForNodes()
   * ou computeAndSetOwnersForFaces().
   */
  virtual void changeOwnersFromCells() =0;

  /*!
   * \brief Détermine les propriétaires des noeuds.
   *
   * La détermination se fait en fonction des propriétaires des mailles.
   * Il ne doit pas y avoir de couches de mailles fantômes.
   *
   * Cette opération est collective.
   */
  virtual void computeAndSetOwnersForNodes() =0;

  /*!
   * \brief Détermine les propriétaires des faces.
   *
   * La détermination se fait en fonction des propriétaires des mailles.
   * Il ne doit pas y avoir de couches de mailles fantômes.
   *
   * Cette opération est collective.
   */
  virtual void computeAndSetOwnersForFaces() =0;

  /*!
   * \brief Ecrit le maillage dans un fichier.
   *
   * Ecrit le maillage dans le fichier \a file_name en utilisant
   * le service implémentant l'interface 'IMeshWriter' et de nom \a service_name.
   *
   * \retval true si le service spécifié n'est pas disponible.
   * \retval false si tout est ok.
   */
  virtual bool writeToFile(const String& file_name,const String& service_name) =0;


  /*!
   * \brief Repartitionne et échange le maillage en gérant la réplication.
   *
   * Cette méthode effectue un repartitionnement du maillage via
   * l'appel à IMeshPartitioner::partitionMesh(bool) et procède à l'échange
   * des entités via IPrimaryMesh::exchangeItems().
   *
   * Elle mais gère aussi la réplication en s'assurant que tous les réplica
   * ont le même maillage.
   * Le principe est le suivant:
   * - seul le réplica maître effectue le repartitionnement en
   * appelant IMeshPartitioner::partitionMesh() avec \a partitioner comme partitionneur
   * - les valeurs des IItemFamily::itemsNewOwner() sont ensuite
   * synchronisées avec les autres réplicas.
   * - les échanges d'entités sont effectués via IPrimaryMesh::exchangeItems().
   *
   * Cette méthode est collective sur l'ensemble des réplicas.
   *
   * \pre Tous les réplicas doivent avoir le même maillage, c'est à dire
   * que toutes les familles d'entités doivent être identiques à l'exception
   * des familles de particules qui ne sont pas concernées.
   * \pre Le maillage doit être une instance de IPrimaryMesh.
   *
   * \post Tous les réplicas ont le même maillage à l'exception des familles
   * de particules.
   *
   * \param partitioner Instance du partitionneur à utiliser
   * \param initial_partition Indique s'il s'agit du partitionnement initial.
   */
  virtual void partitionAndExchangeMeshWithReplication(IMeshPartitionerBase* partitioner,
                                                       bool initial_partition) =0;

  /*!
   * \brief Fusionne des noeuds.
   *
   * Fusionne deux à deux les noeuds de \a nodes_to_merge_local_id avec ceux
   * de \a nodes_local_id. Chaque noeud \a nodes_to_merge_local_id[i] est
   * fusionné avec \a nodes_local_id[i].
   *
   * Les noeuds \a nodes_to_merge_local_id sont détruits après fusion. Les entités
   * reposant entièrement sur ces noeuds fusionnés sont aussi détruites.
   *
   * Il est interdit de fusionner deux noeuds d'une même maille ou d'une même face
   * (après fusion, une face ou une maille ne peut pas avoir deux fois le
   * même noeud).
   */
  virtual void mergeNodes(Int32ConstArrayView nodes_local_id,
                          Int32ConstArrayView nodes_to_merge_local_id) =0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  

