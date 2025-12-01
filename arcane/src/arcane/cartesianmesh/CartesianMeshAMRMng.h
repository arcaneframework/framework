// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* CartesianMeshAMRMng.h                                  (C) 2000-2025 */
/*                                                                           */
/* Gestionnaire de l'AMR par patch d'un maillage cartésien.                  */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#ifndef ARCANE_CARTESIANMESH_CARTESIANMESHAMRMNG_H
#define ARCANE_CARTESIANMESH_CARTESIANMESHAMRMNG_H

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/cartesianmesh/ICartesianMesh.h"
#include "arcane/utils/TraceAccessor.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class ARCANE_CARTESIANMESH_EXPORT CartesianMeshAMRMng
{
 public:

  explicit CartesianMeshAMRMng(ICartesianMesh* cmesh);

 public:

  /*!
   * \brief Nombre de patchs du maillage.
   *
   * Il y a toujours au moins un patch qui représente le maillage cartésien.
   */
  Int32 nbPatch() const;

  /*!
   * \brief Retourne le \a index-ième patch du maillage.
   *
   * Si le maillage est cartésien, il n'y a qu'un seul patch.
   *
   * L'instance retournée reste valide tant que cette instance n'est pas détruite.
   */
  CartesianPatch amrPatch(Int32 index) const;

  /*!
   * \brief Vue sur la liste des patchs.
   */
  CartesianMeshPatchListView patches() const;

  /*!
   * \brief Raffine un bloc du maillage cartésien.
   *
   * Cette méthode ne peut être appelée que si le maillage est un maillage
   * AMR (IMesh::isAmrActivated()==true).
   *
   * Les mailles dont les positions des centres sont comprises entre
   * \a position et \a (position+length) sont raffinées et les informations
   * de connectivité correspondantes sont mises à jour.
   *
   * Cette opération est collective.
   */
  void refineZone(const AMRZonePosition& position) const;

  /*!
   * \brief Dé-raffine un bloc du maillage cartésien.
   *
   * Cette méthode ne peut être appelée que si le maillage est un maillage
   * AMR (IMesh::isAmrActivated()==true).
   *
   * Les mailles dont les positions des centres sont comprises entre
   * \a position et \a (position+length) sont dé-raffinées et les informations
   * de connectivité correspondantes sont mises à jour.
   *
   * Toutes les mailles dans la zone de dé-raffinement doivent être du même
   * niveau.
   *
   * Les patchs ne contenant plus de mailles après l'appel à cette méthode
   * seront supprimés.
   *
   * Cette opération est collective.
   */
  void coarseZone(const AMRZonePosition& position) const;

  /*!
   * \brief TODO
   */
  void refine() const;

  /*!
   * \brief Méthode permettant de supprimer une ou plusieurs couches
   * de mailles fantômes sur un niveau de raffinement défini.
   *
   * Le nombre de couches de mailles fantômes souhaité peut être augmenté
   * par la méthode. Il est nécessaire de récupérer la valeur retournée
   * pour avoir le nombre de couches de mailles fantômes final.
   *
   * \param level Le niveau de raffinement concerné par la suppression
   * des mailles fantômes.
   *
   * \param target_nb_ghost_layers Le nombre de couches souhaité après
   * appel à cette méthode. ATTENTION : Il peut être ajusté par la méthode.
   *
   * \return Le nombre de couches de mailles fantômes final.
   */
  Integer reduceNbGhostLayers(Integer level, Integer target_nb_ghost_layers) const;

  /*!
   * \brief TODO
   */
  void mergePatches() const;

  /*!
   * \brief TODO
   */
  void createSubLevel() const;

 private:

  ICartesianMesh* m_cmesh;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif //ARCANE_CARTESIANMESH_CARTESIANMESHAMRMNG_H
