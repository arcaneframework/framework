// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* CartesianMeshAMRMng.h                                       (C) 2000-2025 */
/*                                                                           */
/* Gestionnaire de l'AMR pour un maillage cartésien.                         */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#ifndef ARCANE_CARTESIANMESH_CARTESIANMESHAMRMNG_H
#define ARCANE_CARTESIANMESH_CARTESIANMESHAMRMNG_H

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/cartesianmesh/CartesianMeshGlobal.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Classe permettant d'accéder aux méthodes spécifiques AMR du maillage
 * cartesien.
 *
 * Une instance de cette classe est valide tant que le ICartesianMesh passé en
 * paramètre du constructeur est valide.
 */
class ARCANE_CARTESIANMESH_EXPORT CartesianMeshAMRMng
{
 public:

  /*!
   * \brief Constructeur.
   */
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
   * \brief Méthode permettant d'adapter le raffinement du maillage selon les
   * mailles à raffiner.
   *
   * Cette méthode ne peut être appelée que si le maillage est un maillage
   * AMR (IMesh::isAmrActivated()==true) et que le type de l'AMR est 3
   * (PatchCartesianMeshOnly).
   *
   * Avant d'appeler cette méthode, il faut ajouter le flag "II_Refine" sur les
   * mailles qui doivent être raffinées. Il est possible de le faire niveau par
   * niveau ou plusieurs niveaux d'un coup (si plusieurs niveaux existent
   * déjà).
   * Pour être sûr de n'avoir aucun flag déjà présent sur le maillage, il est
   * possible d'appeler la méthode \a clearRefineRelatedFlags().
   * Dans le cas d'un raffinement niveau par niveau, il est possible de mettre
   * le paramètre \a clear_refine_flag à false afin de garder les flags des
   * niveaux inférieurs et d'éviter d'avoir à les recalculer. Pour le dernier
   * niveau, il est recommandé de mettre le paramètre \a clear_refine_flag à
   * true pour supprimer les flags devenu inutiles (ou d'appeler la méthode
   * clearRefineRelatedFlags()).
   *
   * Les mailles n'ayant pas de flag "II_Refine" seront déraffinées.
   *
   * Afin d'éviter les mailles orphelines, si une maille est marquée
   * "II_Refine", alors la maille parente est marquée "II_Refine".
   *
   * Exemple d'exécution :
   * ```
   * CartesianMeshAMRMng amr_mng(cmesh());
   * amr_mng.clearRefineRelatedFlags();
   * for (Integer level = 0; level < 2; ++level){
   *   computeInLevel(level); // Va mettre des flags II_Refine sur les mailles
   *   amr_mng.adaptMesh(false);
   * }
   * amr_mng.clearRefineRelatedFlags();
   * ```
   *
   * Cette opération est collective.
   *
   * \param clear_refine_flag true si l'on souhaite supprimer les flags
   * II_Refine après adaptation.
   */
  void adaptMesh(bool clear_refine_flag) const;

  /*!
   * \brief Méthode permettant de supprimer les flags liés au raffinement de
   * toutes les mailles.
   *
   * Les flags concernés sont :
   * - ItemFlags::II_Coarsen
   * - ItemFlags::II_Refine
   * - ItemFlags::II_JustCoarsened
   * - ItemFlags::II_JustRefined
   * - ItemFlags::II_JustAdded
   * - ItemFlags::II_CoarsenInactive
   */
  void clearRefineRelatedFlags() const;

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
   * \brief Méthode permettant de fusionner les patchs qui peuvent l'être.
   *
   * Cette méthode ne peut être appelée que si le maillage est un maillage
   * AMR (IMesh::isAmrActivated()==true).
   * Si le type de l'AMR n'est pas 3 (PatchCartesianMeshOnly), la méthode ne
   * fait rien.
   *
   * Cette méthode peut être utile après plusieurs appels à \a refineZone() et à
   * \a coarseZone(). En revanche, un appel à cette méthode est inutile après
   * un appel à \a adaptMesh() car \a adaptMesh() s'en occupe.
   */
  void mergePatches() const;

  /*!
   * \brief Méthode permettant de créer un sous-niveau ("niveau -1").
   *
   * Cette méthode ne peut être appelée que si le maillage est un maillage
   * AMR (IMesh::isAmrActivated()==true).
   *
   * Dans le cas d'utilisation de l'AMR type 3 (PatchCartesianMeshOnly), il est
   * possible d'appeler cette méthode en cours de calcul et autant de fois que
   * nécessaire (tant qu'il est possible de diviser la taille du niveau 0 par
   * 2).
   * Une fois le niveau -1 créé, tous les niveaux sont "remontés" (donc le
   * niveau -1 devient le niveau 0 "ground").
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
