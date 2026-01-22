// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* CartesianMeshAMRMng.h                                       (C) 2000-2026 */
/*                                                                           */
/* Gestionnaire de l'AMR pour un maillage cartésien.                         */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#ifndef ARCANE_CARTESIANMESH_CARTESIANMESHAMRMNG_H
#define ARCANE_CARTESIANMESH_CARTESIANMESHAMRMNG_H

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/core/ItemTypes.h"

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
   * \brief Méthode permettant de commencer le raffinement du maillage.
   *
   * \warning Méthode expérimentale.
   *
   * Cette méthode ne peut être appelée que si le maillage est un maillage
   * AMR (IMesh::isAmrActivated()==true) et que le type de l'AMR est 3
   * (PatchCartesianMeshOnly).
   *
   * Cette méthode est la première d'un trio de méthodes nécessaires pour
   * raffiner le maillage :
   * - \a void beginAdaptMesh(Int32 max_nb_levels, Int32 level_to_refine_first)
   * - \a void adaptLevel(Int32 level_to_adapt)
   * - \a void endAdaptMesh()
   *
   * Cette première méthode va permettre de préparer le maillage au
   * raffinement.
   *
   * Il est nécessaire de passer en paramètre de la méthode le nombre de
   * niveaux de raffinements qui va y avoir pendant cette phase de
   * raffinement (\a max_nb_levels).
   *
   * Il est recommandé de mettre le nombre exact de niveaux pour éviter un
   * ajustement du nombre de couches de mailles de recouvrements lors de
   * l'appel à la troisième méthode qui est couteux en calcul.
   *
   *
   * Il est aussi nécessaire de passer en paramètre le premier niveau à être
   * raffiné.
   * Si deux niveaux sont déjà présent sur le maillage (0 et 1) et que vous
   * souhaitez uniquement créer un troisième niveau (niveau 2) à partir du
   * second niveau (niveau 1), vous pouvez mettre 1 au paramètre
   * \a level_to_refine_first.
   *
   *
   * Si deux niveaux sont déjà présent sur le maillage (0 et 1) et que vous
   * souhaitez repartir de zéro, vous pouvez mettre 0 au paramètre
   * \a level_to_refine_first.
   * Dans ce cas, les patchs du niveau 1 seront supprimés, mais pas les
   * mailles/faces/noeuds. Une fois les nouveaux patchs de niveau 1 créés à
   * l'aide de la deuxième méthode, la troisième méthode s'occupera de
   * supprimer les items en trop.
   * Cela permet de conserver les valeurs des variables pour les
   * mailles/faces/noeuds qui étaient dans un patch avant et qui sont
   * conservés dans un nouveau patch.
   *
   *
   * Exemple d'exécution :
   * ```
   * CartesianMeshAMRMng amr_mng(cmesh());
   * amr_mng.clearRefineRelatedFlags();
   *
   * amr_mng.beginAdaptMesh(2, 0);
   * for (Integer level = 0; level < 2; ++level){
   *   // Va faire ses calculs et mettre des flags II_Refine sur les mailles
   *   // du niveau level.
   *   computeInLevel(level);
   *   amr_mng.adaptLevel(level);
   * }
   * amr_mng.endAdaptMesh();
   * ```
   *
   * Cette opération est collective.
   *
   * \param max_nb_levels Le nombre de niveaux de raffinement désiré.
   * \param level_to_refine_first Le niveau qui sera raffiné en premier.
   */
  void beginAdaptMesh(Int32 max_nb_levels, Int32 level_to_refine_first);

  /*!
   * \brief Méthode permettant de créer un niveau de raffinement du maillage.
   *
   * \warning Méthode expérimentale.
   *
   * Cette méthode ne peut être appelée que si le maillage est un maillage
   * AMR (IMesh::isAmrActivated()==true) et que le type de l'AMR est 3
   * (PatchCartesianMeshOnly).
   *
   * Cette méthode est la seconde d'un trio de méthodes nécessaires pour
   * raffiner le maillage :
   * - \a void beginAdaptMesh(Int32 max_nb_levels, Int32 level_to_refine_first)
   * - \a void adaptLevel(Int32 level_to_adapt)
   * - \a void endAdaptMesh()
   *
   * Cette seconde méthode va permettre de raffiner le maillage niveau par
   * niveau.
   *
   * Attention, le paramètre \a level_to_adapt désigne bien le niveau à
   * raffiner, donc la création du niveau \a level_to_adapt +1 (si on veut
   * raffiner le niveau 0, alors il y aura création du niveau 1).
   *
   * Avant d'appeler cette méthode, il faut ajouter le flag "II_Refine" sur les
   * mailles qui doivent être raffinées, sur le niveau \a level_to_adapt uniquement.
   * Pour être sûr de n'avoir aucun flag déjà présent sur le maillage, il est
   * possible d'appeler la méthode \a clearRefineRelatedFlags().
   *
   * Pour le raffinement des mailles hors niveau 0 (ce niveau "ground" ayant
   * un statut particulier), les mailles pouvant être raffinées doivent
   * posséder le flag "II_InPatch". Les mailles n'ayant pas le flag
   * "II_InPatch" ne peuvent pas être raffinés.
   * \todo Ajouter le flag "II_InPatch" à toutes les mailles de niveau 0 ?
   *
   * Les mailles du niveau \a level_to_adapt déjà raffinées, mais n'ayant pas
   * de flag "II_Refine" pourront être supprimées lors de l'appel à la
   * troisième méthode.
   * Cette méthode redessine les patchs et créée les nouvelles mailles enfant
   * si nécessaire, mais ne supprime pas de mailles. La troisième méthode se
   * chargera de supprimer toutes les mailles n'appartenant à aucun patch.
   *
   * Une fois cette méthode appelée, le niveau \a level_to_adapt +1 est prêt à
   * être utilisé, notamment pour marquer les mailles "II_Refine", et rappeler
   * cette méthode pour créer un autre niveau, &c.
   *
   * Cette méthode est faite pour être appelé itérativement, niveau par niveau
   * (du niveau le plus bas au niveau le plus haut). Si des patchs de niveaux
   * supérieurs à \a level_to_adapt sont détectés, ils seront supprimés.
   * Il est donc possible d'appeler cette méthode pour un niveau n, puis de la
   * rappeler pour un niveau n-1 par exemple (attention néanmoins au nombre de
   * nouvelles mailles créées).
   *
   * Cette opération est collective.
   *
   * \param level_to_adapt Le niveau à adapter.
   */
  void adaptLevel(Int32 level_to_adapt) const;

  /*!
   * \brief Méthode permettant de terminer le raffinement du maillage.
   *
   * \warning Méthode expérimentale.
   *
   * Cette méthode ne peut être appelée que si le maillage est un maillage
   * AMR (IMesh::isAmrActivated()==true) et que le type de l'AMR est 3
   * (PatchCartesianMeshOnly).
   *
   * Cette méthode est la troisième d'un trio de méthodes nécessaires pour
   * raffiner le maillage :
   * - \a void beginAdaptMesh(Int32 max_nb_levels, Int32 level_to_refine_first)
   * - \a void adaptLevel(Int32 level_to_adapt)
   * - \a void endAdaptMesh()
   *
   * Cette troisième méthode va permettre de terminer le raffinement du
   * maillage, notamment de supprimer les mailles n'appartenant plus à aucun patch.
   *
   * Si le plus haut niveau raffiné avec la seconde méthode ne correspond pas
   * au paramètre \a max_nb_levels de la première méthode, il y aura
   * ajustement du nombre de couches de mailles de recouvrement.
   *
   * Cette opération est collective.
   */
  void endAdaptMesh();

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
   * \brief Méthode permettant de modifier le nombre de couches de mailles de
   * recouvrement sur le niveau de raffinement le plus haut.
   *
   * Un appel à cette méthode va déclencher l'ajustement du nombre de couches
   * pour tous les patchs déjà présent.
   *
   * Le paramètre \a new_size doit être un nombre pair (sinon, il sera modifié
   * au nombre pair supérieur).
   *
   * \param new_size Le nouveau nombre de couches de mailles de recouvrement.
   */
  void setOverlapLayerSizeTopLevel(Int32 new_size) const;

  /*!
   * \brief Méthode permettant de désactiver les couches de mailles de
   * recouvrement (et de les détruire si présentes).
   *
   * \warning Sans cette couche, il peut y avoir plus d'un niveau de
   * raffinement entre deux mailles. C'est à l'utilisateur de gérer lui-même
   * cette contrainte.
   *
   * \note Pour réactiver ces couches, un appel à
   * \a setOverlapLayerSizeTopLevel() est suffisant.
   */
  void disableOverlapLayer();

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
   * un appel à \a adaptLevel() car \a adaptLevel() s'en occupe.
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
