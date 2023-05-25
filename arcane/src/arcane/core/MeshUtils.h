// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* MeshUtils.h                                                 (C) 2000-2023 */
/*                                                                           */
/* Fonctions diverses sur les éléments du maillage.                          */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_MESHUTILS_H
#define ARCANE_MESHUTILS_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/Item.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \file MeshUtils.h
 *
 * \brief Fonctions utilitaires sur le maillage.
 */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{
class XmlNode;
class IVariableSynchronizer;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::mesh_utils
{
  ARCANE_CORE_EXPORT void
  writeMeshItemInfo(ISubDomain*,Cell cell,bool depend_info=true);
  ARCANE_CORE_EXPORT void
  writeMeshItemInfo(ISubDomain*,Node node,bool depend_info=true);
  ARCANE_CORE_EXPORT void
  writeMeshItemInfo(ISubDomain*,Edge edge,bool depend_info=true);
  ARCANE_CORE_EXPORT void
  writeMeshItemInfo(ISubDomain*,Face face,bool depend_info=true);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Réordonne les noeuds d'une face.

 Cette méthode réordonne la liste des noeuds d'une face pour que les
 propritétés suivantes soient respectées:
 - le premier noeud de la face est celui dont le numéro global est le plus petit.
 - le deuxième noeud de la face est celui dont le numéro global est le deuxième plus petit.

 Cela permet:
 - d'orienter les faces de manière identiques en parallèle.
 - d'accélerer les recherches sur les recherches entre faces.

 \a before_ids et \a to doivent avoir le même nombre d'éléments

 \param before_ids numéros globaux des noeuds de la face avant renumérotation.
 \param after_ids en sortie, numéros globaux des noeuds de la face après renumérotation

 \retval true si la face change d'orientation lors de la renumérotation
 \retval false sinon.
*/
ARCANE_CORE_EXPORT bool
reorderNodesOfFace(Int64ConstArrayView before_ids,Int64ArrayView after_ids);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Réordonne les noeuds d'une face.

 Cette méthode réordonne la liste des noeuds d'une face pour que les
 propriétés suivantes soient respectées:
 - le premier noeud de la face est celui dont le numéro global est le plus petit.
 - le deuxième noeud de la face est celui dont le numéro global est le deuxième plus petit.

 Cela permet:
 - d'orienter les faces de manière identiques en parallèle.
 - d'accélerer les recherches sur les recherches entre faces.

 \a nodes_unique_id et \a new_index doivent avoir le même nombre d'éléments

 \param nodes_unique_id numéros uniques des noeuds de la face.
 \param new_index en sortie, position des numéros des noeuds après réorientation.

 Par exemple, si une face possède les 4 noeuds de numéros uniques 7 3 2 5,
 la réorientation donnera le quadruplet (2 3 7 5), soit le tableau d'index
 suivant (2,1,0,3).

 \retval true si la face change d'orientation lors de la renumérotation
 \retval false sinon.
*/
ARCANE_CORE_EXPORT bool
reorderNodesOfFace2(Int64ConstArrayView nodes_unique_id,IntegerArrayView new_index);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
  \brief Recherche une entité face à partir des numéros locaux de ces noeuds.

  Recherche la face donnée par la liste ordonnée des numéros <b>locaux</b> de
  ces noeuds \a face_nodes_local_id. \a node doit être le premier noeud de
  la face. Les noeuds de la face doivent être correctement orientés, comme
  après un appel à reorderNodesOfFace().

  \return la face correspondante ou 0 si la face n'est pas trouvé.
*/
ARCANE_CORE_EXPORT Face
getFaceFromNodesLocal(Node node,Int32ConstArrayView face_nodes_local_id);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
  \brief Recherche une entité face à partir des numéros uniques de ces noeuds.

  Recherche la face donnée par la liste ordonnée des numéros <b>unique</b> de
  ces noeuds \a face_nodes_unique_id. \a node doit être le premier noeud de
  la face. Les noeuds de la face doivent être correctement orientés, comme
  après un appel à reorderNodesOfFace().

  \return la face correspondante ou 0 si la face n'est pas trouvé.
*/
ARCANE_CORE_EXPORT Face
getFaceFromNodesUnique(Node node,Int64ConstArrayView face_nodes_unique_id);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Supprime une entité en conservant l'ordre.
 *
 * Supprime l'entité de numéro local \a local_id de la liste \a items.
 * Les entités situées après l'entité supprimée sont décalées pour remplir le trou.
 * Si aucune valeur de \a items ne vaut \a local_id, une exception est levée.
 */
ARCANE_CORE_EXPORT void
removeItemAndKeepOrder(Int32ArrayView items,Int32 local_id);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*! \brief Vérifie que le maillage possède certaines propriétés.
 *
 * Si \a is_sorted, vérifie que les entités de maillage sont triées par
 * ordre croissant de leur uniqueId().
 * Si \a has_no_hole est vrai, vérifie que si le maillage possède \a n
 * entité d'un type, leur numéro local varie bien de \a 0 à \a n-1.
 * Si \a check_faces est vrai, on vérifie pour les faces. Cette option
 * n'est utilisable que pour les maillages ancienne génération (MeshV1) et
 * sera supprimée dès que cette option ne sera plus utilisée.
 */
ARCANE_CORE_EXPORT void
checkMeshProperties(IMesh* mesh,bool is_sorted,bool has_no_hole,bool check_faces);

/*!
 * \brief Ecrit sur le fichier \a file_name les infos du maillage \a mesh
 *
 * Les identifiants des entités sont triés pour que le maillage soit
 * identique quel que soit la numérotation initiale.
 */
ARCANE_CORE_EXPORT void
writeMeshInfosSorted(IMesh* mesh,const String& file_name);

ARCANE_CORE_EXPORT void
writeMeshInfos(IMesh* mesh,const String& file_name);

/*!
 * \brief Ecrit sur le fichier \a file_name la connectivité du maillage \a mesh
 *
 * La connectivité de chaque entité arête, face et maille est sauvée.
 */
ARCANE_CORE_EXPORT void
writeMeshConnectivity(IMesh* mesh,const String& file_name);

ARCANE_CORE_EXPORT void
checkMeshConnectivity(IMesh* mesh,const XmlNode& root_node,bool check_sub_domain);

ARCANE_CORE_EXPORT void
checkMeshConnectivity(IMesh* mesh,const String& file_name,bool check_sub_domain);


/*!
 * \brief Ecrit dans le flux \a ostr la description des items du groupe \a item_group
 *
 * Pour l'affichage, un nom \a name est associé.
 */
ARCANE_CORE_EXPORT void
printItems(std::ostream& ostr,const String& name,ItemGroup item_group);

/*!
 * \brief Affiche l'utilisation mémoire des groupes du maillage.
 *
 * Si \a print_level vaut 0, affiche uniquement l'usage mémoire total.
 * So \a print_level vaut 1 ou plus, affiche l'usage pour chaque groupe.
 *
 * En retour, indique la mémoire consommée en octets.
 */
ARCANE_CORE_EXPORT Int64
printMeshGroupsMemoryUsage(IMesh* mesh,Int32 print_level);

//! Limite au plus juste l'usage mémoire des groupes.
ARCANE_CORE_EXPORT void
shrinkMeshGroups(IMesh* mesh);

/*!
 * \brief Ecrit dans un fichier les informations sur la topologie d'une synchronisation
 *
 * Ecrit dans le fichier \a filename les informations sur la topologie de \a var_syncer.
 * Cette méthode est collective. Seul le rang 0 écrit l'information de la topologie.
 */
ARCANE_CORE_EXPORT void
dumpSynchronizerTopologyJSON(IVariableSynchronizer* var_syncer,const String& filename);

/*!
 * \interne
 * \brief Calcul et affiche les patterns communs dans les connectivités.
 */
ARCANE_CORE_EXPORT void
computeConnectivityPatternOccurence(IMesh* mesh);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane::mesh_utils

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::MeshUtils
{
// TODO: Mettre dans ce namespace les méthodes qui sont dans 'mesh_utils'.

/*!
 * \brief Indique que les connectivités du maillages ne seront
 * pas régulièrement modifiées.
 *
 * Cette fonction permet d'indiquer que les connectivitées associées aux
 * entités du maillage (Node, Edge, Face et Cell) sont la plupart du temps
 * en lecture. A noter que cela ne concerne pas les particules.
 *
 * En cas d'utilisation sur accélérateur, cela permet de dupliquer les
 * informations entre l'accélérateur et l'hôte pour éviter des aller-retour
 * multiples si les connectivités sont utilisées sur les deux à la fois.
 */
extern "C++" ARCANE_CORE_EXPORT
void markMeshConnectivitiesAsMostlyReadOnly(IMesh* mesh);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif

