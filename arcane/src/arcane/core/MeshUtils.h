// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* MeshUtils.h                                                 (C) 2000-2025 */
/*                                                                           */
/* Fonctions utilitaires diverses sur le maillage.                           */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_MESHUTILS_H
#define ARCANE_CORE_MESHUTILS_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/FunctorUtils.h"
#include "arcane/utils/MemoryUtils.h"

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
} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::MeshUtils
{
extern "C++" ARCANE_CORE_EXPORT void
writeMeshItemInfo(ISubDomain*, Cell cell, bool depend_info = true);
extern "C++" ARCANE_CORE_EXPORT void
writeMeshItemInfo(ISubDomain*, Node node, bool depend_info = true);
extern "C++" ARCANE_CORE_EXPORT void
writeMeshItemInfo(ISubDomain*, Edge edge, bool depend_info = true);
extern "C++" ARCANE_CORE_EXPORT void
writeMeshItemInfo(ISubDomain*, Face face, bool depend_info = true);

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
extern "C++" ARCANE_CORE_EXPORT bool
reorderNodesOfFace(Int64ConstArrayView before_ids, Int64ArrayView after_ids);

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
extern "C++" ARCANE_CORE_EXPORT bool
reorderNodesOfFace2(Int64ConstArrayView nodes_unique_id, IntegerArrayView new_index);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Recherche une entité face à partir des numéros locaux de ces noeuds.
 *
 * Recherche la face donnée par la liste ordonnée des numéros <b>locaux</b> de
 * ces noeuds \a face_nodes_local_id. \a node doit être le premier noeud de
 * la face. Les noeuds de la face doivent être correctement orientés, comme
 * après un appel à reorderNodesOfFace().
 *
 * \return la face correspondante ou la face nulle si elle n'est pas trouvé.
 */
extern "C++" ARCANE_CORE_EXPORT Face
getFaceFromNodesLocalId(Node node, Int32ConstArrayView face_nodes_local_id);

ARCANE_DEPRECATED_REASON("Y2025: Use getFaceFromNodesLocalId() instead")
inline Face
getFaceFromNodesLocal(Node node, Int32ConstArrayView face_nodes_local_id)
{
  return getFaceFromNodesLocalId(node, face_nodes_local_id);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Recherche une entité face à partir des numéros uniques de ces noeuds.
 *
 * Recherche la face donnée par la liste ordonnée des numéros <b>unique</b> de
 * ces noeuds \a face_nodes_unique_id. \a node doit être le premier noeud de
 * la face. Les noeuds de la face doivent être correctement orientés, comme
 * après un appel à reorderNodesOfFace().
 *
 * \return la face correspondante ou la face nulle si elles n'est pas trouvé.
 */
extern "C++" ARCANE_CORE_EXPORT Face
getFaceFromNodesUniqueId(Node node, Int64ConstArrayView face_nodes_unique_id);

ARCANE_DEPRECATED_REASON("Y2025: Use getFaceFromNodesUniqueId() instead")
inline Face
getFaceFromNodesUnique(Node node, Int64ConstArrayView face_nodes_unique_id)
{
  return getFaceFromNodesUniqueId(node, face_nodes_unique_id);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Génère un identifiant unique à partir d'une liste d'identifiants de noeuds.
 */
extern "C++" ARCANE_CORE_EXPORT Int64
generateHashUniqueId(SmallSpan<const Int64> nodes_unique_id);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Supprime une entité en conservant l'ordre.
 *
 * Supprime l'entité de numéro local \a local_id de la liste \a items.
 * Les entités situées après l'entité supprimée sont décalées pour remplir le trou.
 * Si aucune valeur de \a items ne vaut \a local_id, une exception est levée.
 */
extern "C++" ARCANE_CORE_EXPORT void
removeItemAndKeepOrder(Int32ArrayView items, Int32 local_id);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Vérifie que le maillage possède certaines propriétés.
 *
 * Si \a is_sorted, vérifie que les entités de maillage sont triées par
 * ordre croissant de leur uniqueId().
 * Si \a has_no_hole est vrai, vérifie que si le maillage possède \a n
 * entité d'un type, leur numéro local varie bien de \a 0 à \a n-1.
 * Si \a check_faces est vrai, on vérifie pour les faces. Cette option
 * n'est utilisable que pour les maillages ancienne génération (MeshV1) et
 * sera supprimée dès que cette option ne sera plus utilisée.
 */
extern "C++" ARCANE_CORE_EXPORT void
checkMeshProperties(IMesh* mesh, bool is_sorted, bool has_no_hole, bool check_faces);

/*!
 * \brief Ecrit sur le fichier \a file_name les infos du maillage \a mesh
 *
 * Les identifiants des entités sont triés pour que le maillage soit
 * identique quel que soit la numérotation initiale.
 */
extern "C++" ARCANE_CORE_EXPORT void
writeMeshInfosSorted(IMesh* mesh, const String& file_name);

extern "C++" ARCANE_CORE_EXPORT void
writeMeshInfos(IMesh* mesh, const String& file_name);

/*!
 * \brief Ecrit sur le fichier \a file_name la connectivité du maillage \a mesh
 *
 * La connectivité de chaque entité arête, face et maille est sauvée.
 */
extern "C++" ARCANE_CORE_EXPORT void
writeMeshConnectivity(IMesh* mesh, const String& file_name);

extern "C++" ARCANE_CORE_EXPORT void
checkMeshConnectivity(IMesh* mesh, const XmlNode& root_node, bool check_sub_domain);

extern "C++" ARCANE_CORE_EXPORT void
checkMeshConnectivity(IMesh* mesh, const String& file_name, bool check_sub_domain);

/*!
 * \brief Ecrit dans le flux \a ostr la description des items du groupe \a item_group
 *
 * Pour l'affichage, un nom \a name est associé.
 */
extern "C++" ARCANE_CORE_EXPORT void
printItems(std::ostream& ostr, const String& name, ItemGroup item_group);

/*!
 * \brief Affiche l'utilisation mémoire des groupes du maillage.
 *
 * Si \a print_level vaut 0, affiche uniquement l'usage mémoire total.
 * So \a print_level vaut 1 ou plus, affiche l'usage pour chaque groupe.
 *
 * En retour, indique la mémoire consommée en octets.
 */
extern "C++" ARCANE_CORE_EXPORT Int64
printMeshGroupsMemoryUsage(IMesh* mesh, Int32 print_level);

//! Limite au plus juste l'usage mémoire des groupes.
extern "C++" ARCANE_CORE_EXPORT void
shrinkMeshGroups(IMesh* mesh);

/*!
 * \brief Ecrit dans un fichier les informations sur la topologie d'une synchronisation
 *
 * Ecrit dans le fichier \a filename les informations sur la topologie de \a var_syncer.
 * Cette méthode est collective. Seul le rang 0 écrit l'information de la topologie.
 */
extern "C++" ARCANE_CORE_EXPORT void
dumpSynchronizerTopologyJSON(IVariableSynchronizer* var_syncer, const String& filename);

/*!
 * \internal
 * \brief Calcul et affiche les patterns communs dans les connectivités.
 */
extern "C++" ARCANE_CORE_EXPORT void
computeConnectivityPatternOccurence(IMesh* mesh);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
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
 *
 * Si \a q est non nul et que \a do_prefetch vaut \a true, alors
 * VariableUtils::prefetchVariableAsync() est appelé pour chaque variable
 * gérant la connectivité.
 */
extern "C++" ARCANE_CORE_EXPORT void
markMeshConnectivitiesAsMostlyReadOnly(IMesh* mesh, RunQueue* q = nullptr,
                                       bool do_prefetch = false);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*
 * \brief Retourne l'entité de la famille \a family de numéro unique \a unique_id.
 *
 * Si aucune entité avec cet \a unique_id n'est trouvé, retourne l'entité nulle.
 *
 * \pre family->hasUniqueIdMap() == true
 */
extern "C++" ARCANE_CORE_EXPORT ItemBase
findOneItem(IItemFamily* family, Int64 unique_id);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*
 * \brief Retourne l'entité de la famille \a family de numéro unique \a unique_id.
 *
 * Si aucune entité avec cet \a unique_id n'est trouvé, retourne l'entité nulle.
 *
 * \pre family->hasUniqueIdMap() == true
 */
extern "C++" ARCANE_CORE_EXPORT ItemBase
findOneItem(IItemFamily* family, ItemUniqueId unique_id);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Visite l'ensemble des groupes de \a family avec le functor \a functor.
 */
extern "C++" ARCANE_CORE_EXPORT void
visitGroups(IItemFamily* family, IFunctorWithArgumentT<ItemGroup&>* functor);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Visite l'ensemble des groupes de \a mesh avec le functor \a functor.
 */
extern "C++" ARCANE_CORE_EXPORT void
visitGroups(IMesh* mesh, IFunctorWithArgumentT<ItemGroup&>* functor);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Visite l'ensemble des groupes de \a family avec la lambda \a f.
 *
 * Cette fonction permet d'appliquer un visiteur pour l'ensemble des
 * groupes de la famille \a family.
 *
 * Par exemple:
 *
 * \code
 * IMesh* mesh = ...;
 * auto xx = [](const ItemGroup& x) { std::cout << "name=" << x.name(); };
 * MeshUtils::visitGroups(mesh,xx);
 * \endcode
 */
template <typename LambdaType> inline void
visitGroups(IItemFamily* family, const LambdaType& f)
{
  StdFunctorWithArgumentT<ItemGroup&> sf(f);
  // Il faut caster en le bon type pour que le compilateur utilise la bonne surcharge.
  IFunctorWithArgumentT<ItemGroup&>* sf_addr = &sf;
  visitGroups(family, sf_addr);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Visite l'ensemble des groupes de \a mesh avec la lambda \a f.
 *
 * Cette fonction permet d'appliquer un visiteur pour l'ensemble des
 * groupes de l'ensemble des familles du maillage \a mesh
 *
 * Elle s'utilise comme suit:
 *
 * \code
 * IMesh* mesh = ...;
 * auto xx = [](const ItemGroup& x) { std::cout << "name=" << x.name(); };
 * MeshVisitor::visitGroups(mesh,xx);
 * \endcode
 */
template <typename LambdaType> inline void
visitGroups(IMesh* mesh, const LambdaType& f)
{
  StdFunctorWithArgumentT<ItemGroup&> sf(f);
  // Il faut caster en le bon pour que le compilateur utilise la bonne surcharge.
  IFunctorWithArgumentT<ItemGroup&>* sf_addr = &sf;
  visitGroups(mesh, sf_addr);
}

namespace impl
{
  inline Int64 computeCapacity(Int64 size)
  {
    return Arcane::MemoryUtils::impl::computeCapacity(size);
  }
} // namespace impl

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Redimensionne un tableau qui est indexé par des 'ItemLocalId'.
 *
 * Le tableau \a array est redimensionné uniquement si \a new_size est
 * supérieure à la taille actuelle du tableau ou si \a force_resize est vrai.
 *
 * Si le tableau est redimensionné, on réserve une capacité supplémentaire
 * pour éviter de réallouer à chaque fois.
 *
 * Cette fonction est appelée en général pour les tableaux indexés par un
 * ItemLocalId et donc cette fonction peut être appelée à chaque fois qu'on
 * ajoute une entité au maillage.
 *
 * \retval true si un redimensionnement a eu lieu
 * \retval false sinon
 */
template <typename DataType> inline bool
checkResizeArray(Array<DataType>& array, Int64 new_size, bool force_resize)
{
  return Arcane::MemoryUtils::checkResizeArrayWithCapacity(array, new_size, force_resize);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Retourne le maximum des uniqueId() des entités standards du maillage.
 *
 * Les entités standards sont les noeuds, mailles, faces et arêtes.
 * L'opération est collective sur mesh->parallelMng().
 */
extern "C++" ARCANE_CORE_EXPORT ItemUniqueId
getMaxItemUniqueIdCollective(IMesh* mesh);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Vérifie le hash des uniqueId() des entités d'une famille.
 *
 * Calcule via l'algo \a hash_algo un hash des uniqueId() des entités
 * d'une famille. Pour ce calcul, le rang 0 récupère l'ensemble des uniqueId()
 * des entités propres de chaque sous-domaine, les trie et calcul le hash
 * sur le tableau trié.
 *
 * Comme la majorité du travail est effectuée par le rang 0, cette méthode
 * n'est pas très extensible et ne doit donc être utilisée qu'à des fins
 * de test.
 *
 * \a expected_hash est la valeur attendue du hash sous forme de caractères
 * hexadécimaux (obtenu via Convert::toHexaString()). Si \a expected_hash
 * est non nul, compare le résultat avec cette valeur et si elle est différente,
 * lance une exception FatalErrorException.
 *
 * Cette opération est collective.
 */
extern "C++" ARCANE_CORE_EXPORT void
checkUniqueIdsHashCollective(IItemFamily* family, IHashAlgorithm* hash_algo,
                             const String& expected_hash, bool print_hash_value,
                             bool include_ghost);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Rempli \a uids avec les uniqueId() des entités de \a view.
 */
extern "C++" ARCANE_CORE_EXPORT void
fillUniqueIds(ItemVectorView items,Array<Int64>& uids);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Créé ou recréé une connectivité noeuds-noeuds via les arêtes.
 *
 * La connectivité aura pour nom \a connectivity_name.
 */
extern "C++" ARCANE_CORE_EXPORT Ref<IIndexedIncrementalItemConnectivity>
computeNodeNodeViaEdgeConnectivity(IMesh* mesh, const String& connectivity_name);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Créé ou recréé une connectivité noeuds-noeuds via les arêtes
 * pour les noeuds sur les faces frontières du maillage.
 *
 * La connectivité aura pour nom \a connectivity_name.
 */
extern "C++" ARCANE_CORE_EXPORT Ref<IIndexedIncrementalItemConnectivity>
computeBoundaryNodeNodeViaEdgeConnectivity(IMesh* mesh, const String& connectivity_name);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::MeshUtils

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::mesh_utils
{
// Using pour compatibilité avec l'existant.
// Ces using ont été ajoutés pour la version 3.10 de Arcane (juin 2023).
// On pourra les rendre obsolètes début 2024.
using MeshUtils::checkMeshConnectivity;
using MeshUtils::checkMeshProperties;
using MeshUtils::computeConnectivityPatternOccurence;
using MeshUtils::dumpSynchronizerTopologyJSON;
using MeshUtils::getFaceFromNodesLocal;
using MeshUtils::getFaceFromNodesUnique;
using MeshUtils::printItems;
using MeshUtils::printMeshGroupsMemoryUsage;
using MeshUtils::removeItemAndKeepOrder;
using MeshUtils::reorderNodesOfFace;
using MeshUtils::reorderNodesOfFace2;
using MeshUtils::shrinkMeshGroups;
using MeshUtils::writeMeshConnectivity;
using MeshUtils::writeMeshInfos;
using MeshUtils::writeMeshInfosSorted;
using MeshUtils::writeMeshItemInfo;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::mesh_utils

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
