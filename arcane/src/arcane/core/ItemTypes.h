// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2023 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ItemTypes.h                                                 (C) 2000-2023 */
/*                                                                           */
/* Déclaration des types liés aux entités de maillage.                       */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_ITEMTYPES_H
#define ARCANE_ITEMTYPES_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/ArcaneTypes.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \file ItemTypes.h
 *
 * \brief Déclarations de types sur les entités.
 */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

// A définir si on souhaiter ajouter un offset aux classes 'ItemVector' et
// 'ItemEnumerator'. Cela change la taille de ces structures et il ne faut
// donc recompiler code code utilisateur si on change ce '#define'.
#define ARCANE_HAS_OFFSET_FOR_ITEMVECTORVIEW

// A définir si on souhaite cacher les méthodes d'accès aux structures
// internes des connectivités. Pour l'instant (mi-2023) on ne le fait que
// pour les sources internes à Arcane mais ensuite il faudra le généraliser.
// (La macro ARCANE_FORCE_... est définie dans le CMakeLists.txt principal)
#ifdef ARCANE_FORCE_HIDE_ITEM_CONNECTIVITY_STRUCTURE
#define ARCANE_HIDE_ITEM_CONNECTIVITY_STRUCTURE
#endif

// A définir si on souhaite utiliser les classes spécifiques pour gérer
// les entités connectées (sinon on utilise ItemVectorView)
#define ARCANE_USE_SPECIFIC_ITEMCONNECTED

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class Node;
class Cell;
class Edge;
class Face;
class Particle;
class DoF;

class Item;
class ItemWithNodes;
class ItemInternal;
namespace impl
{
class ItemBase;
class MutableItemBase;
template<int Extent = DynExtent> class ItemIndexedListView;
class ItemLocalIdListContainerView;
}


class IMesh;
class IPrimaryMesh;
class IItemFamily;
class IParticleFamily;
class IDoFFamily;

class ItemLocalId;
class ItemTypeId;
template<typename T> class ItemLocalIdT;
using NodeLocalId = ItemLocalIdT<Node>;
using EdgeLocalId = ItemLocalIdT<Edge>;
using FaceLocalId = ItemLocalIdT<Face>;
using CellLocalId = ItemLocalIdT<Cell>;
using ParticleLocalId = ItemLocalIdT<Particle>;
using DoFLocalId = ItemLocalIdT<DoF>;
class IndexedItemConnectivityViewBase;
class IndexedItemConnectivityViewBase2;
class IndexedItemConnectivityAccessor;
class ItemInternalConnectivityList;
class ItemInternalVectorView;
class ItemLocalIdListView;
template<typename T> class ItemLocalIdViewT;

class ItemGroup;
class ItemGroupImpl;
template<typename T> class ItemGroupT;

class ItemPairGroup;
template<typename ItemKind,typename SubItemKind> class ItemPairGroupT;

class ItemVector;
template<typename T> class ItemVectorT;

class ItemVectorViewConstIterator;
template<typename ItemType>
class ItemVectorViewConstIteratorT;

class ItemConnectedListViewConstIterator;
template<typename ItemType>
class ItemConnectedListViewConstIteratorT;

template <typename ItemType>
class ItemLocalIdViewT;

// (Avril 2022) Fait un typedef de 'ItemLocalIdViewT' vers 'ItemLocalIdView'
// pour compatibilité avec l'existant. A supprimer dès que possible.
template <typename ItemType>
using ItemLocalIdView ARCANE_DEPRECATED_REASON("Use 'ItemLocalIdViewT' instead") = ItemLocalIdViewT<ItemType>;

template<typename ItemType1,typename ItemType2>
class IndexedItemConnectivityViewT;

// (Avril 2022) Fait un typedef de 'IndexedItemConnectivityView' vers 'IndexedItemConnectivityViewT'
// pour compatibilité avec l'existant. A supprimer dès que possible.
template<typename ItemType1,typename ItemType2>
using IndexedItemConnectivityView ARCANE_DEPRECATED_REASON("Use 'IndexedItemConnectivityViewT' instead") = IndexedItemConnectivityViewT<ItemType1,ItemType2>;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

//! Groupe de noeuds connectés à des noeuds
typedef ItemPairGroupT<Node,Node> NodeNodeGroup;
//! Groupe d'arêtes connectées à des noeuds
typedef ItemPairGroupT<Edge,Node> EdgeNodeGroup;
//! Groupe de faces connectées à des noeuds
typedef ItemPairGroupT<Face,Node> FaceNodeGroup;
//! Groupe de mailles connectées à des noeuds
typedef ItemPairGroupT<Cell,Node> CellNodeGroup;

//! Groupe de noeuds connectés à des faces
typedef ItemPairGroupT<Node,Face> NodeFaceGroup;
//! Groupe d'arêtes connectées à des faces
typedef ItemPairGroupT<Edge,Face> EdgeFaceGroup;
//! Groupe de faces connectées à des faces
typedef ItemPairGroupT<Face,Face> FaceFaceGroup;
//! Groupe de mailles connectées à des faces
typedef ItemPairGroupT<Cell,Face> CellFaceGroup;

//! Groupe de noeuds connectés à des mailless
typedef ItemPairGroupT<Node,Cell> NodeCellGroup;
//! Groupe d'arêtes connectées à des mailles
typedef ItemPairGroupT<Edge,Cell> EdgeCellGroup;
//! Groupe de faces connectées à des mailles
typedef ItemPairGroupT<Face,Cell> FaceCellGroup;
//! Groupe de mailles connectées à des mailles
typedef ItemPairGroupT<Cell,Cell> CellCellGroup;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \ingroup Mesh
 * \brief Groupe de noeuds.
 */
typedef ItemGroupT<Node> NodeGroup;
/*!
 * \ingroup Mesh
 * \brief Groupe d'arêtes
 * \note Cette classe n'est pas implémentée.
 */
typedef ItemGroupT<Edge> EdgeGroup;
/*!
 * \ingroup Mesh
 * \brief Groupe de faces.
 */
typedef ItemGroupT<Face> FaceGroup;
/*!
 * \ingroup Mesh
 * \brief Groupe de mailles.
 */
typedef ItemGroupT<Cell> CellGroup;
/*!
 * \ingroup Mesh
 * \brief Groupe de particules.
 */
typedef ItemGroupT<Particle> ParticleGroup;
/*!
 * \ingroup Mesh
 * \brief Groupe de Degre de Liberte.
 */
typedef ItemGroupT<DoF> DoFGroup;
/*!
 * \ingroup Mesh
 * \internal
 * \brief Enumerateur sur la partie interne d'une entité.
 */
class ItemInternalEnumerator;
class ItemEnumerator;
template<typename ItemType>
class ItemEnumeratorT;

class ItemConnectedEnumerator;
template<typename ItemType>
class ItemConnectedEnumeratorT;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class EnumeratorTraceInfo;
class SimdItemEnumeratorBase;

template<typename ItemType>
class ItemEnumeratorBaseT;

template<typename ItemType>
class ItemConnectedEnumeratorBaseT;

template<typename ItemType>
class SimdItemEnumeratorT;
template<typename ItemType>
class SimdItemT;

/*!
 * \ingroup Mesh
 * \brief Vue sur un vecteur d'entité.
 */
class ItemVectorView;
template<typename ItemType>
class ItemVectorViewT;

/*!
 * \ingroup Mesh
 * \brief Vue sur une liste de connectivité
 */
template<int Extent = DynExtent> class ItemConnectedListView;
template<typename ItemType, int Extent = DynExtent>
class ItemConnectedListViewT;

/*!
 * \ingroup Mesh
 * \brief Enumérateur sur une paire d'entité.
 */
class ItemPairEnumerator;
template<typename ItemType,typename SubItemType>
class ItemPairEnumeratorT;

/*!
 * \ingroup Mesh
 * \brief Enumérateurs sur des noeuds.
 */
typedef ItemEnumeratorT<Node> NodeEnumerator;
/*!
 * \ingroup Mesh
 * \brief Enumérateurs sur des arêtes
 */
typedef ItemEnumeratorT<Edge> EdgeEnumerator;

/*!
 * \ingroup Mesh
 * \brief Enumérateurs sur des faces.
 */
typedef ItemEnumeratorT<Face> FaceEnumerator;

/*!
 * \ingroup Mesh
 * \brief Enumérateurs sur des mailles.
 */
typedef ItemEnumeratorT<Cell> CellEnumerator;

/*!
 * \ingroup Mesh
 * \brief Enumérateurs sur des particules.
 */
typedef ItemEnumeratorT<Particle> ParticleEnumerator;

/*!
 * \ingroup Mesh
 * \brief Enumérateurs sur des DoFs.
 */
typedef ItemEnumeratorT<DoF> DoFEnumerator;

/*!
 * \ingroup Mesh
 * \brief Vue sur un vecteur de noeuds.
 */
typedef ItemVectorViewT<Node> NodeVectorView;
/*!
 * \ingroup Mesh
 * \brief Vue sur un vecteur d'arêtes.
 */
typedef ItemVectorViewT<Edge> EdgeVectorView;
/*!
 * \ingroup Mesh
 * \brief Vue sur un vecteur de faces.
 */
typedef ItemVectorViewT<Face> FaceVectorView;
/*!
 * \ingroup Mesh
 * \brief Vue sur un vecteur de mailles.
 */
typedef ItemVectorViewT<Cell> CellVectorView;
/*!
 * \ingroup Mesh
 * \brief Vue sur un vecteur de particules.
 */
typedef ItemVectorViewT<Particle> ParticleVectorView;

/*!
 * \ingroup Mesh
 * \brief Vue sur un vecteur de degre de liberte.
 */
typedef ItemVectorViewT<DoF> DoFVectorView;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \ingroup Mesh
 * \brief Vue sur une liste de noeuds connectés à une entité
 */
using NodeConnectedListView = ItemConnectedListViewT<Node>;
/*!
 * \ingroup Mesh
 * \brief Vue sur une liste d'arêtes connectées à une entité
 */
using EdgeConnectedListView = ItemConnectedListViewT<Edge>;
/*!
 * \ingroup Mesh
 * \brief Vue sur une liste de faces connectées à une entité
 */
using FaceConnectedListView = ItemConnectedListViewT<Face>;
/*!
 * \ingroup Mesh
 * \brief Vue sur une liste de mailles connectées à une entité
 */
using CellConnectedListView = ItemConnectedListViewT<Cell>;
/*!
 * \ingroup Mesh
 * \brief Vue sur une liste de DoFs connectés à une entité
 */
using DoFConnectedListView = ItemConnectedListViewT<DoF>;

#ifdef ARCANE_USE_SPECIFIC_ITEMCONNECTED
//! Liste d'entités connectées
using ItemConnectedListViewType = ItemConnectedListView<DynExtent>;
//! Liste de noeuds connectés
using NodeConnectedListViewType = NodeConnectedListView;
//! Liste d'arêtes connectées
using EdgeConnectedListViewType = EdgeConnectedListView;
//! Liste de faces connectées
using FaceConnectedListViewType = FaceConnectedListView;
//! Liste de mailles connectées
using CellConnectedListViewType = CellConnectedListView;
//! Liste générique d'entités connectées
template<typename ItemType> using ItemConnectedListViewTypeT = ItemConnectedListViewT<ItemType>;
#else
//! Liste d'entités connectées
using ItemConnectedListViewType = ItemVectorView;
//! Liste de noeuds connectés
using NodeConnectedListViewType = NodeVectorView;
//! Liste d'arêtes connectées
using EdgeConnectedListViewType = EdgeVectorView;
//! Liste de faces connectées
using FaceConnectedListViewType = FaceVectorView;
//! Liste de mailles connectées
using CellConnectedListViewType = CellVectorView;
//! Liste générique d'entités connectées
template<typename ItemType> using ItemConnectedListViewTypeT = ItemVectorViewT<ItemType>;
#endif

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*! \brief Collection de groupes de noeuds. */
typedef Collection<NodeGroup> NodeGroupCollection;
/*! \brief Collection de groupes d'arêtes. */
typedef Collection<EdgeGroup> EdgeGroupCollection;
/*! \brief Collection de groupes de faces. */
typedef Collection<FaceGroup> FaceGroupCollection;
/*! \brief Collection de groupes de mailles. */
typedef Collection<CellGroup> CellGroupCollection;
/*! \brief Collection de groupes de particules. */
typedef Collection<ParticleGroup> ParticleGroupCollection;
/*! \brief Collection de groupes de degre de liberte. */
typedef Collection<DoFGroup> DoFGroupCollection;


/*! \brief Tableau de groupes de noeuds. */
typedef List<NodeGroup> NodeGroupList;
/*! \brief Tableau de groupes d'arêtes. */
typedef List<EdgeGroup> EdgeGroupList;
/*! \brief Tableau de groupes de faces. */
typedef List<FaceGroup> FaceGroupList;
/*! \brief Tableau de groupes de mailles. */
typedef List<CellGroup> CellGroupList;
/*! \brief Tableau de groupes de particules. */
typedef List<ParticleGroup> ParticleGroupList;
/*! \brief Tableau de groupes de degre de liberte. */
typedef List<DoFGroup> DoFGroupList;

/*!
 * \ingroup Mesh
 * \brief Vue sur les localId() d'une liste de noeuds.
 */
typedef ItemLocalIdViewT<Node> NodeLocalIdView;
/*!
 * \ingroup Mesh
 * \brief Vue sur les localId() d'une liste d'arêtes.
 */
typedef ItemLocalIdViewT<Edge> EdgeLocalIdView;
/*!
 * \ingroup Mesh
 * \brief Vue sur les localId() d'une liste de faces.
 */
typedef ItemLocalIdViewT<Face> FaceLocalIdView;
/*!
 * \ingroup Mesh
 * \brief Vue sur les localId() d'une liste de mailles.
 */
typedef ItemLocalIdViewT<Cell> CellLocalIdView;
/*!
 * \ingroup Mesh
 * \brief Vue sur les localId() d'une liste de particules.
 */
typedef ItemLocalIdViewT<Particle> ParticleLocalIdView;

/*!
 * \ingroup Mesh
 * \brief Vue sur les localId() d'une liste de DoF.
 */
typedef ItemLocalIdViewT<DoF> DoFLocalIdView;

/*! \brief Type de la liste interne des entités
  
  \deprecated Utiliser ItemInternalArrayView.
  
*/
typedef ConstArrayView<ItemInternal*> ItemInternalList;

typedef ConstArrayView<ItemInternal*> ItemInternalArrayView;

typedef ArrayView<ItemInternal*> ItemInternalMutableArrayView;

class IItemOperationByBasicType;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class ItemInfoListView;
template <typename ItemType> class ItemInfoListViewT;
class NodeInfoListView;
class EdgeInfoListView;
class FaceInfoListView;
class CellInfoListView;
class ParticleInfoListView;
class DoFInfoListView;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Politique d'utilisation des connectivités.
 *
 * Cette énumération sert à faire la transition entre les connectivités
 * historiques et la nouvelle implémentation.
 *
 * Actuellement, seule la valeur InternalConnectivityPolicy::NewOnly est utilisée
 */
enum class InternalConnectivityPolicy
{
  /*!
   * \brief Connectivités historiques.
   *
   * Ce mode est identique au mode d'avant l'incorporation des nouvelles
   * connectivités. Son empreinte mémoire est la plus faible de tous les modes
   * disponibles.
   * \warning Ce mode n'est plus opérationnel.
   */
  Legacy,
  /*!
   * \brief Utilise les connectivités historiques et alloue les accesseurs pour
   * ces connectivités
   * \warning Ce mode n'est plus opérationnel.
   */
  LegacyAndAllocAccessor,
  /*!
   * \brief Alloue les anciennes et les nouvelles connectivités
   * et utilise les anciennes via les nouveaux accesseurs dans ItemInternal.
   * \warning Ce mode n'est plus opérationnel.
   */
  LegacyAndNew,
  /*!
   * \brief Alloue les anciennes et les nouvelles connectivités
   * et utilise les nouvelles via les nouveaux accesseurs dans ItemInternal.
   * \warning Ce mode n'est plus opérationnel.
   */
  NewAndLegacy,
  /*!
   * \brief Alloue les anciennes et les nouvelles connectivités
   * utilise les nouvelles via les nouveaux accesseurs dans ItemInternal
   * et s'appuie sur un graphe de dépendances des familles (Familles,Connectivités).
   * \warning Ce mode n'est plus opérationnel.
   */
  NewWithDependenciesAndLegacy,
  /*!
   * \brief Alloue uniquement les nouvelles connectivités
   */
  NewOnly,
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Classe statique pour avoir des informations sur la
 * configuration des connectivités.
 */
class ARCANE_CORE_EXPORT InternalConnectivityInfo
{
 public:
  //! Vrai si les anciennes connectivités sont actives
  static constexpr bool hasLegacyConnectivity(InternalConnectivityPolicy) { return false; }
  //! Vrai si les nouvelles connectivités sont actives
  static constexpr bool hasNewConnectivity(InternalConnectivityPolicy) { return true; }
  /*!
   * \brief Indique si on utilise les nouvelles connectivités pour accéder
   * aux entités dans ItemInternal.
   */
  static constexpr bool useNewConnectivityAccessor(InternalConnectivityPolicy) { return true; }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \ingroup Mesh
 * \brief Vecteur de noeuds.
 */
typedef ItemVectorT<Node> NodeVector;
/*!
 * \ingroup Mesh
 * \brief Vecteur d'arêtes
 * \note Cette classe n'est pas implémentée.
 */
typedef ItemVectorT<Edge> EdgeVector;
/*!
 * \ingroup Mesh
 * \brief Vecteur de faces.
 */
typedef ItemVectorT<Face> FaceVector;
/*!
 * \ingroup Mesh
 * \brief Vecteur de mailles.
 */
typedef ItemVectorT<Cell> CellVector;
/*!
 * \ingroup Mesh
 * \brief Vecteur de particules.
 */
typedef ItemVectorT<Particle> ParticleVector;
/*!
 * \ingroup Mesh
 * \brief Vecteur de degres de liberte.
 */
typedef ItemVectorT<DoF> DoFVector;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \internal
 * \brief Caractéristiques des éléments du maillage.
 *
 * A spécialiser par type d'élément.
 */
template<class T>
class ItemTraitsT
{
 public:
   //! Genre de l'entité
  static eItemKind kind() { return IK_Unknown; }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \internal
 * \brief Caractéristiques des entités du maillage de type \c Node.
 */
template<>
class ItemTraitsT<Node>
{
 public:

  //! Type de cette classe
  typedef ItemTraitsT<Node> ItemTraitsType;
  //! Type de l'entité de maillage
  typedef Node ItemType;
  //! Type du groupe de l'entité
  typedef NodeGroup ItemGroupType;
  //! Type du localId()
  typedef NodeLocalId LocalIdType;

 public:

  //! Genre de l'entité
  static eItemKind kind() { return IK_Node; }

  //! Nom de la famille par défaut associée
  static const char* defaultFamilyName() { return "Node"; }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \internal
 * \brief Caractéristiques des entités du maillage de type \c Edge.
 */
template<>
class ItemTraitsT<Edge>
{
 public:

  //! Type de cette classe
  typedef ItemTraitsT<Edge> ItemTraitsType;
  //! Type de l'entité de maillage
  typedef Edge ItemType;
  //! Type du groupe de l'entité
  typedef EdgeGroup ItemGroupType;
  //! Type du localId()
  typedef EdgeLocalId LocalIdType;

 public:

  //! Genre de l'entité
  static eItemKind kind() { return IK_Edge; }

  //! Nom de la famille par défaut associée
  static const char* defaultFamilyName() { return "Edge"; }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \internal
 * \brief Caractéristiques des entités du maillage de type \c Face.
 */
template<>
class ItemTraitsT<Face>
{
 public:

  //! Type de cette classe
  typedef ItemTraitsT<Face> ItemTraitsType;
  //! Type de l'entité de maillage
  typedef Face ItemType;
  //! Type du groupe de l'entité
  typedef FaceGroup ItemGroupType;
  //! Type du localId()
  typedef FaceLocalId LocalIdType;

 public:

  //! Genre de l'entité
  static eItemKind kind() { return IK_Face; }

  //! Nom de la famille par défaut associée
  static const char* defaultFamilyName() { return "Face"; }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \internal
 * \brief Caractéristiques des entités du maillage de type \c Cell.
 */
template<>
class ItemTraitsT<Cell>
{
 public:

  //! Type de cette classe
  typedef ItemTraitsT<Cell> ItemTraitsType;
  //! Type de l'entité de maillage
  typedef Cell ItemType;
  //! Type du groupe de l'entité
  typedef CellGroup ItemGroupType;
  //! Type du localId()
  typedef CellLocalId LocalIdType;

 public:

  //! Genre de l'entité
  static eItemKind kind() { return IK_Cell; }

  //! Nom de la famille par défaut associée
  static const char* defaultFamilyName() { return "Cell"; }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \internal
 * \brief Caractéristiques des entités du maillage de type \c Particle.
 */
template<>
class ItemTraitsT<Particle>
{
 public:

  //! Type de cette classe
  typedef ItemTraitsT<Particle> ItemTraitsType;
  //! Type de l'entité de maillage
  typedef Particle ItemType;
  //! Type du groupe de l'entité
  typedef ParticleGroup ItemGroupType;
  //! Type du localId()
  typedef ParticleLocalId LocalIdType;

 public:

  //! Genre de l'entité
  static eItemKind kind() { return IK_Particle; }

  //! Nom de la famille par défaut associée
  static const char* defaultFamilyName() { return nullptr; }
};


/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \internal
 * \brief Caractéristiques des entités du maillage de type \c DoF
 */
template<>
class ItemTraitsT<DoF>
{
 public:

  //! Type de cette classe
  typedef ItemTraitsT<DoF> ItemTraitsType;
  //! Type de l'entité de maillage
  typedef DoF ItemType;
  //! Type du groupe de l'entité
  typedef DoFGroup ItemGroupType;
  //! Type du localId()
  using LocalIdType = ItemLocalIdT<DoF>;

 public:

  //! Genre de l'entité
  static eItemKind kind() { return IK_DoF; }

  // NOTE: GG: devrait être nullptr car pas de défaut pour la famille de DoF ?
  //! Nom de la famille par défaut associée
  static const char* defaultFamilyName() { return nullptr; }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Caractéristique associées aux 'localId()'.
 */
template<typename ItemType>
class ItemLocalIdTraitsT
{
 public:

  //! Type du localId()
  using LocalIdType = typename ItemTraitsT<ItemType>::LocalIdType;
};

//! Spécialisation pour 'Item' qui n'a pas de 'ItemTraitsT'.
template<>
class ItemLocalIdTraitsT<Item>
{
 public:
  //! Type du localId()
  using LocalIdType = ItemLocalId;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  
