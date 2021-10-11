// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2021 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ItemTypes.h                                                 (C) 2000-2017 */
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

ARCANE_BEGIN_NAMESPACE

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

class IMesh;
class IPrimaryMesh;
class IItemFamily;

class ItemLocalId;
class NodeLocalId;
class CellLocalId;
class EdgeLocalId;
class FaceLocalId;
class ParticleLocalId;
class IndexedItemConnectivityViewBase;
class IndexedItemConnectivityAccessor;

class ItemGroup;
template<typename T> class ItemGroupT;

class ItemPairGroup;
template<typename ItemKind,typename SubItemKind> class ItemPairGroupT;

class ItemVector;
template<typename T> class ItemVectorT;

template <typename ItemType>
class ItemLocalIdView;

template<typename ItemType1,typename ItemType2>
class IndexedItemConnectivityView;

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
template<typename ItemType>
class ItemEnumeratorT;

/*!
 * \ingroup Mesh
 * \brief Vue sur un vecteur d'entité.
 */
class ItemVectorView;
template<typename ItemType>
class ItemVectorViewT;

class ItemEnumerator;
template<typename ItemType>
class ItemEnumeratorT;

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

/*! \brief Type de la liste interne des entités
  
  \deprecated Utiliser ItemInternalArrayView.
  
*/
typedef ConstArrayView<ItemInternal*> ItemInternalList;

typedef ConstArrayView<ItemInternal*> ItemInternalArrayView;

typedef ArrayView<ItemInternal*> ItemInternalMutableArrayView;

class IItemOperationByBasicType;

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
  static bool hasLegacyConnectivity(InternalConnectivityPolicy p);
  //! Vrai si les nouvelles connectivités sont actives
  static bool hasNewConnectivity(InternalConnectivityPolicy p);
  /*!
   * \brief Indique si on utilise les nouvelles connectivités pour accéder
   * aux entités dans ItemInternal.
   */
  static bool useNewConnectivityAccessor(InternalConnectivityPolicy p);
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
  static const char* defaultFamilyName() { return 0; }
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

 public:

  //! Genre de l'entité
  static eItemKind kind() { return IK_DoF; }

  //! Nom de la famille par défaut associée
  static const char* defaultFamilyName() { return "DoF"; }
};


/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_END_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  
