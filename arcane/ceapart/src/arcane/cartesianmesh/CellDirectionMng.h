// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2021 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* CellDirectionMng.cc                                         (C) 2000-2021 */
/*                                                                           */
/* Infos sur les mailles d'une direction X Y ou Z d'un maillage structuré.   */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CARTESIANMESH_CELLDIRECTIONMNG_H
#define ARCANE_CARTESIANMESH_CELLDIRECTIONMNG_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/ArcaneTypes.h"

#include "arcane/Item.h"
#include "arcane/ItemEnumerator.h"

#include "arcane/cartesianmesh/CartesianMeshGlobal.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \ingroup ArcaneCartesianMesh
 * \brief Maille avant et après une maille suivant une direction.
 *
 * Les instances de cette classe sont temporaires et construites via
 * CellDirectionMng::cell().
 */
class ARCANE_CARTESIANMESH_EXPORT DirCell
{
 public:
  DirCell(Cell n,Cell p) : m_previous(p), m_next(n){}
 public:
  //! Maille avant
  Cell previous() const { return m_previous; }
  //! Maille avant
  CellLocalId previousId() const { return CellLocalId(m_previous.localId()); }
  //! Maille après
  Cell next() const { return m_next; }
  //! Maille après
  CellLocalId nextId() const { return CellLocalId(m_next.localId()); }
 private:
  Cell m_previous;
  Cell m_next;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \ingroup ArcaneCartesianMesh
 * \brief Maille avec info directionnelle des noeuds.
 *
 * Les instances de cette classe sont temporaires et construites via
 * CellDirectionMng::cellNode().
 */
class ARCANE_CARTESIANMESH_EXPORT DirCellNode
{
  friend CellDirectionMng;
 private:

  DirCellNode(Cell c,const Int32* nodes_indirection)
  : m_cell(c), m_nodes_indirection(nodes_indirection){}

 public:

  //! Maille associée
  Cell cell() const { return m_cell; }
  //! Maille associée
  CellLocalId cellId() const { return CellLocalId(m_cell.localId()); }

  //! Noeud devant à gauche dans la direction
  Node nextLeft() const { return m_cell.node(m_nodes_indirection[CNP_NextLeft]); }
  //! Noeud devant à droite dans la direction
  Node nextRight() const { return m_cell.node(m_nodes_indirection[CNP_NextRight]); }
  //! Noeud derrière à droite dans la direction
  Node previousRight() const { return m_cell.node(m_nodes_indirection[CNP_PreviousRight]); }
  //! Noeud derrière à gauche dans la direction
  Node previousLeft() const { return m_cell.node(m_nodes_indirection[CNP_PreviousLeft]); }

  //! Noeud devant à gauche dans la direction
  NodeLocalId nextLeftId() const { return NodeLocalId(m_cell.node(m_nodes_indirection[CNP_NextLeft])); }
  //! Noeud devant à droite dans la direction
  NodeLocalId nextRightId() const { return NodeLocalId(m_cell.node(m_nodes_indirection[CNP_NextRight])); }
  //! Noeud derrière à droite dans la direction
  NodeLocalId previousRightId() const { return NodeLocalId(m_cell.node(m_nodes_indirection[CNP_PreviousRight])); }
  //! Noeud derrière à gauche dans la direction
  NodeLocalId previousLeftId() const { return NodeLocalId(m_cell.node(m_nodes_indirection[CNP_PreviousLeft])); }

  //! Noeud devant à gauche dans la direction
  Node topNextLeft() const { return m_cell.node(m_nodes_indirection[CNP_TopNextLeft]); }
  //! Noeud devant à droite dans la direction
  Node topNextRight() const { return m_cell.node(m_nodes_indirection[CNP_TopNextRight]); }
  //! Noeud derrière à droite dans la direction
  Node topPreviousRight() const { return m_cell.node(m_nodes_indirection[CNP_TopPreviousRight]); }
  //! Noeud derrière à gauche dans la direction
  Node topPreviousLeft() const { return m_cell.node(m_nodes_indirection[CNP_TopPreviousLeft]); }

  //! Noeud devant à gauche dans la direction
  NodeLocalId topNextLeftId() const { return NodeLocalId(m_cell.node(m_nodes_indirection[CNP_TopNextLeft])); }
  //! Noeud devant à droite dans la direction
  NodeLocalId topNextRightId() const { return NodeLocalId(m_cell.node(m_nodes_indirection[CNP_TopNextRight])); }
  //! Noeud derrière à droite dans la direction
  NodeLocalId topPreviousRightId() const { return NodeLocalId(m_cell.node(m_nodes_indirection[CNP_TopPreviousRight])); }
  //! Noeud derrière à gauche dans la direction
  NodeLocalId topPreviousLeftId() const { return NodeLocalId(m_cell.node(m_nodes_indirection[CNP_TopPreviousLeft])); }

 private:
  
  Cell m_cell;
  const Int32* m_nodes_indirection;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \ingroup ArcaneCartesianMesh
 * \brief Maille avec info directionnelle des faces.
 *
 * Les instances de cette classe sont temporaires et construites via
 * CellDirectionMng::cellFace().
 */
class ARCANE_CARTESIANMESH_EXPORT DirCellFace
{
  friend CellDirectionMng;
 private:
  DirCellFace(Cell c,Int32 next_face_index,Int32 previous_face_index)
  : m_cell(c), m_next_face_index(next_face_index), m_previous_face_index(previous_face_index)
  {
  }

 public:

  //! Maille associée
  Cell cell() const { return m_cell; }
  //! Maille associée
  CellLocalId cellId() const { return CellLocalId(m_cell.localId()); }

  //! Face connectée à la maille d'après la maille courante dans la direction
  Face next() const { return m_cell.face(m_next_face_index); }
  //! Face connectée à la maille d'après la maille courante dans la direction
  FaceLocalId nextId() const { return FaceLocalId(next().localId()); }

  //! Face connectée à la maille d'avant la maille courante dans la direction
  Face previous() const { return m_cell.face(m_previous_face_index); }
  //! Face connectée à la maille d'avant la maille courante dans la direction
  FaceLocalId previousId() const { return FaceLocalId(previous().localId()); }

  //! Indice locale dans la maille de la face next()
  Int32 nextLocalIndex() const { return m_next_face_index; }

  //! Indice locale dans la maille de la face previous()
  Int32 previousLocalIndex() const { return m_previous_face_index; }

 private:
  
  Cell m_cell;
  Int32 m_next_face_index;
  Int32 m_previous_face_index;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \ingroup ArcaneCartesianMesh
 * \brief Infos sur les mailles d'une direction spécifique X,Y ou Z d'un maillage structuré.
 *
 * Cette classe contient les informations pour retourner la liste
 * des mailles dans une direction donnée et pour ces mailles
 * connaitre la maille avant et après dans cette direction.
 *
 * Les instances de cette classe sont gérées par un ICartesianMesh et sont
 * temporaires. Il ne faut pas les conserver d'une itération à l'autre
 * car elles sont invalidées si le maillage change.
 *
 * Cette classe à une sémantique par référence.
 *
 * Par exemple, pour itérer sur les mailles de la direction X:
 
 \code
 * ICartesianMesh* cartesian_mesh = ...;
 * CellDirectionMng cdm(cartesian_mesh->cellDirection(MD_DirX));
 * ENUMERATE_CELL(icell,cdm.allCells()){
 *   DirCell dir_cell(cdm[icell]);
 *   Cell next = dir_cell.next();
 *   Cell prev = dir_cell.previous();
 * }
 \endcode
 *
 */
class ARCANE_CARTESIANMESH_EXPORT CellDirectionMng
{
  friend CartesianMesh;
  friend CartesianMeshPatch;
  class Impl;
  static const int MAX_NB_NODE = 8;

 private:

  struct ItemDirectionInfo
  {
   public:
    /*!
     * \brief Constructeur par défaut.
     * \warning Les valeurs m_next_item et m_previous_item sont initialisées
     * à nullptr.
     */
    ItemDirectionInfo()
    : m_next_item(nullptr), m_previous_item(nullptr){}
    ItemDirectionInfo(ItemInternal* next,ItemInternal* prev)
    : m_next_item(next), m_previous_item(prev){}
   public:
    //! entité après l'entité courante dans la direction
    ItemInternal* m_next_item;
    //! entité avant l'entité courante dans la direction
    ItemInternal* m_previous_item;
  };
 public:
  
  //! Créé une instance vide. L'instance n'est pas valide tant que init() n'a pas été appelé.
  CellDirectionMng();
  CellDirectionMng(const CellDirectionMng& rhs);
  ~CellDirectionMng();

  //! Maille direction correspondant à la maille \a c.
  DirCell cell(Cell c)
  {
    return _cell(c.localId());
  }
  //! Maille direction correspondant à la maille \a c.
  DirCell cell(CellLocalId c)
  {
    return _cell(c.localId());
  }

  //! Maille avec infos directionnelles aux noeuds correspondant à la maille \a c.
  DirCellNode cellNode(Cell c) const
  {
    return DirCellNode(c,m_nodes_indirection);
  }

  //! Maille avec infos directionnelles aux noeuds correspondant à la maille \a c.
  DirCellNode cellNode(CellLocalId c) const
  {
    return DirCellNode(m_items[c.localId()],m_nodes_indirection);
  }

  //! Maille avec infos directionnelles aux faces correspondant à la maille \a c.
  DirCellFace cellFace(Cell c) const
  {
    return DirCellFace(c,m_next_face_index,m_previous_face_index);
  }
  //! Maille avec infos directionnelles aux faces correspondant à la maille \a c.
  DirCellFace cellFace(CellLocalId c) const
  {
    return DirCellFace(m_items[c.localId()],m_next_face_index,m_previous_face_index);
  }

  //! Groupe de toutes les mailles dans la direction.
  CellGroup allCells() const;

  /*!
   * \brief Groupe de toutes les mailles internes dans la direction.
   *
   * Une maille est considérée comme interne si sa maille
   * avant ou après n'est pas nulle.
   */

  CellGroup innerCells() const;
  /*!
   * \brief Groupe de toutes les mailles externes dans la direction.
   *
   * Une maille est considérée comme externe si sa maille
   * avant ou après est nulle.
   */
  CellGroup outerCells() const;

  //! Maille direction correspondant à la maille \a c.
  DirCell operator[](Cell c)
  {
    return _cell(c.localId());
  }

  //! Maille direction correspondant à la maille \a c.
  DirCell operator[](CellLocalId c)
  {
    return _cell(c.localId());
  }

  //! Maille direction correspondant à l'itérateur de la maille \a icell.
  DirCell operator[](CellEnumerator icell)
  {
    return _cell(icell.itemLocalId());
  }

  /*!
   * \brief Nombre global de mailles dans cette direction.
   *
   * \note La valeur retournée n'est valide que si le
   * maillage a été créé avec un générateur spécifique, tel
   * le SodMeshGenerator ou le CartesianMeshGenerator. Si ce n'est
   * pas le cas, la valeur retournée vaut (-1)
   */
  Int64 globalNbCell() const { return m_global_nb_cell; }

  /*!
   * \brief Nombre de mailles propres dans cette direction.
   *
   * \note La valeur retournée n'est valide que si le
   * maillage a été créé avec un générateur spécifique, tel
   * le SodMeshGenerator ou le CartesianMeshGenerator. Si ce n'est
   * pas le cas, la valeur retournée vaut (-1)
   */
  Int32 ownNbCell() const { return m_own_nb_cell; }

  /*!
   * \brief Offset dans cette direction du sous-domaine.
   *
   * En supposant que le maillage cartésien global est découpé en
   * plusieurs sous-domaines rectangulaires qui forment une grille,
   * cette méthode retourne la position dans cette grille de ce sous-domaine
   * pour cette direction.
   *
   * \warning L'utilisation de cette méthode suppose que chaque
   * sous-domaine est parallélépipédique (en 3D) ou rectangulaire (en 2D)
   * ce qui n'est pas forcément le cas, notamment avec des mécanismes
   * d'équilibrage de charge par migration de maille.
   *
   * \note La valeur retournée n'est valide que si le
   * maillage a été créé avec un générateur spécifique, tel que
   * le CartesianMeshGenerator. Si ce n'est pas le cas,
   * la valeur retournée vaut (-1)
   */
  Int32 subDomainOffset() const { return m_sub_domain_offset; }

  /*!
   * \brief Offset dans cette direction de la première maille propre de ce sous-domaine.
   *
   * En supposant que le maillage cartésien global est découpé en
   * plusieurs sous-domaines rectangulaires qui forment une grille,
   * cette méthode retourne la position dans cette grille de la première
   * maille propre de ce sous-domaine pour cette direction.
   *
   * \warning L'utilisation de cette méthode suppose que chaque
   * sous-domaine est parallélépipédique (en 3D) ou rectangulaire (en 2D)
   * ce qui n'est pas forcément le cas, notamment avec des mécanismes
   * d'équilibrage de charge par migration de maille.
   *
   * \note La valeur retournée n'est valide que si le
   * maillage a été créé avec un générateur spécifique, tel que
   * le CartesianMeshGenerator. Si ce n'est pas le cas,
   * la valeur retournée vaut (-1)
   */
  Int64 ownCellOffset() const { return m_own_cell_offset; }

 private:
  
  //! Maille direction correspondant à la maille de numéro local \a local_id
  DirCell _cell(Int32 local_id) const
  {
    Cell next = Cell(m_infos[local_id].m_next_item);
    Cell prev = Cell(m_infos[local_id].m_previous_item);
    return DirCell(next,prev);
  }

  void setNodesIndirection(Int32ConstArrayView nodes_indirection);

 protected:

  /*!
   * \internal
   * \brief Usage interne à Arcane. Calcul les entités internes et externes.
   * Suppose que init() a été appelé.
   */
  void _internalComputeInnerAndOuterItems(const ItemGroup& items);

  /*!
   * \internal
   * Initialise l'instance.
   */
  void _internalInit(ICartesianMesh* cm,eMeshDirection dir,Integer patch_index);

  /*!
   * \internal
   * Détruit les ressources associées à l'instance.
   */
  void _internalDestroy();

  /*!
   * \internal
   * Positionne les indices locaux de la face vers la maille d'après et d'avant.
   */
  void _internalSetLocalFaceIndex(Int32 next_index,Int32 previous_index)
  {
    m_next_face_index = next_index;
    m_previous_face_index = previous_index;
  }

 public:

  //! Valeur de la direction
  eMeshDirection direction() const
  {
    return m_direction;
  }

 private:

  SharedArray<ItemDirectionInfo> m_infos;
  eMeshDirection m_direction;
  Impl* m_p;
  ItemInternalList m_items;
  Int32 m_nodes_indirection[MAX_NB_NODE];
  Int32 m_next_face_index;
  Int32 m_previous_face_index;
  Int32 m_sub_domain_offset;
  Int32 m_own_nb_cell;
  Int64 m_global_nb_cell;
  Int64 m_own_cell_offset;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  
