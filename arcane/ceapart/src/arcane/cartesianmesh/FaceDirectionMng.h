// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2021 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* FaceDirectionMng.cc                                         (C) 2000-2021 */
/*                                                                           */
/* Infos sur les faces d'une direction X Y ou Z d'un maillage structuré.     */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CARTESIANMESH_FACEDIRECTIONMNG_H
#define ARCANE_CARTESIANMESH_FACEDIRECTIONMNG_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/ArcaneTypes.h"
#include "arcane/cartesianmesh/CartesianMeshGlobal.h"

#include "arcane/Item.h"
#include "arcane/VariableTypedef.h"
#include "arcane/ItemEnumerator.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \ingroup ArcaneCartesianMesh
 * \brief Infos sur maille avant et après une face suivant une direction.
 *
 * Les instances de cette classe sont temporaires et construites via
 * FaceDirectionMng::face().
 */
class ARCANE_CARTESIANMESH_EXPORT DirFace
{
  friend FaceDirectionMng;
 private:
  DirFace(Cell n,Cell p) : m_previous(p), m_next(n){}
 public:
  //! Maille avant
  Cell previousCell() const { return m_previous; }
  //! Maille avant
  CellLocalId previousCellId() const { return CellLocalId(m_previous.localId()); }
  //! Maille après
  Cell nextCell() const { return m_next; }
  //! Maille après
  CellLocalId nextCellId() const { return CellLocalId(m_next.localId()) ; }
 private:
  Cell m_previous;
  Cell m_next;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \ingroup ArcaneCartesianMesh
 * \brief Infos sur les face d'une direction spécifique X,Y ou Z
 * d'un maillage structuré.
 */
class ARCANE_CARTESIANMESH_EXPORT FaceDirectionMng
{
  friend CartesianMeshImpl;
  friend CartesianMeshPatch;
  class Impl;

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
  FaceDirectionMng();
  ~FaceDirectionMng();

  //! Face direction correspondant à la face \a f.
  DirFace face(Face f) const
  {
    return _face(f.localId());
  }
  //! Face direction correspondant à la face \a f.
  DirFace face(FaceLocalId f) const
  {
    return _face(f.localId());
  }

  //! Groupe de toutes les faces dans la direction.
  FaceGroup allFaces() const;

  /*!
   * \brief Groupe de toutes les faces internes dans la direction.
   *
   * Une face est considérée comme interne si sa maille
   * devant et derrière n'est pas nulle.
   */

  FaceGroup innerFaces() const;

  /*!
   * \brief Groupe de toutes les faces externes dans la direction.
   *
   * Une face est considérée comme externe si sa maille
   * devant ou derrière est nulle.
   */
  FaceGroup outerFaces() const;

  //! Face direction correspondant à la face \a f.
  DirFace operator[](Face f) const
  {
    return _face(f.localId());
  }

  //! Face direction correspondant à la face \a f.
  DirFace operator[](FaceLocalId f) const
  {
    return _face(f.localId());
  }

  //! Face direction correspondant à l'itérateur de la face \a iface
  DirFace operator[](FaceEnumerator iface) const
  {
    return _face(iface.itemLocalId());
  }

  //! Valeur de la direction
  eMeshDirection direction() const
  {
    return m_direction;
  }

 private:
  
  //! Face direction correspondant à la face de numéro local \a local_id
  DirFace _face(Int32 local_id) const
  {
    Cell next = Cell(m_infos[local_id].m_next_item);
    Cell prev = Cell(m_infos[local_id].m_previous_item);
    return DirFace(next,prev);
  }

 protected:

  /*!
   * \internal
   * \brief Calcule les informations sur les faces associées aux mailles de
   * la direction \a cell_dm.
   * Suppose que _internalInit() a été appelé.
   */
  void _internalComputeInfos(const CellDirectionMng& cell_dm,
                             const VariableCellReal3& cells_center,
                             const VariableFaceReal3& faces_center);

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

 private:

  SharedArray<ItemDirectionInfo> m_infos;
  eMeshDirection m_direction;
  Impl* m_p;

  void _computeCellInfos(const CellDirectionMng& cell_dm,
                         const VariableCellReal3& cells_center,
                         const VariableFaceReal3& faces_center);
  bool _hasFace(Cell cell,Int32 face_local_id) const;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  

