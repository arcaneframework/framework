﻿// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* FaceDirectionMng.cc                                         (C) 2000-2022 */
/*                                                                           */
/* Infos sur les faces d'une direction X Y ou Z d'un maillage structuré.     */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CARTESIANMESH_FACEDIRECTIONMNG_H
#define ARCANE_CARTESIANMESH_FACEDIRECTIONMNG_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/ArcaneTypes.h"
#include "arcane/Item.h"
#include "arcane/VariableTypedef.h"
#include "arcane/ItemEnumerator.h"

#include "arcane/cartesianmesh/CartesianMeshGlobal.h"
#include "arcane/cartesianmesh/CartesianItemDirectionInfo.h"

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
  CellLocalId previousCellId() const { return m_previous.itemLocalId(); }
  //! Maille après
  Cell nextCell() const { return m_next; }
  //! Maille après
  CellLocalId nextCellId() const { return m_next.itemLocalId(); }

 private:

  Cell m_previous;
  Cell m_next;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \ingroup ArcaneCartesianMesh
 * \brief Infos sur maille avant et après une face suivant une direction.
 *
 * Les instances de cette classe sont temporaires et construites via
 * FaceDirectionMng::dirFaceId().
 */
class ARCANE_CARTESIANMESH_EXPORT DirFaceLocalId
{
  friend FaceDirectionMng;

 private:

  ARCCORE_HOST_DEVICE DirFaceLocalId(CellLocalId n, CellLocalId p)
  : m_previous(p)
  , m_next(n)
  {}

 public:

  //! Maille avant
  ARCCORE_HOST_DEVICE CellLocalId previousCell() const { return m_previous; }
  //! Maille avant
  ARCCORE_HOST_DEVICE CellLocalId previousCellId() const { return m_previous; }
  //! Maille après
  ARCCORE_HOST_DEVICE CellLocalId nextCell() const { return m_next; }
  //! Maille après
  ARCCORE_HOST_DEVICE CellLocalId nextCellId() const { return m_next; }

 private:

  CellLocalId m_previous;
  CellLocalId m_next;
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

  using ItemDirectionInfo = impl::CartesianItemDirectionInfo;

 public:

  /*!
   * \brief Créé une instance vide.
   *
   * L'instance n'est pas valide tant que _internalInit() n'a pas été appelé.
   */
  FaceDirectionMng();

 public:

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

  //! Face direction correspondant à la face \a f.
  ARCCORE_HOST_DEVICE DirFaceLocalId dirFaceId(FaceLocalId f) const
  {
    return _dirFaceId(f);
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
    ItemDirectionInfo d = m_infos_view[local_id];
    return DirFace(m_cells[d.m_next_lid], m_cells[d.m_previous_lid]);
  }

  //! Face direction correspondant à la face de numéro local \a local_id
  ARCCORE_HOST_DEVICE DirFaceLocalId _dirFaceId(FaceLocalId local_id) const
  {
    ItemDirectionInfo d = m_infos_view[local_id];
    return DirFaceLocalId(CellLocalId(d.m_next_lid), CellLocalId(d.m_previous_lid));
  }

 private:

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
  void _internalInit(ICartesianMesh* cm, eMeshDirection dir, Integer patch_index);

  /*!
   * \internal
   * Détruit les ressources associées à l'instance.
   */
  void _internalDestroy();

  /*!
   * \brief Redimensionne le conteneur contenant les \a ItemDirectionInfo.
   *
   * Cela invalide les instances courantes de FaceDirectionMng.
   */
  void _internalResizeInfos(Int32 new_size);

 private:

  SmallSpan<ItemDirectionInfo> m_infos_view;
  CellInfoListView m_cells;
  eMeshDirection m_direction;
  Impl* m_p;

  void _computeCellInfos(const CellDirectionMng& cell_dm,
                         const VariableCellReal3& cells_center,
                         const VariableFaceReal3& faces_center);
  bool _hasFace(Cell cell, Int32 face_local_id) const;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  

