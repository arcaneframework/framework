// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* FaceReorienter.h                                            (C) 2000-2025 */
/*                                                                           */
/* Vérifie la bonne orientation d'une face et la réoriente le cas échéant.   */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_MESH_FACEREORIENTER_H
#define ARCANE_MESH_FACEREORIENTER_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/mesh/MeshGlobal.h"

#include "arcane/utils/Array.h"

#include "arcane/core/ItemTypes.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::mesh
{
class FaceFamily;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief  Cette fonction/classe réoriente les faces.
 * 
 * Cette classe sert par exemple à s'assurrer de la bonne orientation
 * des faces après un changement de topologie du maillage
 */
class ARCANE_MESH_EXPORT FaceReorienter
{
 public:

  /*! Constructeur.
   * \deprecated Utiliser FaceReorienter(IMesh*) à la place.
   */
  ARCANE_DEPRECATED_260 FaceReorienter(ITraceMng* tm);
  //! Constructeur.
  explicit FaceReorienter(IMesh* mesh);

 public:

  /*!
   * \deprecated Utiliser checkAndChangeOrientation(Face) à la place.
   */
  ARCANE_DEPRECATED_260 void checkAndChangeOrientation(ItemInternal* face);

  /*!
   * \deprecated Utiliser checkAndChangeOrientationAMR(Face) à la place.
   */
  ARCANE_DEPRECATED_260 void checkAndChangeOrientationAMR(ItemInternal* face);

  /*!
   * \brief Vérifie et éventuellement change l'orientation de la face.
   *
   * \param face face a réorienter
   */
  void checkAndChangeOrientation(Face face);

  /*!
   * \brief Vérifie et éventuellement change l'orientation de la face.
   *
   * \param face face a réorienter
   */
  // AMR
  void checkAndChangeOrientationAMR(Face face);

 private:

  ITraceMng* m_trace_mng = nullptr;
  IItemFamily* m_face_family = nullptr;
  UniqueArray<Int64> m_nodes_unique_id;
  UniqueArray<Int32> m_nodes_local_id;
  UniqueArray<Integer> m_face_nodes_index;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::mesh

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
