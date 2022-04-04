// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* FaceReorienter.h                                            (C) 2000-2018 */
/*                                                                           */
/* Vérifie la bonne orientation d'une face et la réoriente le cas échéant.   */
/*---------------------------------------------------------------------------*/
#ifndef FACE_REORIENTER_H
#define FACE_REORIENTER_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/mesh/MeshGlobal.h"

#include "arcane/utils/TraceAccessor.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_BEGIN_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class ItemInternal;
class IMesh;
class Face;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_MESH_BEGIN_NAMESPACE

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
  FaceReorienter(IMesh* mesh);
  //! Destructeur
  ~FaceReorienter();

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

  ITraceMng* m_trace_mng;
  FaceFamily* m_face_family;
  Int64UniqueArray m_nodes_unique_id;
  Int32UniqueArray m_nodes_local_id;
  IntegerUniqueArray m_face_nodes_index;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_MESH_END_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_END_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif // FACE_REORIENTER_H
