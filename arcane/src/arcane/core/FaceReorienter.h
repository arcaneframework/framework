// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* FaceReorienter.h                                            (C) 2000-2025 */
/*                                                                           */
/* Checks the correct orientation of a face and reorients it if necessary.   */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_FACEREORIENTER_H
#define ARCANE_CORE_FACEREORIENTER_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/Array.h"

#include "arcane/core/ItemTypes.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief This function/class reorients faces.
 * 
 * This class is used, for example, to ensure the correct orientation
 * of faces after a mesh topology change.
 */
class ARCANE_CORE_EXPORT FaceReorienter
{
 public:

  /*! Constructor.
   * \deprecated Use FaceReorienter(IMesh*) instead.
   */
  ARCANE_DEPRECATED_260 FaceReorienter(ITraceMng* tm);
  //! Constructor.
  explicit FaceReorienter(IMesh* mesh);

 public:

  /*!
   * \deprecated Use checkAndChangeOrientation(Face) instead.
   */
  ARCANE_DEPRECATED_260 void checkAndChangeOrientation(ItemInternal* face);

  /*!
   * \deprecated Use checkAndChangeOrientationAMR(Face) instead.
   */
  ARCANE_DEPRECATED_260 void checkAndChangeOrientationAMR(ItemInternal* face);

  /*!
   * \brief Checks and optionally changes the orientation of the face.
   *
   * \param face face to reorient
   */
  void checkAndChangeOrientation(Face face);

  /*!
   * \brief Checks and optionally changes the orientation of the face.
   *
   * \param face face to reorient
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
