// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* AMRPatchPositionSignature.h                                 (C) 2000-2026 */
/*                                                                           */
/* Calculation of patch position signatures.                                 */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CARTESIANMESH_INTERNAL_AMRPATCHPOSITIONSIGNATURE_H
#define ARCANE_CARTESIANMESH_INTERNAL_AMRPATCHPOSITIONSIGNATURE_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/cartesianmesh/CartesianMeshGlobal.h"

#include "arcane/utils/UniqueArray.h"

#include "arcane/cartesianmesh/AMRPatchPosition.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Class for managing patch signatures.
 *
 * The signature of a patch in one dimension corresponds to the number of meshes
 * to refine in the other dimension (or in the other two dimensions in
 * 3D).
 * Example: for the X signature, for each Xn, for all Yn and for all Zn,
 * we count the number of meshes to refine at (Xn, Yn, Zn).
 */
class AMRPatchPositionSignature
{
 public:

  AMRPatchPositionSignature();
  AMRPatchPositionSignature(const AMRPatchPosition& patch, ICartesianMesh* cmesh);
  AMRPatchPositionSignature(const AMRPatchPositionSignature&) = default;
  ~AMRPatchPositionSignature() = default;

 private:

  AMRPatchPositionSignature(const AMRPatchPosition& patch, ICartesianMesh* cmesh, Int32 nb_cut);

public:

  AMRPatchPositionSignature& operator=(const AMRPatchPositionSignature&) = default;

 public:

  /*!
  * \brief Method for removing zeros at the beginning and end of the
  * signatures.
  * \note Method called by \a compute().
  * \warning The method \a fillSig() must have been called before.
  */
  void compress();

  /*!
  * \brief Method for calculating the signatures.
  * \note Method called by \a compute().
  * Collective method.
  */
  void fillSig();

  /*!
  * \brief Method for determining if the signatures are valid.
  */
  bool isValid() const;

  /*!
  * \brief Method for determining if the patch can be cut into two
  * via the \a cut() method.
  */
  bool canBeCut() const;

  /*!
  * \brief Method for calculating the signatures of a patch.
  * This method must be called after construction.
  */
  void compute();

  /*!
  * \brief Method to determine the patch's efficiency.
  */
  Real efficacity() const;

  /*!
  * \brief Method for cutting the patch.
  * The patch is not modified.
  * \param dim The dimension of the \a cut_point.
  * \param cut_point The position of the cut.
  * \return The two patches resulting from the cut.
  */
  std::pair<AMRPatchPositionSignature, AMRPatchPositionSignature> cut(Integer dim, CartCoord cut_point) const;

  /*!
  * \brief Method for retrieving the X signature.
  */
  ConstArrayView<CartCoord> sigX() const;

  /*!
  * \brief Method for retrieving the Y signature.
  */
  ConstArrayView<CartCoord> sigY() const;

  /*!
  * \brief Method for retrieving the Z signature.
  */
  ConstArrayView<CartCoord> sigZ() const;

  /*!
  * \brief Method for retrieving a copy of the patch.
  */
  AMRPatchPosition patch() const;

  ICartesianMesh* mesh() const;

  /*!
  * \brief Method for determining if the patch can still be cut.
  */
  bool stopCut() const;

  /*!
  * \brief Method for defining whether the patch can still be cut.
  */
  void setStopCut(bool stop_cut);

  /*!
  * \brief Method for determining whether the \a compute() method has already been called.
  */
  bool isComputed() const;

 private:

  bool m_is_null;
  AMRPatchPosition m_patch;
  ICartesianMesh* m_mesh;
  Int32 m_nb_cut;
  bool m_stop_cut;

  ICartesianMeshNumberingMngInternal* m_numbering;

  bool m_have_cells;
  bool m_is_computed;

  UniqueArray<CartCoord> m_sig_x;
  UniqueArray<CartCoord> m_sig_y;
  UniqueArray<CartCoord> m_sig_z;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
