// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* AMRPatchPositionSignature.h                                 (C) 2000-2026 */
/*                                                                           */
/* Calcul des signatures d'une position de patch.                            */
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
 * \brief Classe permettant de gérer les signatures d'un patch.
 *
 * La signature d'un patch dans une dimension correspond au nombre de mailles
 * à raffiner dans l'autre dimension (ou dans les deux autres dimensions en
 * 3D).
 * Exemple : pour la signature en X, pour chaque Xn, pour tout Yn et pour tout Zn,
 * on compte le nombre de mailles à raffiner en (Xn, Yn, Zn).
 */
class AMRPatchPositionSignature
{
 public:

  AMRPatchPositionSignature();
  AMRPatchPositionSignature(const AMRPatchPosition& patch, ICartesianMesh* cmesh);
  ~AMRPatchPositionSignature() = default;

 private:

  AMRPatchPositionSignature(const AMRPatchPosition& patch, ICartesianMesh* cmesh, Int32 nb_cut);

 public:

  /*!
  * \brief Méthode permettant de retirer les 0 au début et à la fin des
  * signatures.
  * \note Méthode appelée par \a compute().
  * \warning La méthode \a fillSig() doit avoir été appelée avant.
  */
  void compress();

  /*!
  * \brief Méthode permettant de calculer les signatures.
  * \note Méthode appelée par \a compute().
  * Méthode collective.
  */
  void fillSig();

  /*!
  * \brief Méthode permettant de savoir si les signatures sont valides.
  */
  bool isValid() const;

  /*!
  * \brief Méthode permettant de savoir si le patch peut être découpé en deux
  * via la méthode \a cut().
  */
  bool canBeCut() const;

  /*!
  * \brief Méthode permettant de calculer les signatures d'un patch.
  * Cette méthode doit être appelée après construction.
  */
  void compute();

  /*!
  * \brief Méthode permettant de connaitre l'efficacité du patch.
  */
  Real efficacity() const;

  /*!
  * \brief Méthode permettant de découper le patch.
  * Le patch n'est pas modifié.
  * \param dim La dimension du \a cut_point.
  * \param cut_point La position de la découpe.
  * \return Les deux patchs résultant de la découpe.
  */
  std::pair<AMRPatchPositionSignature, AMRPatchPositionSignature> cut(Integer dim, CartCoord cut_point) const;

  /*!
  * \brief Méthode permettant de récupérer la signature X.
  */
  ConstArrayView<CartCoord> sigX() const;

  /*!
  * \brief Méthode permettant de récupérer la signature Y.
  */
  ConstArrayView<CartCoord> sigY() const;

  /*!
  * \brief Méthode permettant de récupérer la signature Z.
  */
  ConstArrayView<CartCoord> sigZ() const;

  /*!
  * \brief Méthode permettant de récupérer une copie du patch.
  */
  AMRPatchPosition patch() const;

  ICartesianMesh* mesh() const;

  /*!
  * \brief Méthode permettant de savoir si le patch peut encore être découpé.
  */
  bool stopCut() const;

  /*!
  * \brief Méthode permettant de définir si le patch peut encore être découpé.
  */
  void setStopCut(bool stop_cut);

  /*!
  * \brief Méthode permettant de savoir si la méthode \a compute() a déjà été appelée.
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
