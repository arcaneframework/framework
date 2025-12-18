// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* AMRPatchPositionSignatureCut.h                              (C) 2000-2025 */
/*                                                                           */
/* Méthodes de découpages de patchs selon leurs signatures.                  */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CARTESIANMESH_INTERNAL_AMRPATCHPOSITIONSIGNATURECUT_H
#define ARCANE_CARTESIANMESH_INTERNAL_AMRPATCHPOSITIONSIGNATURECUT_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/cartesianmesh/CartesianMeshGlobal.h"
#include "arcane/utils/UniqueArray.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Classe permettant de découper un patch en plusieurs petits patchs.
 */
class AMRPatchPositionSignatureCut
{
 public:
  AMRPatchPositionSignatureCut();
  ~AMRPatchPositionSignatureCut();

 public:

  /*!
  * \brief Méthode permettant de chercher le meilleur point pour effectuer
  * une découpe.
  * \param sig La signature sur laquelle la recherche doit se faire.
  * \return Le meilleur point pour la découpe (-1 si problème).
  */
  static CartCoord _cutDim(ConstArrayView<CartCoord> sig);

  /*!
  * \brief Méthode permettant de découper un patch en deux.
  * \param sig Le patch à découper.
  * \return Les deux patchs resultant de la découpe.
  */
  static std::pair<AMRPatchPositionSignature, AMRPatchPositionSignature> cut(const AMRPatchPositionSignature& sig);

  /*!
  * \brief Méthode permettant de découper le ou les patchs du tableau \a sig_array_a.
  * \param sig_array_a [IN/OUT] Le tableau de patchs.
  */
  static void cut(UniqueArray<AMRPatchPositionSignature>& sig_array_a);
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  

