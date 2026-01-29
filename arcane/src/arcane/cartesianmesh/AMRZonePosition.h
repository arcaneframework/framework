// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* AMRZonePosition.h                                           (C) 2000-2026 */
/*                                                                           */
/* Definition d'une zone 2D ou 3D d'un maillage.                             */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CARTESIANMESH_AMRBOXPOSITION_H
#define ARCANE_CARTESIANMESH_AMRBOXPOSITION_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/cartesianmesh/CartesianMeshGlobal.h"
#include "arcane/utils/Real3.h"
#include "arcane/core/VariableTypes.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Classe permettant de définir une zone d'un maillage.
 */
class ARCANE_CARTESIANMESH_EXPORT AMRZonePosition
{
 public:

  /*!
   * \brief Constructeur de zone 3D.
   * \param position La position géométrique de la zone.
   * \param length La taille de la zone.
   */
  AMRZonePosition(const Real3& position, const Real3& length)
  : m_position(position)
  , m_length(length)
  , m_is_3d(true)
  {}

  /*!
   * \brief Constructeur de zone 2D.
   * \param position La position géométrique de la zone.
   * \param length La taille de la zone.
   */
  AMRZonePosition(const Real2& position, const Real2& length)
  : m_position(position)
  , m_length(length)
  , m_is_3d(false)
  {}

 public:

  /*!
   * \brief Méthode permettant de retrouver la position de la zone.
   * \return La position géométrique de la zone.
   */
  Real3 position() const
  {
    return m_position;
  }

  /*!
   * \brief Méthode permettant de retrouver la taille de la zone.
   * \return La taille de la zone.
   */
  Real3 length() const
  {
    return m_length;
  }

  /*!
   * \brief Méthode permettant de savoir si c'est une zone 2D ou 3D.
   * \return True si c'est une zone 3D.
   */
  bool is3d() const
  {
    return m_is_3d;
  }

  /*!
   * \brief Méthode permettant de retrouver les mailles incluses dans la zone.
   * \param mesh Le maillage.
   * \param cells_local_id Le tableau qui contiendra les localIds des mailles de la zone.
   *                       Attention : le tableau sera d'abord effacé.
   */
  void cellsInPatch(IMesh* mesh, UniqueArray<Int32>& cells_local_id) const;

  /*!
   * \brief Méthode permettant de retrouver les mailles incluses dans la zone.
   * Un objet AMRPatchPosition désignant la position du patch est aussi rempli.
   * \param mesh Le maillage.
   * \param cells_local_id Le tableau qui contiendra les localIds des mailles de la zone.
   *                       Attention : le tableau sera d'abord effacé.
   * \param position [OUT] La position du patch.
   */
  void cellsInPatch(ICartesianMesh* mesh, UniqueArray<Int32>& cells_local_id, AMRPatchPosition& position) const;

  /*!
   * \brief Méthode permettant de convertir ce AMRZonePosition en
   * AMRPatchPosition.
   *
   * \warning La taille de la couche de mailles de recouvrement n'est pas
   * correctement défini ! À faire après l'appel à cette méthode.
   *
   * \param mesh Le maillage cartesien.
   * \return Le AMRPatchPosition correspondant.
   */
  AMRPatchPosition toAMRPatchPosition(ICartesianMesh* mesh) const;

 private:

  Real3 m_position;
  Real3 m_length;
  bool m_is_3d;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
