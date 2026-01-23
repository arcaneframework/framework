// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* CartesianPatch.h                                            (C) 2000-2025 */
/*                                                                           */
/* Patch AMR d'un maillage cartésien.                                        */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CARTESIANMESH_CARTESIANPATCH_H
#define ARCANE_CARTESIANMESH_CARTESIANPATCH_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/core/ItemTypes.h"

#include "arcane/cartesianmesh/ICartesianMeshPatch.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \ingroup ArcaneCartesianMesh
 * \brief Patch AMR d'un maillage cartésien.
 */
class ARCANE_CARTESIANMESH_EXPORT CartesianPatch
{
 public:

  //! Patch nul.
  CartesianPatch() = default;

  //! Patch nul.
  explicit CartesianPatch(ICartesianMeshPatch* patch_interface)
  : m_patch(patch_interface)
  {
  }
  CartesianPatch& operator=(const CartesianPatch&) = default;
  CartesianPatch& operator=(ICartesianMeshPatch* patch_interface)
  {
    m_patch = patch_interface;
    return (*this);
  }

 public:

  //! Groupe de mailles du patch (incluant les mailles de recouvrement).
  CellGroup cells() const;

  /*!
   * \brief Groupe de mailles du patch (sans les mailles de recouvrement).
   *
   * Valide uniquement avec l'AMR type 3 (PatchCartesianMeshOnly).
   */
  CellGroup inPatchCells() const;

  //! Index du patch dans le tableau des patchs.
  Integer index() const;

  /*!
   * \brief Niveau du patch.
   *
   * Valide uniquement avec l'AMR type 3 (PatchCartesianMeshOnly).
   */
  Integer level() const
  {
    ARCANE_CHECK_POINTER(m_patch);
    return m_patch->position().level();
  }

  //! Liste des mailles dans la direction \a dir
  CellDirectionMng& cellDirection(eMeshDirection dir)
  {
    ARCANE_CHECK_POINTER(m_patch);
    return m_patch->cellDirection(dir);
  }

  //! Liste des mailles dans la direction \a dir (0, 1 ou 2)
  CellDirectionMng& cellDirection(Integer idir)
  {
    ARCANE_CHECK_POINTER(m_patch);
    return m_patch->cellDirection(idir);
  }

  //! Liste des faces dans la direction \a dir
  FaceDirectionMng& faceDirection(eMeshDirection dir)
  {
    ARCANE_CHECK_POINTER(m_patch);
    return m_patch->faceDirection(dir);
  }

  //! Liste des faces dans la direction \a dir (0, 1 ou 2)
  FaceDirectionMng& faceDirection(Integer idir)
  {
    ARCANE_CHECK_POINTER(m_patch);
    return m_patch->faceDirection(idir);
  }

  //! Liste des noeuds dans la direction \a dir
  NodeDirectionMng& nodeDirection(eMeshDirection dir)
  {
    ARCANE_CHECK_POINTER(m_patch);
    return m_patch->nodeDirection(dir);
  }

  //! Liste des noeuds dans la direction \a dir (0, 1 ou 2)
  NodeDirectionMng& nodeDirection(Integer idir)
  {
    ARCANE_CHECK_POINTER(m_patch);
    return m_patch->nodeDirection(idir);
  }

  //! Effectue des vérifications sur la validité de l'instance.
  void checkValid() const
  {
    ARCANE_CHECK_POINTER(m_patch);
    m_patch->checkValid();
  }

  /*!
   * \brief Méthode permettant de récupérer la position du patch dans le
   * maillage cartesien.
   *
   * \return Une copie de la position.
   */
  AMRPatchPosition position() const
  {
    ARCANE_CHECK_POINTER(m_patch);
    return m_patch->position();
  }

  //! Indique si le patch est nul.
  bool isNull() const { return !m_patch; }

  //! Interface associée au patch (pour compatibilité avec l'existant)
  ICartesianMeshPatch* patchInterface() const { return m_patch; }

 private:

  ICartesianMeshPatch* m_patch = nullptr;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  

