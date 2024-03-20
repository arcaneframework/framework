// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2024 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* MeshMaterialModifier.h                                      (C) 2000-2024 */
/*                                                                           */
/* Objet permettant de modifier les matériaux.                               */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_MATERIALS_MESHMATERIALMODIFIER_H
#define ARCANE_MATERIALS_MESHMATERIALMODIFIER_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/UtilsTypes.h"
#include "arcane/materials/MaterialsGlobal.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::Materials
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \ingroup ArcaneMaterials
 * \brief Objet permettant de modifier les matériaux ou les milieux.
 *
 * Cette classe fournit les méthodes pour modifier la liste des
 * mailles composant un matériaux ou un milieu.
 *
 * Les modifications se font directement sur les matériaux. Les milieux
 * correspondants sont automatiquement mis à jour. Il est possible
 * soit d'ajouter des mailles dans un matériaux (addCells())
 * soit d'en supprimer (removeCells()). Les modifications ne sont
 * prise en compte que lors de l'appel à endUpdate(). Cette dernière méthode
 * n'a pas besoin d'être appelée explicitement: elle l'est automatiquement
 * lors de l'appel au destructeur.
 * \todo ajouter exemple.
 */
class ARCANE_MATERIALS_EXPORT MeshMaterialModifier
{
 public:

  explicit MeshMaterialModifier(IMeshMaterialMng*);
  ~MeshMaterialModifier() ARCANE_NOEXCEPT_FALSE;

 public:

  /*!
   * \brief Ajoute les mailles d'indices locaux \a ids au matériau \a mat.
   */
  void addCells(IMeshMaterial* mat, SmallSpan<const Int32> ids);

  /*!
   * \brief Supprime les mailles d'indices locaux \a ids au matériau \a mat.
   */
  void removeCells(IMeshMaterial* mat, SmallSpan<const Int32> ids);

  /*!
   * \brief Met à jour les structures après une modification.
   *
   * Cette méthode est automatiquement appelée dans le destructeur de
   * l'instance si nécessaire.
   */
  void endUpdate();

 private:

  MeshMaterialModifierImpl* m_impl = nullptr;
  bool m_has_update = false;

  void _checkHasUpdate();
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::Materials

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
