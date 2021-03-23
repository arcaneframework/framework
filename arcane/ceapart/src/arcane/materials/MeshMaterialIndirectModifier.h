// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2021 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* MeshMaterialIndirectModifier.h                              (C) 2000-2018 */
/*                                                                           */
/* Objet permettant de modifier indirectement les matériaux.                 */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_MATERIALS_MESHMATERIALINDIRECTMODIFIER_H
#define ARCANE_MATERIALS_MESHMATERIALINDIRECTMODIFIER_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/UtilsTypes.h"
#include "arcane/materials/MaterialsGlobal.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_BEGIN_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class IMesh;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

MATERIALS_BEGIN_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class IMeshMaterialMng;
class MeshMaterialBackup;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \ingroup ArcaneMaterials
 * \brief Objet permettant de modifier indirectement les matériaux ou les milieux.
 */
class ARCANE_MATERIALS_EXPORT MeshMaterialIndirectModifier
{
 public:

  MeshMaterialIndirectModifier(IMeshMaterialMng*);
  ~MeshMaterialIndirectModifier();

 public:

  /*!
   * \brief Prépare une modification.
   *
   */
  void beginUpdate();

  /*!
   * \brief Met à jour les structures après une modification.
   *
   * Cette méthode est automatiquement appelée dans le destructeur de
   * l'instance si nécessaire.
   */
  void endUpdate();

  /*!
   * \brief Met à jour les structures après une modification avec tri
   * préable des groupes de milieux et matériaux.
   *
   * Cette méthode est identique à endUpdate() mais garantit que
   * les groupes associés aux composants (IMeshComponent::cells())
   * seront triés par uniqueId() croissant à la fin de la mise à jour.
   */
  void endUpdateWithSort();

 private:

  IMeshMaterialMng* m_material_mng;
  MeshMaterialBackup* m_backup;
  bool m_has_update;

 private:

  void _endUpdate(bool do_sort);
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

MATERIALS_END_NAMESPACE
ARCANE_END_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  

