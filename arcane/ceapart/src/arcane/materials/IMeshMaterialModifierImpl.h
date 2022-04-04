﻿// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* IMeshMaterialModifierImpl.h                                 (C) 2000-2018 */
/*                                                                           */
/* Interface de l'implémentation de la modification des matériaux.           */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_MATERIALS_IMESHMATERIALMODIFIERIMPL_H
#define ARCANE_MATERIALS_IMESHMATERIALMODIFIERIMPL_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/ArrayView.h"

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
class IMeshMaterial;
class IMeshMaterialModifierImpl;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Interface de l'implémentation de la modification des matériaux.
 *
 * Cette classe ne doit en général pas être utilisée directement.
 * Il faut utiliser \a MeshMaterialModifier à la place. Cette dernière
 * prend en charge automatiquement les appels à beginUpdate() et endUpdate().
 */
class IMeshMaterialModifierImpl
{
 public:

  virtual ~IMeshMaterialModifierImpl(){}

 public:

  /*!
   * \brief Ajoute les mailles d'indices locaux \a ids au matériau \a mat.
   */
  virtual void addCells(IMeshMaterial* mat,Int32ConstArrayView ids) =0;

  /*!
   * \brief Supprime les mailles d'indices locaux \a ids au matériau \a mat.
   */
  virtual void removeCells(IMeshMaterial* mat,Int32ConstArrayView ids) =0;

  /*!
   * \brief Positionne les mailles d'indices locaux \a ids au matériau \a mat.
   *
   * Cette méthode n'est valide qu'en mode compatibilité
   * (IMeshMaterialMng::isCompatibilityMode())
   */
  virtual void setCells(IMeshMaterial* mat,Int32ConstArrayView ids) =0;

  /*!
   * \brief Indique qu'on commence une modification.
   */
  virtual void beginUpdate() =0;

  /*!
   * \brief Met à jour les structures après une modification.
   */
  virtual void endUpdate() =0;

  //! Affiche les statistiques sur les modifications
  virtual void dumpStats() =0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

MATERIALS_END_NAMESPACE
ARCANE_END_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  

