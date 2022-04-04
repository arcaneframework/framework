﻿// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* IItemConnectivityInfo.h                                     (C) 2000-2013 */
/*                                                                           */
/* Interface des informations sur la connectivité par type d'entité.         */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_IITEMCONNECTIVITYINFO_H
#define ARCANE_IITEMCONNECTIVITYINFO_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/ArcaneTypes.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_BEGIN_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \ingroup Mesh
 *
 * \brief Interface des informations sur la connectivité par type d'entité.
 *
 * Cette inferface permet de connaître pour un type d'entité donné
 * le nombre maximal d'entitée connectés. Cela peut être utilisé
 * par exemple pour dimensionner des variables.
 *
 * Les instances de cette interface se récupèrent généralement
 * via IItemFamily::localConnectivityInfos() pour les infos locales au
 * sous-domaine ou IItemFamily::globalConnectivityInfos() pour les
 * infos globales sur tous les maillages.
 */
class IItemConnectivityInfo
{
 public:

  virtual ~IItemConnectivityInfo() {} //<! Libère les ressources

 public:

  //! Nombre maximal de noeuds par entité
  virtual Integer maxNodePerItem() const =0;
  
  //! Nombre maximal d'arêtes par entité
  virtual Integer maxEdgePerItem() const =0;

  //! Nombre maximal de faces par entité
  virtual Integer maxFacePerItem() const =0;

  //! Nombre maximal de mailles par entité
  virtual Integer maxCellPerItem() const =0;

  //! Nombre maximal de noeuds pour les ItemTypeInfo de ce type d'entité
  virtual Integer maxNodeInItemTypeInfo() const =0;
  
  //! Nombre maximal d'arêtes pour les ItemTypeInfo de ce type d'entité
  virtual Integer maxEdgeInItemTypeInfo() const =0;

  //! Nombre maximal de faces pour les ItemTypeInfo de ce type d'entité
  virtual Integer maxFaceInItemTypeInfo() const =0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_END_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  
