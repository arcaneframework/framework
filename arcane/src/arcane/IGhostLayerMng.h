// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2021 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* IGhostLayerMng.h                                            (C) 2000-2013 */
/*                                                                           */
/* Interface du gestionnaire de couche fantômes d'un maillage.               */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_IGHOSTLAYERMNG_H
#define ARCANE_IGHOSTLAYERMNG_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/ArcaneTypes.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_BEGIN_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \internal
 * Interface du gestionnaire de couche fantômes d'un maillage.
 */
class IGhostLayerMng
{
 public:

  //! Libère les ressources
  virtual ~IGhostLayerMng() {}

 public:

  //! Positionne le nombre de couches fantômes.
  virtual void setNbGhostLayer(Integer n) =0;

  //! Nombre de couches fantômes.
  virtual Integer nbGhostLayer() const =0;

  /*!
   * \brief Positionne la version du constructeur de mailles fantômes.
   * Pour l'instant (version 1.20), les valeurs possibles sont 1, 2 ou 3.
   * La valeur par défaut est 2 et la valeur 1 est obsolète. La valeur
   * 3 permet le support de plusieurs couches de mailles fantomes.
   */  
  virtual void setBuilderVersion(Integer n) =0;

  //! Version du constructeur de mailles fantômes.
  virtual Integer builderVersion() const =0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_END_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
