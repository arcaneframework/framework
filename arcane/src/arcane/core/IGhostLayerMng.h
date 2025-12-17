// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* IGhostLayerMng.h                                            (C) 2000-2025 */
/*                                                                           */
/* Interface du gestionnaire de couches fantômes d'un maillage.              */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_IGHOSTLAYERMNG_H
#define ARCANE_CORE_IGHOSTLAYERMNG_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/core/ArcaneTypes.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

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
  virtual ~IGhostLayerMng() =default;

 public:

  //! Positionne le nombre de couches fantômes.
  virtual void setNbGhostLayer(Integer n) =0;

  //! Nombre de couches fantômes.
  virtual Integer nbGhostLayer() const =0;

  /*!
   * \brief Positionne la version du constructeur de mailles fantômes.
   * Pour l'instant (version 3.3), les valeurs possibles sont 2, 3 ou 4.
   * La valeur par défaut est 2. Les valeurs 3 et 4 permettent le support
   * de plusieurs couches de mailles fantômes.
   */  
  virtual void setBuilderVersion(Integer n) =0;

  //! Version du constructeur de mailles fantômes.
  virtual Integer builderVersion() const =0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
