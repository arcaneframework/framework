// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2021 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* IMeshConnectivityInfo.h                                     (C) 2000-2007 */
/*                                                                           */
/* Informations sur la connectivité du maillage.                             */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_IMESHCONNECTIVITYINFO_H
#define ARCANE_IMESHCONNECTIVITYINFO_H
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
 * \brief Informations sur la connectivité du maillage.
 *
 * Ces informations sont calculées dynamiquement en fonction du maillage.
 */
class IMeshConnectivityInfo
{
 public:

  virtual ~IMeshConnectivityInfo() {} //<! Libère les ressources

 public:

 public:

  //! Nombre maximal de noeuds par maille
  virtual Integer maxNodePerCell() =0;
  
  //! Nombre maximal d'arêtes par maille
  virtual Integer maxEdgePerCell() =0;

  //! Nombre maximal de faces par maille
  virtual Integer maxFacePerCell() =0;

  //! Nombre maximal de noeuds par faces
  virtual Integer maxNodePerFace() =0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_END_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  
