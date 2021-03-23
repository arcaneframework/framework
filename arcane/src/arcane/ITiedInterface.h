// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2021 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ITiedInterface.h                                            (C) 2000-2014 */
/*                                                                           */
/* Interface d'une classe gérant une semi-conformité du maillage.            */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_ITIEDINTERFACE_H
#define ARCANE_ITIEDINTERFACE_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/MultiArray2View.h"
#include "arcane/ArcaneTypes.h"
#include "arcane/TiedNode.h"
#include "arcane/TiedFace.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_BEGIN_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

typedef ConstMultiArray2View<TiedNode> TiedInterfaceNodeList;
typedef ConstMultiArray2View<TiedFace> TiedInterfaceFaceList;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \ingroup Mesh
 * \brief Interface d'un classe gérant une semi-conformité du maillage.
 */
class ITiedInterface
{
 public:

  virtual ~ITiedInterface() {} //<! Libère les ressources

 public:

  /*!
   * \brief Groupe contenant les faces maîtres.
   *
   * Il s'agit d'un groupe contenant uniquement les entités
   * propres à ce sous-domaine.
   */
  virtual FaceGroup masterInterface() const =0;

  //! Nom du groupe contenant les mailles maîtres
  virtual String masterInterfaceName() const =0;

  /*!
   * \brief Groupe contenant les faces esclaves.
   *
   * Il s'agit d'un groupe contenant uniquement les entités
   * propres à ce sous-domaine.
   */
  virtual FaceGroup slaveInterface() const =0;

  //! Nom du groupe contenant les mailles esclaves
  virtual String slaveInterfaceName() const =0;
  
  //! Liste des informations sur les noeuds esclaves d'une face maître
  virtual TiedInterfaceNodeList tiedNodes() const =0;

  //! Liste des informations sur les faces esclaves d'une face maître
  virtual TiedInterfaceFaceList tiedFaces() const =0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_END_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  

