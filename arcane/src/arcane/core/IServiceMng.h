// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* IServiceMng.h                                               (C) 2000-2025 */
/*                                                                           */
/* Interface du gestionnaire des services.                                   */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_ISERVICEMNG_H
#define ARCANE_CORE_ISERVICEMNG_H
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
 * \brief Interface du gestionnaire de services.
 */
class IServiceMng
{
 public:

  virtual ~IServiceMng() = default; //!< Libère les ressources.

 public:

  //! Gestionnaire de trace associé
  virtual ITraceMng* traceMng() const =0;

  //! Ajoute une référence au service \a sv
  virtual void addSingletonInstance(SingletonServiceInstanceRef sv) =0;

  //! Retourne la liste des services singleton
  virtual SingletonServiceInstanceCollection singletonServices() const =0;

  /*!
   * Service singleton de nom \a name.
   *
   * Retourne une référence nulle si aucune instance de nom \a name n'existe.
   */
  virtual SingletonServiceInstanceRef singletonServiceReference(const String& name) const =0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  

