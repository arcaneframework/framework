// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* IModuleMng.h                                                (C) 2000-2019 */
/*                                                                           */
/* Interface du gestionnaire des modules.                                    */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_IMODULEMNG_H
#define ARCANE_IMODULEMNG_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/


#include "arcane/core/ArcaneTypes.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class IModule;
class Msg;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Interface du gestionnaire de modules.
 * \ingroup Module
 */
class IModuleMng
{
 public:

  //! Destructeur
  /*! Libère les ressources */
  virtual ~IModuleMng() {}

 public:

  //! Ajoute le module \a m au gestionnaire
  virtual void addModule(Ref<IModule> m) =0;

  //! Supprime le module \a m
  virtual void removeModule(Ref<IModule> m) =0;

  //! Affiche la liste des modules du gestionnaire sur un flux \a o
  virtual void dumpList(std::ostream& o)   =0;

  //! Liste des modules
  virtual ModuleCollection modules() const =0;

  //! Supprime et détruit les modules gérés par ce gestionnaire
  virtual void removeAllModules() =0;

  //! Indique si le module de nom \a name est actif
  /*!
   * Si aucune module de nom \a name n'existe, retourne false.
   */  
  virtual bool isModuleActive(const String& name) =0;

  //! Retourne l'instance du module de nom \a name.
  /*!
   * Si aucune module de nom \a name n'existe, retourne 0.
   */  
  virtual IModule* findModule(const String& name) =0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  

