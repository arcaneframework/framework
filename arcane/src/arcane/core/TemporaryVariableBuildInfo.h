// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* TemporaryVariableBuildInfo.h                                (C) 2000-2025 */
/*                                                                           */
/* Informations pour construire une variable temporaraire.                   */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_TEMPORARYVARIABLEBUILDINFO_H
#define ARCANE_CORE_TEMPORARYVARIABLEBUILDINFO_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/core/VariableBuildInfo.h"
#include "arcane/core/IVariableMng.h"
#include "arcane/core/ISubDomain.h"
#include "arcane/core/IModule.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \internal
 * \brief Paramètres nécessaires à la construction d'une variable temporaire.
 *
 * Une variable, même temporaire, doit être créée avec les mêmes paramètres
 * sur tous les sous-domaines.
 *
 * \warning Cette classe n'est pas encore opérationnelle
 */
class ARCANE_CORE_EXPORT TemporaryVariableBuildInfo
: public VariableBuildInfo
{
 public:

  /*!
   * \brief Construit un initialiseur pour une variable.
   *
   * \param name nom de la variable
   * \param m module associé
   */
  TemporaryVariableBuildInfo(IModule* m, const String& name);

  /*!
   * \brief Construit un initialiseur pour une variable sans l'associer à
   * un module.
   *
   * \param sub_domain gestionnaire de sous-domaine
   * \param name nom de la variable
   */
  TemporaryVariableBuildInfo(ISubDomain* sub_domain, const String& name);

  /*!
   * \brief Construit un initialiseur pour une variable.
   *
   * \param m module associé
   * \param name nom de la variable
   * \param item_family_name nom de la famille d'entité
   */
  TemporaryVariableBuildInfo(IModule* m, const String& name,
                             const String& item_family_name);

  /*!
   * \brief Construit un initialiseur pour une variable associée à un
   * maillage.
   *
   * \param sub_domain gestionnaire de sous-domaine
   * \param name nom de la variable
   */
  TemporaryVariableBuildInfo(IMesh* mesh, const String& name);

  /*!
   * \brief Construit un initialiseur pour une variable associée à un
   * maillage.
   *
   * \param sub_domain gestionnaire de sous-domaine
   * \param name nom de la variable
   * \param item_family_name nom de la famille d'entité
   */
  TemporaryVariableBuildInfo(IMesh* mesh, const String& name,
                             const String& item_family_name);

 protected:

  static int property();
  static String _generateName(IVariableMng* vm, const String& name);
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  
