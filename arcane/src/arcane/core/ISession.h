// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ISession.h                                                  (C) 2000-2025 */
/*                                                                           */
/* Interface d'une session.                                                  */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_ISESSION_H
#define ARCANE_CORE_ISESSION_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/core/ArcaneTypes.h"
#include "arcane/core/IBase.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Interface d'une session d'exécution d'un cas.
 *
 * Une session gère l'exécution d'un cas dans un processus.
 *
 * Cette exécution peut être répartie sur plusieurs sous-domaine en multi-threading.
 */
class ARCANE_CORE_EXPORT ISession
: public IBase
{
 public:

  virtual ~ISession() = default; //!< Libère les ressources

 public:

  //! Application
  virtual IApplication* application() const = 0;

  /*!
    \brief Créé un sous-domaine avec les paramètres contenus dans \a sdbi.
   
    Le sous-domaine créé est ajouté à la liste des sous-domaines de
    la session. Le document contenant le jeu de données est ouvert
    et et sa validité XML est vérifiée mais les options des services
    et modules ne sont pas lues.
  */
  virtual ISubDomain* createSubDomain(const SubDomainBuildInfo& sdbi) = 0;

  //! Termine la session avec le code de retour ret_val
  virtual void endSession(int ret_val) = 0;

  //! Liste des sous-domaines de la session
  virtual SubDomainCollection subDomains() = 0;

  //! Effectue un abort
  virtual void doAbort() = 0;

  /*!
   * \brief Vérifie que la version \a version du jeu de données est valide.
   *
   * \retval true si la version est valide
   * \retval false sinon   
   */
  virtual bool checkIsValidCaseVersion(const String& version) = 0;

  //! Écrit le fichier des informations sur l'exécution
  virtual void writeExecInfoFile() = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  

