// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* VersionInfo.h                                               (C) 2000-2018 */
/*                                                                           */
/* Informations sur une version d'un objet.                                  */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_UTILS_VERSIONINFO_H
#define ARCANE_UTILS_VERSIONINFO_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/ArcaneGlobal.h"

#include <iosfwd>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_BEGIN_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Informations sur une version.
 *
 Cette classe contient les informations sur une version d'un objet.
 Le numéro de version comprends 3 valeurs entières:

 \arg le numéro de version majeure,
 \arg le numéro de version mineure,
 \arg le numéro de version patch,

 Le numéro de version majeure correspondant à une évolution fondamentale
 de l'objet. Le numéro de version mineure correspondant à des évolutions
 moins importantes. Une évolution de la version majeure ou mineure
 suppose qu'on ne garde pas la compatibilité binaire.

 \note le numéro de sous-version n'est plus utilisé.
 */
class ARCANE_UTILS_EXPORT VersionInfo
{
 public:

  //! Construit une version nulle
  VersionInfo();

  //! Construit une informations de version
  VersionInfo(int vmajor,int vminor,int vpatch);

  /*! \brief Construit une informations de version
   * \a version_str doit être de la forme "M.m.p.b" avec \e M version majeure,
   * \m version mineure, \a p numéro de patch et \b numéro béta.
   */
  VersionInfo(const Arccore::String& version_str);

 public:
	
  //! Retourne le numéro de version majeur
  int versionMajor() const { return m_major; }
  //! Retourne le numéro de version mineur
  int versionMinor() const { return m_minor; }
  //! Retourne le numéro de version patch
  int versionPatch() const { return m_patch; }

  //! Numéro de version sous la forme d'une chaîne de caractères
  String versionAsString() const;

 public:

  // Imprime les numéros de version sur le flot \a o
  void write(std::ostream& o) const;

 private:

  int m_major; //!< Numéro de version majeur
  int m_minor; //!< Numéro de version mineur
  int m_patch; //!< Numéro de version patch
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

extern "C++" ARCANE_UTILS_EXPORT std::ostream&
operator<<(std::ostream& o,const VersionInfo& vi);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_END_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  
