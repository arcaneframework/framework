// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* FlexLMTools.h                                               (C) 2000-2025 */
/*                                                                           */
/* Gestion des protections FlexLM.                .                          */
/*---------------------------------------------------------------------------*/

#ifndef ARCANE_UTILS_FLEXLMTOOLS_H_
#define ARCANE_UTILS_FLEXLMTOOLS_H_

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/ArcaneVersion.h"
#include "arcane/utils/String.h"
#include "arcane/utils/Exception.h"
#include <map>
#include <iostream>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_BEGIN_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class IParallelSuperMng;
class TraceInfo;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

//! Exception de licence
class ARCANE_IMPL_EXPORT LicenseErrorException
: public Exception
{
 public:

  LicenseErrorException(const String& where);
  LicenseErrorException(const TraceInfo& where);
  LicenseErrorException(const String& where, const String& message);
  LicenseErrorException(const TraceInfo& where, const String& message);
  ~LicenseErrorException() ARCANE_NOEXCEPT {}

 public:

  virtual void explain(std::ostream& m) const;
  virtual void write(std::ostream& o) const;

 private:
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

//! FlexLM manager
/*! Singleton for recording all taken feature licenses
 * 
 *  Les contrôles sont effectués par le noeud maître (commRank==0)
 *  Pour vérifier la validité de ce contrôle soit la fonctionnalité ArcaneMasterFlexLM
 *  est disponible (ce qui ne locke pas les autres noeuds de l'exécution parallèle sur 
 *  le noeud de licence) ou que tous les noeuds ont l'autorisation de ArcaneCore. 
 *  Ceci est testé dans la phase init().
 */
class ARCANE_IMPL_EXPORT FlexLMMng
{
 private:

  //! Constructeur
  FlexLMMng();

  //! Destructeur
  virtual ~FlexLMMng() {}

 public:

  //! Accès au singleton
  static FlexLMMng* instance();

 public:

  //! Initialise le gestionnaire de licences
  void init(IParallelSuperMng* parallel_super_mng);

  //! Définit une nouvelle périodicité du contrôle des licences
  /*! La valeur par défaut est 120s.
   * si t == -1     : désactive le contrôle périodique
   * si 0 <= t < 30 : la valeur n'est pas prise en compte 
   * si t >= 30     : définit une nouvelle périodicité du contrôle
   */
  void setCheckInterval(const Integer t = 120);

  //! Teste la présence d'une fonctionnalité statique
  /*! Cette fonctionnalité n'utilisera pas de jeton de licence.
   * \param do_fatal indique s'il faut générer une erreur si non disponible
   * \return 0 si aucune erreur */
  bool checkLicense(const String name, const Real version, bool do_fatal = true) const;

  //! Demande l'allocation de \param nb_licenses licences pour la fonctionnalité \param name
  /*! Les licences demandées sont indépendantes du nombre de processeurs
   *  \param nb_licenses vaut par défaut 1
   *  \return 0 si aucune erreur */
  void getLicense(const String name, const Real version, Integer nb_licenses = 1);

  //! Relache les licences de la fonctionnalité \param name
  /*! \param nb_licenses vaut 0 s'il faut relacher toutes les licences
   *  \return 0 si aucune erreur */
  void releaseLicense(const String name, Integer nb_licenses = 0);

  //! Relache toutes les licences allouées
  /*! \return 0 si aucune erreur */
  void releaseAllLicenses();

  //! Return info on feature
  String featureInfo(const String name, const Real version) const;

 private:

  typedef std::map<String, Integer> FeatureMapType;
  FeatureMapType m_features;
  static FlexLMMng* m_instance;
  IParallelSuperMng* m_parallel_super_mng;
  bool m_is_master; //!< Cet host est il le maître des contrôles ?
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

//! Wrapper pour accéder aux FlexLMMng pour un jeu de fonctionnalités donné
template <typename FeatureModel>
class FlexLMTools
{
 public:

  //! Constructeur
  FlexLMTools() {}

  //! Destructeur
  virtual ~FlexLMTools() {}

 public:

  //! Teste la disponibilité d'une fonctionnalité
  /*! \return true si aucune erreur */
  bool checkLicense(typename FeatureModel::eFeature feature, const bool do_fatal) const
  {
    const String name = FeatureModel::getName(feature);
    const Real version = FeatureModel::getVersion(feature);
    return FlexLMMng::instance()->checkLicense(name, version, do_fatal);
  }

  //! Teste la disponibilité d'une fonctionnalité sur une version maximale
  /*! La version peut-être employée pour tester une quantité; ex: 3 pour 3 composantes maximum
   * \return true si aucune erreur */
  bool checkLicense(typename FeatureModel::eFeature feature, const Real version, const bool do_fatal) const
  {
    const String name = FeatureModel::getName(feature);
    return FlexLMMng::instance()->checkLicense(name, version, do_fatal);
  }

  //! Demande l'allocation de \param nb_licenses pour la fonctionnalité \param feature
  /*! \return 0 si aucune erreur */
  void getLicense(typename FeatureModel::eFeature feature, Integer nb_licenses = 1)
  {
    const String name = FeatureModel::getName(feature);
    const Real version = FeatureModel::getVersion(feature);
    return FlexLMMng::instance()->getLicense(name, version, nb_licenses);
  }

  //! Relache \param nb_licenses pour la fonctionnalité \param feature
  /*! \return 0 si aucune erreur */
  void releaseLicense(typename FeatureModel::eFeature feature, Integer nb_licenses = 0)
  {
    const String name = FeatureModel::getName(feature);
    return FlexLMMng::instance()->releaseLicense(name, nb_licenses);
  }

  //! Return info on feature
  String featureInfo(typename FeatureModel::eFeature feature) const
  {
    const String name = FeatureModel::getName(feature);
    const Real version = FeatureModel::getVersion(feature);
    return FlexLMMng::instance()->featureInfo(name, version);
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class ArcaneFeatureModel
{
 public:

  typedef enum
  {
#ifndef ARCANE_TEST_RLM
    ArcaneCore = 0, //<! Fonctionnalité noyau (liée à l'exécution)
#else
    Arcane = 0, //<! Fonctionnalité noyau (liée à l'exécution)
#endif
  } eFeature;

  static String getName(eFeature feature)
  {
    return m_arcane_feature_name[feature];
  }

  static Real getVersion(eFeature feature)
  {
    ARCANE_UNUSED(feature);
    // Ecrit une version comparable numériquement; ex: 1.0610 (au lieu de 1.6.1)
    return (Real)ARCANE_VERSION_MAJOR + (Real)ARCANE_VERSION_MINOR / 100 + (Real)ARCANE_VERSION_RELEASE / 1000 + (Real)ARCANE_VERSION_BETA / 10000;
  }

 private:

  static const String m_arcane_feature_name[];
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_END_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif /*FLEXLMTOOLS_H_*/
