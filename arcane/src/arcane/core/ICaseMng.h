// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ICaseMng.h                                                  (C) 2000-2025 */
/*                                                                           */
/* Interface de la classe gérant le jeu de données.                          */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_ICASEMNG_H
#define ARCANE_CORE_ICASEMNG_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/core/ArcaneTypes.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class ICaseMngInternal;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Types des évènements supportés par ICaseMng.
 *
 * Il est possible de s'enregistrer sur ces évènements via la méthode
 * ICaseMng::observable().
 */
enum class eCaseMngEventType
{
  //! Évènement généré avant de lire les options dans la phase 1
  BeginReadOptionsPhase1,
  //! Évènement généré avant de lire les options dans la phase 2.
  BeginReadOptionsPhase2
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \ingroup CaseOption
 * \brief Interface du gestionnaire de cas.
 *
 * Cette interface est gérée par un compteur de référence et ne doit pas
 * être détruite explictement.
 */
class ICaseMng
{
  ARCCORE_DECLARE_REFERENCE_COUNTED_INCLASS_METHODS();

 public:

  // TODO: rendre privé (début 2024)
  virtual ~ICaseMng() = default; //!< Libère les ressources

 public:

  //! Application associée
  virtual IApplication* application() = 0;

  //! Gestionnaire de traces
  virtual ITraceMng* traceMng() = 0;

  //! Gestionnaire de maillage associé
  virtual IMeshMng* meshMng() const = 0;

  //! Gestionnaire de sous-domaine.
  virtual ISubDomain* subDomain() = 0;

  //! Document XML du jeu de données (peut être nul si pas de jeu de donneés)
  virtual ICaseDocument* caseDocument() = 0;

  //! Fragment du Document XML associé au jeu de données (peut être nul si pas de jeu de donneés)
  virtual ICaseDocumentFragment* caseDocumentFragment() = 0;

  //! Système d'unité associé.
  virtual IPhysicalUnitSystem* physicalUnitSystem() const = 0;

  //! Lit le document XML du jeu de données.
  virtual ICaseDocument* readCaseDocument(const String& filename, ByteConstArrayView bytes) = 0;

  //! Lit les options du jeu de donnée correspondant aux modules utilisés
  virtual void readOptions(bool is_phase1) = 0;

  //! Affiche les valeurs des options
  virtual void printOptions() = 0;

  //! Lit les tables du jeu de donnée.
  virtual void readFunctions() = 0;

 public:

  //! Enregistre une liste d'options du jeu de donnée
  virtual void registerOptions(ICaseOptions*) = 0;

  //! Déseregistre une liste d'options du jeu de donnée
  virtual void unregisterOptions(ICaseOptions*) = 0;

  //! Collection des blocs d'options.
  virtual CaseOptionsCollection blocks() const = 0;

 public:

  //! Retourne la fonction de nom \a name ou \a nullptr s'il n'y en a pas.
  virtual ICaseFunction* findFunction(const String& name) const = 0;

  /*!
   * \brief Retourne la liste des tables.
   *
   * Le pointeur retourné n'est plus valide dès que la liste des tables change.
   */
  virtual CaseFunctionCollection functions() = 0;

  /*!
   * \brief Supprime une fonction.
   *
   * Supprime la fonction \a func. Si cette fonction n'est pas dans cette liste,
   * ne fait rien.
   * Si \a dofree est vrai, l'opérateur delete est appelé sur cette fonction.
   */
  ARCCORE_DEPRECATED_2019("Use removeFunction(ICaseFunction*) instead.")
  virtual void removeFunction(ICaseFunction* func, bool dofree) = 0;

  /*!
   * \brief Supprime une fonction.
   *
   * Supprime la fonction \a func. Si cette fonction n'est pas dans cette liste,
   * ne fait rien.
   */
  virtual void removeFunction(ICaseFunction* func) = 0;

  /*!
   * \brief Ajoute la fonction \a func.
   *
   * L'ajout ne peut se faire que lors de l'initialisation. L'appelant reste
   * propriétaire de l'instance \a func et doit l'enlever via removeFunction().
   */
  ARCCORE_DEPRECATED_2019("Use addFunction(Ref<ICaseFunction>) instead.")
  virtual void addFunction(ICaseFunction* func) = 0;

  /*!
   * \brief Ajoute la fonction \a func.
   *
   * L'ajout ne peut se faire que lors de l'initialisation.
   */
  virtual void addFunction(Ref<ICaseFunction> func) = 0;

  /*!
   * \brief Met à jour les options basée sur une table de marche en temps.
   *
   * Pour chaque option dépendant d'une table de marche, met à jour sa valeur
   * en utilisant le paramètre \a current_time s'il s'agit d'une table de
   * marche avec paramètre réel ou \a current_iteration s'il s'agit d'une
   * table de marche avec paramètre entier.
   * Si la fonction de l'option possède un coefficient ICaseFunction::deltatCoef()
   * non nul, le temps utilisé est égal à current_time + coef*current_deltat.
   *
   * \param current_time temps utilisé comme paramètre pour la fonction
   * \param current_deltat deltat utilisé comme paramètre pour la fonction
   * \param current_iteration itération utilisé comme paramètre pour la fonction
   */
  virtual void updateOptions(Real current_time, Real current_deltat, Integer current_iteration) = 0;

  /*!
   * \brief Positionne la manière de traiter les avertissements.
   * \sa isTreatWarningAsError().
   */
  virtual void setTreatWarningAsError(bool v) = 0;

  /*!
   * \brief Indique si les avertissements dans le jeu de données doivent être traités
   * comme des erreurs et provoquer l'arrêt du code.
   */
  virtual bool isTreatWarningAsError() const = 0;

  //! Positionne l'autorisation des éléments inconnus à la racine du document.
  virtual void setAllowUnkownRootElelement(bool v) = 0;

  //! Indique si les éléments inconnus à la racine du document sont autorisés
  virtual bool isAllowUnkownRootElelement() const = 0;

  /*!
   * \brief Observable sur l'instance.
   *
   * Le type de l'observable est donné par \a type
   */
  virtual IObservable* observable(eCaseMngEventType type) = 0;

 public:

  virtual Ref<ICaseMng> toReference() = 0;

 public:

  //! Implémentation interne
  virtual ICaseMngInternal* _internalImpl() = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  
