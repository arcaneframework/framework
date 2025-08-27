// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* VariableRef.h                                               (C) 2000-2025 */
/*                                                                           */
/* Classe gérant une référence sur une variable.                             */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_VARIABLEREF_H
#define ARCANE_CORE_VARIABLEREF_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/String.h"

#include "arcane/core/IVariable.h"
#include "arcane/core/VariableComputeFunction.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class IModule;
class IVariableComputeFunction;
class VariableBuildInfo;
typedef VariableBuildInfo VariableBuilder;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \ingroup Variable
 * \brief Référence à une variable.
 *
 * Cette classe gère une référence sur une variable (IVariable).
 *
 * Si la variable n'est pas associée à un module, la méthode module() retourne 0.
 *
 * Cette classe doit obligatoirement être dérivée.
 *
 * La classe la plus dérivée de cette classe doit appeler _internalInit()
 * dans son constructeur. Elle seule doit le faire, et dans le constructeur
 * pour garantir que la référence à la variable est valide dès que
 * l'objet est construit et que les méthodes virtuelles qui doivent être appelés
 * lors de cette initialisation correspondent à l'instance en cours
 * de création.
 */
class ARCANE_CORE_EXPORT VariableRef
{
 public:

  class UpdateNotifyFunctorList;
  friend class UpdateNotifyFunctorList;

 protected:

  //! Construit une référence sur une variable avec les infos \a vbi
  explicit VariableRef(const VariableBuildInfo& vbi);
  //! Constructeur de copie
  VariableRef(const VariableRef& from);
  //! Construit une référence sur une variable \a var
  explicit VariableRef(IVariable* var);
  //! Opérateur de recopie
  VariableRef& operator=(const VariableRef& from);
  //! Constructeur vide
  VariableRef();

 public:

  //! Libère les ressources
  virtual ~VariableRef();

 public:

  //! Sous-domaine associé à la variable (TODO rendre obsolète fin 2023)
  ISubDomain* subDomain() const;

 public:

  //! Gestionnaire de variable associé à la variable.
  IVariableMng* variableMng() const;

  //! Nom de la variable
  String name() const;

 public:

  //TODO Supprimer virtual
  //! Type de la variable (Real, Integer, ...)
  virtual eDataType dataType() const;

  //! Affiche la valeur de la variable
  virtual void print(std::ostream& o) const;

  //TODO Supprimer virtual
  //! Module associé à la variable (ou nullptr, s'il n'y en a pas)
  virtual IModule* module() const { return m_module; }

  //TODO Supprimer virtual
  //! Propriétés de la variable
  virtual int property() const;

  //! Propriétés de la référence (interne)
  virtual int referenceProperty() const;

  //! Positionne la propriété \a property
  virtual void setProperty(int property);

  //! Supprime la propriété \a property
  virtual void unsetProperty(int property);

  //! Enregistre la variable (interne)
  virtual void registerVariable();

  //! Supprime l'enregistrement de la variable (interne)
  virtual void unregisterVariable();

  //! Variable associée
  IVariable* variable() const { return m_variable; }

  /*! \brief Vérifie si la variable est bien synchronisée.
   * \sa IVariable::checkIfSync()
   */
  virtual Integer checkIfSync(int max_print = 0);

  /*!
   * \brief Vérifie si la variable a les mêmes valeurs sur tous les réplicas.
   * \sa IVariable::checkIfSameOnAllReplica()
   */
  virtual Integer checkIfSameOnAllReplica(int max_print = 0);

  //! Mise à jour à partir de la partie interne
  virtual void updateFromInternal();

  //! Si la variable est un tableau, retourne sa dimension, sinon retourne 0
  virtual Integer arraySize() const { return 0; }

 public:

  void setUsed(bool v) { m_variable->setUsed(v); }
  bool isUsed() const { return m_variable->isUsed(); }

  virtual void internalSetUsed(bool /*v*/) {}

 public:

  /*!
   * \brief Pile d'appel au moment de l'assignation de cette instance.
   *
   * La pile n'est accessible qu'en mode vérification ou débug. Si
   * ce n'est pas le cas, retourne une chaîne nulle.
   */
  const String& assignmentStackTrace() const { return m_assignment_stack_trace; }

 public:

  //@{ @name Gestion des tags
  //! Ajoute le tag \a tagname avev la valeur \a tagvalue
  void addTag(const String& tagname, const String& tagvalue);
  /*! \brief Supprime le tag \a tagname
   *
   * Si le tag \a tagname n'est pas dans la liste, rien ne se passe.
   */
  void removeTag(const String& tagname);
  //! \a true si la variable possède le tag \a tagname
  bool hasTag(const String& tagname) const;
  //! Valeur du tag \a tagname. La chaîne est nulle si le tag n'existe pas.
  String tagValue(const String& tagname) const;
  //@}

 public:

  /*!
   * \name Gestion des dépendances
   *
   * Opérations liées à la gestion des dépendances des variables.
   */
  //@{
  /*! \brief Recalcule la variable si nécessaire
   *
   * Par le mécanisme de dépendances, cette opération est appelée récursivement
   * sur toutes les variables dont dépend l'instance. La fonction de recalcul
   * computeFunction() est ensuite appelée s'il s'avère qu'une des variables
   * dont elle dépend a été modifiée plus récemment.
   */
  void update();

  /*! \brief Indique que la variable vient d'être mise à jour.
   *
   * Pour une gestion correcte des dépendances, il faut que cette propriété
   * soit appelée toutes les fois où la mise à jour d'une variable a été
   * effectuée.
   */
  void setUpToDate();

  //! Temps auquel la variable a été mise à jour
  Int64 modifiedTime();

  //! Ajoute \a var à la liste des dépendances au temps courant
  void addDependCurrentTime(const VariableRef& var);

  //! Ajoute \a var à la liste des dépendances au temps courant avec les infos de trace \a tinfo
  void addDependCurrentTime(const VariableRef& var, const TraceInfo& tinfo);

  //! Ajoute \a var à la liste des dépendances au temps précédent
  void addDependPreviousTime(const VariableRef& var);

  //! Ajoute \a var à la liste des dépendances au temps précédent avec les infos de trace \a tinfo
  void addDependPreviousTime(const VariableRef& var, const TraceInfo& tinfo);

  /*! \brief Supprime \a var de la liste des dépendances
   */
  void removeDepend(const VariableRef& var);

  /*!
   * \brief Positionne la fonction de recalcule de la variable.
   *
   * Si une fonction de recalcule existait déjà, elle est détruite
   * et remplacée par celle-ci.
   */
  template <typename ClassType> void
  setComputeFunction(ClassType* instance, void (ClassType::*func)())
  {
    _setComputeFunction(new VariableComputeFunction(instance, func));
  }

  /*!
   * \brief Positionne la fonction de recalcule de la variable.
   *
   * Si une fonction de recalcule existait déjà, elle est détruite
   * et remplacée par celle-ci.
   * \a tinfo contient les infos permettant de savoir où est défini la fonction (pour le débug)
   */
  template <typename ClassType> void
  setComputeFunction(ClassType* instance, void (ClassType::*func)(), const TraceInfo& tinfo)
  {
    _setComputeFunction(new VariableComputeFunction(instance, func, tinfo));
  }
  //@}

 public:

  //! Référence précédente (ou null) sur variable()
  VariableRef* previousReference();

  //! Référence suivante (ou null) sur variable()
  VariableRef* nextReference();

  /*!
   * \internal
   * \brief Positionne la référence précédente.
   *
   * For internal use only.
   */
  void setPreviousReference(VariableRef* v);

  /*!
   * \internal
   * \brief Positionne la référence suivante.
   *
   * For internal use only.
   */
  void setNextReference(VariableRef* v);

 public:

  static void setTraceCreation(bool v);
  static bool hasTraceCreation();

 protected:

  void _setComputeFunction(IVariableComputeFunction* v);

  /*!
   * \brief Initialisation interne de la variable.
   *
   * \warning Cette méthode doit <strong>obligatoirement</strong> être
   * appelée dans le constructeur de la classe dérivée
   * avant toute utilisation de la référence.
   */
  void _internalInit(IVariable*);

  /*!
   * \brief Variable référencée.
   *
   * Cette méthode vérifie qu'une variable est bien référencée.
   */
  IVariable* _variable() const
  {
    _checkValid();
    return m_variable;
  }

 private:

  //! Variable associée
  IVariable* m_variable = nullptr;

  //! Module associé (ou 0 si aucun)
  IModule* m_module = nullptr;

  //! \a true si la variable a été enregistrée
  bool m_is_registered = false;

  //! Propriétés de la référence
  int m_reference_property = 0;

  //! Référence précédente sur \a m_variable
  VariableRef* m_previous_reference = nullptr;

  //! Référence suivante sur \a m_variable
  VariableRef* m_next_reference = nullptr;

  /*!
   * \brief Pile d'appel lors de l'assignation de la variable.
   *
   * Utilisé uniquement lorsque les traces sont actives.
   */
  String m_assignment_stack_trace;

 protected:

  void _executeUpdateFunctors();

  bool m_has_trace = false;

 private:

  void _checkValid() const
  {
#ifdef ARCANE_CHECK
    if (!m_variable)
      _throwInvalid();
#endif
  }
  void _throwInvalid() const;
  bool _checkValidPropertyChanged(int property);
  void _setAssignmentStackTrace();

 protected:

  void _internalAssignVariable(const VariableRef& var);

 private:

  static bool m_static_has_trace_creation;
  UpdateNotifyFunctorList* m_notify_functor_list = nullptr;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End variable Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  

//TODO: a supprimer quand tous les codes inclueront directement ce fichier
#include "arcane/core/VariableList.h"
