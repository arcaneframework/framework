// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* VariableFactoryRegisterer.h                                 (C) 2000-2025 */
/*                                                                           */
/* Singleton permettant d'enregister une fabrique de variable.               */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_VARIABLEFACTORYREGISTERER_H
#define ARCANE_CORE_VARIABLEFACTORYREGISTERER_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/core/VariableTypeInfo.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \internal
 * \brief Enregistreur d'une fabrique de variables.
 *
 Chaque instance de ce type doit être une variable globale qui référence
 une fonction de création de variable d'un type donnée. Chaque instance
 étant globale, sa création a lieu avant de rentrer dans le main(). Il ne
 faut donc faire aucune opération utilisant des objets externes ou même
 des allocations dynamiques car on ne connait pas les objets déjà créés

 La classe est concu pour que toutes ses instances soient liées par une
 liste chaînée. Le premier maillon étant récupérable par firstVariableFactory();
*/
class ARCANE_CORE_EXPORT VariableFactoryRegisterer
{
 public:

  using VariableFactoryFunc = VariableFactoryVariableRefCreateFunc;

 public:

 //! Crée un enregistreur pour une VariableFactory pour le type \a var_type_info et pour fonction de création \a func
  VariableFactoryRegisterer(VariableFactoryFunc func,const VariableTypeInfo& var_type_info);

 public:

  /*!
   * \brief Créé une fabrique pour ce type de variable.
   *
   * La fabrique doit être détruite par l'opérateur delete lorsqu'elle n'est
   * plus utilisée.
   */  
  IVariableFactory* createFactory();

  //! VariableFactory précédent (0 si le premier)
  VariableFactoryRegisterer* previousVariableFactory() const { return m_previous; }

  //! VariableFactory suivant (0 si le dernier)
  VariableFactoryRegisterer* nextVariableFactory() const { return m_next; }

  //! Genre des variables de données de la variable créée par cette fabrique
  eItemKind itemKind() const { return m_variable_type_info.itemKind(); }

  //! Type de données de la variable créée par cette fabrique
  eDataType dataType() const { return m_variable_type_info.dataType(); }

  //! Dimension de la variable créée par cette fabrique
  Integer dimension() const { return m_variable_type_info.dimension(); }

  //! Tag indiquant le type multiple (0 si non multiple, 1 si multiple, 2 si multiple deprecated)
  Integer multiTag() const { return m_variable_type_info.multiTag(); }

  //! indique si la fabrique est pour une variable partielle.
  bool isPartial() const { return m_variable_type_info.isPartial(); }

  //! Informations sur le type de la variable
  const VariableTypeInfo& variableTypeInfo() const { return m_variable_type_info; }

  /*!
   * \brief Positionne le VariableFactory précédent.
   *
   * Cette méthode est automatiquement appelée par IVariableFactoryRegistry.
   */
  void setPreviousVariableFactory(VariableFactoryRegisterer* s) { m_previous = s; }

  /*!
   *  \brief Positionne le VariableFactory suivant
   *
   * Cette méthode est automatiquement appelée par IVariableFactoryRegistry.
   */
  void setNextVariableFactory(VariableFactoryRegisterer* s) { m_next = s; }

 public:

  static VariableFactoryRegisterer* firstVariableFactory();
 
 private:

  //! Fonction de création du IVariableFactoryFactory
  VariableFactoryFunc m_function;

  //! Informations sur le type de la variable
  VariableTypeInfo m_variable_type_info;

  //! VariableFactory précédent
  VariableFactoryRegisterer* m_previous;

  //! VariableFactory suivant
  VariableFactoryRegisterer* m_next;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif

