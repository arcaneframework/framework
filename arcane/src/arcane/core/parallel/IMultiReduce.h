// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* IMultiReduce.h                                              (C) 2000-2016 */
/*                                                                           */
/* Gestion de réductions multiples.                                          */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_PARALLEL_IMULTIREDUCE_H
#define ARCANE_PARALLEL_IMULTIREDUCE_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/Array.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_BEGIN_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Classe gérant une réduction d'une somme de valeur.
 *
 * Les instances de cette classe doivent être créés via IMultiReduce::getSumOfReal().
 * L'utilisateur doit accumuler les valeurs via l'appel à add(). Après exécution
 * de la réduction via IMultiReduce::execute(), il est possible
 * de récupérer la valeur réduite via reducedValue().
 * \sa IMultiReduce
 */
class ARCANE_CORE_EXPORT ReduceSumOfRealHelper
{
 public:

  ReduceSumOfRealHelper(bool is_strict)
  : m_reduced_value(0.0), m_is_strict(is_strict)
  {
    if (!m_is_strict)
      m_values.add(0.0);
  }

 public:

  //! Ajoute la valeur \a v
  void add(Real v)
  {
    if (m_is_strict)
      m_values.add(v);
    else
      m_values[0] += v;
  }
  
  //! Supprime les valeurs accumulées.
  void clear()
  {
    m_values.clear();
  }
  
  //! Liste des valeurs accumulées.
  RealConstArrayView values() const { return m_values; }
  
  //! Valeur réduite
  Real reducedValue() const { return m_reduced_value; }

  //! Positionne la valeur réduite.
  void setReducedValue(Real v) { m_reduced_value = v; }

 private:
  SharedArray<Real> m_values;
  Real m_reduced_value;
  bool m_is_strict;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Gestion de réductions multiples.
 *
 * Pour l'instant, seule les réductions de type 'somme' sur des réels
 * est supporté.
 *
 * Il est possible de spécifier un mode strict, via setStrict(), qui
 * permet que ces sommes soient identiques quelles que soit l'ordre
 * des opérations. Cela nécessite cependant de stocker toutes les valeurs
 * intermédiaire et donc est couteux en mémoire et non extensible car
 * un seul processeur se chargera du calcul de la somme.
 *
 * Le mode strict doit être spécifié avant la création des réductions.
 * Le mode strict est automatiquement actif si la variable d'environnement
 * ARCANE_STRICT_REDUCE est positionnée.
 */
class ARCANE_CORE_EXPORT IMultiReduce
{
 public:

  virtual ~IMultiReduce(){} //!< Libère les ressources

 public:

  static IMultiReduce* create(IParallelMng* pm);

 public:
  
  //! Exécute les réductions
  virtual void execute() =0;

  //! Indique si on utilise le mode strict
  virtual bool isStrict() const =0;

  //! Positionne le mode strict
  virtual void setStrict(bool is_strict) =0;

 public:

  /*!
   * \brief Retourne le gestionnaire de nom \a name.
   * S'il n'existe pas de gestionnaire de nom \a name il est créé.
   * L'objet retourné reste la propriété de cette instance et ne doit pas
   * être détruit explicitement. Il le sera lorsque cette instance sera
   * détruite.
   */
  virtual ReduceSumOfRealHelper* getSumOfReal(const String& name) =0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_END_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  

