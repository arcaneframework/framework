// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* IRandomNumberGenerator.h                                    (C) 2000-2022 */
/*                                                                           */
/* Interface pour générateur de nombres aléatoires.                          */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#ifndef IRANDOMNUMBERGENERATOR_H
#define IRANDOMNUMBERGENERATOR_H

#include <arcane/ItemTypes.h>

using namespace Arcane;

class IRandomNumberGenerator
{
 public:
  virtual ~IRandomNumberGenerator(){};

 public:
  /**
   * @brief Méthode permettant d'initialiser le service.
   * 
   * Avec la graine en option (ou la graine par défaut si l'on est
   * en mode singleton).
   * 
   */
  virtual void initSeed() = 0;

  /**
   * @brief Méthode permettant d'initialiser le service.
   * 
   * @param seed La graine d'origine.
   */
  virtual void initSeed(Int64 seed) = 0;

  /**
   * @brief Méthode permettant de récupérer la graine actuelle.
   * 
   * @return Int64 La graine.
   */
  virtual Int64 seed() = 0;

  /**
   * @brief Méthode permettant de savoir si les sauts sont permis sur le
   * générateur de graines.
   * 
   * @return true Si oui.
   * @return false Si non.
   */
  virtual bool isLeapSeedSupported() = 0;

  /**
   * @brief Méthode permettant de générer une graine "enfant" à partir d'une
   * graine "parent".
   * 
   * @param leap Le saut à effectuer (0 = la graine n+1+0 / 1 = la graine n+1+1).
   * @return Int64 La nouvelle graine généré à partir de la graine en mémoire.
   */
  virtual Int64 generateRandomSeed(Integer leap = 0) = 0;

  /**
   * @brief Méthode permettant de générer une graine "enfant" à partir d'une
   * graine "parent".
   * 
   * Cette méthode n'utilise pas la graine en mémoire.
   * 
   * @param parent_seed La graine "parent".
   * @param leap Le saut à effectuer (0 = la graine n+1+0 / 1 = la graine n+1+1).
   * @return Int64 La nouvelle graine généré à partir de la graine "parent".
   */
  virtual Int64 generateRandomSeed(Int64* parent_seed, Integer leap = 0) = 0;

  /**
   * @brief Méthode permettant de savoir si les sauts sont permis sur le
   * générateur de nombres.
   * 
   * @return true Si oui.
   * @return false Si non.
   */
  virtual bool isLeapNumberSupported() = 0;

  /**
   * @brief Méthode permettant de générer un nombre aléatoire avec
   * la graine en mémoire.
   * 
   * @param leap Le saut à effectuer (0 = le nombre n+1+0 / 1 = le nombre n+1+1).
   * @return Real Le nombre généré (entre 0 et 1).
   */
  virtual Real generateRandomNumber(Integer leap = 0) = 0;

  /**
   * @brief Méthode permettant de générer un nombre aléatoire avec
   * la graine transmise en paramètre.
   *    
   * Cette méthode n'utilise pas la graine en mémoire.
   * 
   * @param seed La graine.
   * @param leap Le saut à effectuer (0 = le nombre n+1+0 / 1 = le nombre n+1+1).
   * @return Real Le nombre généré (entre 0 et 1).
   */
  virtual Real generateRandomNumber(Int64* seed, Integer leap = 0) = 0;
};

#endif
