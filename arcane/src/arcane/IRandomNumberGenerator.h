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

#ifndef ARCANE_IRANDOMNUMBERGENERATOR_H
#define ARCANE_IRANDOMNUMBERGENERATOR_H

#include "arcane/utils/Array.h"
#include "arcane/utils/UtilsTypes.h"

namespace Arcane
{

/**
 * @brief Classe représentant une graine.
 */
class ARCANE_CORE_EXPORT RandomNumberGeneratorSeed
{
 public:
  /**
   * @brief Constructeur avec graine de type T.
   * 
   * @tparam T Le type de graine (Int64 par ex).
   * @param seed La valeur de la graine.
   */
  template <class T>
  explicit RandomNumberGeneratorSeed(T seed)
  {
    setSeed(seed);
  }

  /**
   * @brief Constructeur copie.
   * 
   * @param seed La graine source.
   */
  RandomNumberGeneratorSeed(const RandomNumberGeneratorSeed& seed)
  {
    m_seed.resize(seed.sizeOfSeed());
    seed.seed(m_seed.data());
  }

  /**
   * @brief Constructeur par défaut.
   */
  RandomNumberGeneratorSeed()
  : m_seed(0)
  {}

  virtual ~RandomNumberGeneratorSeed() {}

 public:
  /**
   * @brief Méthode permettant de définir une graine.
   * 
   * @tparam T Le type de graine.
   * @param seed La valeur de la graine.
   * @return true Si la graine a pu être construite.
   * @return false Si la graine n'a pas pu être construite.
   */
  template <class T>
  bool setSeed(T seed)
  {
    Integer size = sizeof(T);
    m_seed.resize(size);
    memcpy(m_seed.data(), &seed, size);
    return true;
  }

  /**
   * @brief Méthode permettant de récupérer la valeur de la graine.
   * 
   * @tparam T Le type de la graine.
   * @param seed [OUT] La valeur de la graine.
   * @return true Si la graine a pu être placé.
   * @return false Si la graine n'a pas pu être placé.
   */
  template <class T>
  bool seed(T& seed) const
  {
    if (m_seed.size() == 0) {
      return false;
    }
    memcpy(&seed, dataPtr(), m_seed.size());
    return true;
  }

  /**
   * @brief Méthode permettant de récupérer la valeur de la graine.
   * 
   * @tparam T Le type de la graine.
   * @param seed [OUT] La valeur de la graine.
   * @return true Si la graine a pu être placé.
   * @return false Si la graine n'a pas pu être placé.
   */
  template <class T>
  bool seed(T* seed) const
  {
    if (m_seed.size() == 0) {
      return false;
    }
    memcpy(seed, dataPtr(), m_seed.size());
    return true;
  }

  /**
   * @brief Méthode permettant de récupérer la taille de la graine.
   * 
   * @return Integer La taille de la graine (en octet).
   */
  Integer sizeOfSeed() const
  {
    return m_seed.size();
  }

  /**
   * @brief Méthode permettant de récupérer un pointeur vers le tableau de Bytes (hors toute protection).
   * 
   * @return const Byte* Un pointeur vers le début du tableau.
   */
  const Byte* dataPtr() const
  {
    return m_seed.data();
  }

  /**
   * @brief Opérateur de copie.
   * 
   * @param other La graine source.
   * @return RandomNumberGeneratorSeed& La graine destination.
   */
  RandomNumberGeneratorSeed& operator=(const RandomNumberGeneratorSeed& other)
  {
    if (this == &other) {
      return *this;
    }

    m_seed.resize(other.sizeOfSeed());
    other.seed(m_seed.data());
    return *this;
  }

  /**
   * @brief Opérateur de copie depuis une valeur de graine.
   * 
   * @tparam T Le type de la graine.
   * @param other La valeur de la graine.
   * @return RandomNumberGeneratorSeed& La graine destination.
   */
  template <class T>
  RandomNumberGeneratorSeed& operator=(T other)
  {
    setSeed(other);
    return *this;
  }

  /**
   * @brief Opérateur de comparaison.
   * 
   * @param other La graine à comparer.
   * @return true Si les deux graines sont identiques.
   * @return false Si les deux graines ne sont pas identiques.
   */
  bool operator==(RandomNumberGeneratorSeed& other)
  {
    if (sizeOfSeed() != other.sizeOfSeed()) {
      return false;
    }

    const Byte* other_data = other.dataPtr();
    const Byte* my_data = dataPtr();

    for (Integer i = 0; i < sizeOfSeed(); i++) {
      if (my_data[i] != other_data[i]) {
        return false;
      }
    }
    return true;
  }

  /**
   * @brief Opérateur de comparaison.
   * 
   * @param other La graine à comparer.
   * @return true Si les deux graines ne sont pas identiques.
   * @return false Si les deux graines sont identiques.
   */
  bool operator!=(RandomNumberGeneratorSeed& other)
  {
    return !operator==(other);
  }

 protected:
  ByteUniqueArray m_seed;
};

/**
 * @ingroup StandardService
 * @brief Interface pour un générateur de nombre aléatoire.
 */
class ARCANE_CORE_EXPORT IRandomNumberGenerator
{
 public:
  virtual ~IRandomNumberGenerator() = default;

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
  virtual void initSeed(RandomNumberGeneratorSeed seed) = 0;

  /**
   * @brief Méthode permettant de récupérer la graine actuelle.
   * 
   * @return Int64 La graine.
   */
  virtual RandomNumberGeneratorSeed seed() = 0;

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
  virtual RandomNumberGeneratorSeed generateRandomSeed(Integer leap = 0) = 0;

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
  virtual RandomNumberGeneratorSeed generateRandomSeed(RandomNumberGeneratorSeed* parent_seed, Integer leap = 0) = 0;

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
  virtual Real generateRandomNumber(RandomNumberGeneratorSeed* seed, Integer leap = 0) = 0;
};
} // End namespace Arcane

#endif
