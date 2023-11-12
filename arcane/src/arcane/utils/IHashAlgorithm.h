// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2023 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* IHashAlgorithm.h                                            (C) 2000-2023 */
/*                                                                           */
/* Interface d'un algorithme de hashage.                                     */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_UTILS_IHASHALGORITHM_H
#define ARCANE_UTILS_IHASHALGORITHM_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/UtilsTypes.h"

#include "arccore/base/Span.h"
#include "arccore/base/RefDeclarations.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Valeur retournée par un algorithme de hashage.
 */
class ARCANE_UTILS_EXPORT HashAlgorithmValue
{
 public:

  static constexpr Int32 MAX_SIZE = 64;

 public:

  SmallSpan<std::byte> bytes()
  {
    return { m_value.data(), m_size };
  }
  SmallSpan<const std::byte> bytes() const
  {
    return { m_value.data(), m_size };
  }
  SmallSpan<const Byte> asLegacyBytes() const
  {
    return { reinterpret_cast<const Byte*>(m_value.data()), m_size };
  }
  void setSize(Int32 size);

 private:

  std::array<std::byte, MAX_SIZE> m_value = {};
  Int32 m_size = MAX_SIZE;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Contexte pour calculer un hash de manière incrémentale.
 *
 * On peut utiliser un même contexte plusieurs fois en appelant reset() pour
 * réinitialiser l'instance.
 *
 * \code
 * IHashAlgorithm* algo = ...;
 * HashAlgorithmValue hash_value;
 * hash_value.reserve(algo->hashSize());
 * UniqueArray<std::byte> array1 = ...;
 * UniqueArray<std::byte> array2 = ...;
 *
 * Ref<IHashContext> context = algo->createContext();
 * context->updateHash(array1);
 * context->updateHash(array2);
 * context->computeHashValue(hash_value);
 * \endcode
 */
class ARCANE_UTILS_EXPORT IHashAlgorithmContext
{
 public:

  virtual ~IHashAlgorithmContext() = default;

 public:

  //! Réinitialise l'instance pour calculer une nouvelle valeur de hash.
  virtual void reset() = 0;

  //! Ajoute le tableau \a input au hash calculé
  virtual void updateHash(Span<const std::byte> input) = 0;

  //! Calcule la valeur de hashage et la retourne dans hash_value.
  virtual void computeHashValue(HashAlgorithmValue& hash_value) = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Interface d'un algorithme de hashage.
 */
class ARCANE_UTILS_EXPORT IHashAlgorithm
{
 public:

  virtual ~IHashAlgorithm() = default;

 public:

  //NOTE: pour l'instant (version 3.10) par encore virtuel pure pour rester
  // compatible avec l'existant
  //! Nom de l'algorithme
  virtual String name() const;

  //NOTE: pour l'instant (version 3.10) par encore virtuel pure pour rester
  // compatible avec l'existant. Envoi FatalErrorException si pas surchargée
  //! Taille (en octet) de la clé de hash.
  virtual Int32 hashSize() const;

  /*!
   * \brief Calcule la valeur du hash pour le tableau \a input.
   *
   * La valeur de hashage est <strong>ajoutée</string> dans \a output.
   * La longueur ajoutée est égale à hashSize().
   */
  virtual void computeHash64(Span<const Byte> input, ByteArray& output);

  /*!
   * \brief Calcule la valeur du hash pour le tableau \a input.
   *
   * La valeur de hashage est <strong>ajoutée</string> dans \a output.
   * La longueur ajoutée est égale à hashSize().
   */
  virtual void computeHash64(Span<const std::byte> input, ByteArray& output);

  //NOTE: pour l'instant (version 3.11) par encore virtuel pure pour rester
  // compatible avec l'existant
  /*!
   * \brief Créé un contexte pour calculer la valeur du hash
   * de manière incrémentale.
   *
   * Si l'implémentation ne supporte pas le mode incrémental (hasCreateContext()==false),
   * une exception est levée.
   */
  virtual Ref<IHashAlgorithmContext> createContext();

  //NOTE: pour l'instant (version 3.11) par encore virtuel pure pour rester
  // compatible avec l'existant
  //! Indique si l'implémentation supporte un hash incrémental
  virtual bool hasCreateContext() const { return false; }

 public:

  /*!
   * \brief Calcule la valeur du hash pour le tableau \a input.
   *
   * La valeur de hashage est <strong>ajoutée</string> dans \a output.
   * La longueur dépend de l'algorithme utilisé.
   */
  ARCANE_DEPRECATED_REASON("Y2023: Use computeHash64(Span<const std::byte> input,ByteArray& output) instead")
  virtual void computeHash(ByteConstArrayView input, ByteArray& output) = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  

